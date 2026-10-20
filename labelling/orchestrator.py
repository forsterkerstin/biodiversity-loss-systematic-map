from __future__ import annotations

import argparse
import time
from pathlib import Path

from src import batching, batch_api, data_loader, tasks

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals_local.run import run_tasks
from src.config import (
    BATCH_SIZE,
    BATCHES_DIR,
    BATCH_OUTPUTS_DIR,
    DATASETS_DIR,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    LIMIT_DOCS,
    RUN_EVALS,
    SUBMISSION_MODE,
    ensure_artifact_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batching and OpenAI submission pipeline.")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing WoS export files (.xls, .xlsx, or .csv).",
    )
    parser.add_argument(
        "run_name",
        help="Name to tag dataset, batches, and outputs.",
    )
    parser.add_argument(
        "--task",
        dest="task",
        required=True,
        help="Classification task to run (e.g., screening, driver, threats_l0, geography, ecosystems_realm, study, taxa).",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Override the model (falls back to orchestrator.model in config).",
    )
    parser.add_argument(
        "--reasoning",
        dest="reasoning",
        default=None,
        help="Override the reasoning effort (falls back to orchestrator.reasoning in config).",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=None,
        help="Override the batch size (falls back to orchestrator.batch_size in config).",
    )
    return parser.parse_args()


def main() -> None:
    ensure_artifact_dirs()

    args = parse_args()
    input_dir: Path = args.input_dir
    run_name: str = args.run_name
    task_name: str = args.task
    task = tasks.get_task(task_name)

    limit_docs = LIMIT_DOCS
    batch_size = args.batch_size or BATCH_SIZE
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    model = args.model or DEFAULT_MODEL
    reasoning = args.reasoning or DEFAULT_REASONING_EFFORT
    run_evals = RUN_EVALS
    submission_mode = SUBMISSION_MODE
    if submission_mode not in {"live", "batch"}:
        raise ValueError("Config key orchestrator.submission_mode must be 'live' or 'batch'.")
    if submission_mode == "batch" and not task.supports_batch:
        raise ValueError(f"Task '{task.name}' only supports live submission.")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    df = data_loader.load_wos_excels(input_dir, limit=limit_docs)
    documents = data_loader.dataframe_to_documents(df)
    dataset_path = data_loader.save_dataset(documents, path=DATASETS_DIR / f"{run_name}-dataset.json")

    print(f"Loaded {len(documents)} documents from {input_dir}")
    print(f"Dataset saved to {dataset_path}")

    manifest = batching.write_batches(documents, batch_size=batch_size, run_name=run_name, output_dir=BATCHES_DIR)
    print(f"Wrote {len(manifest['batches'])} batches to {manifest['data_dir']}")

    prompts = batch_api.load_prompts()
    request_files: list[Path] = []
    for entry in manifest["batches"]:
        data_path = Path(entry.get("data_path") or entry["path"])
        docs = data_loader.load_jsonl(data_path)
        if entry.get("request_path"):
            raw_dest = Path(entry["request_path"])
            dest = raw_dest.parent / f"{task.name}-{raw_dest.name}"
        else:
            request_dir = Path(manifest["request_dir"])
            dest = request_dir / f"{task.name}-{data_path.stem}-requests.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        built_path = task.build_requests_file(
            docs,
            dest,
            prompts=prompts,
            model=model,
            reasoning_effort=reasoning,
        )
        request_files.append(built_path)
    print(f"Prepared {len(request_files)} request files in {manifest['request_dir']}")

    outputs_ready = False
    submitted: list[dict] = []
    client = batch_api.build_client()
    output_dir = BATCH_OUTPUTS_DIR / run_name

    non_empty: list[Path] = []
    for req_path in request_files:
        if req_path.stat().st_size == 0:
            print(f"[{task.name}] Skipping empty request file: {req_path.name}")
            submitted.append(batch_api.create_empty_output(req_path, output_dir))
        else:
            non_empty.append(req_path)
    request_files = non_empty

    if submission_mode == "live":
        submitted.extend(
            task.run_live(
                request_files,
                output_dir=output_dir,
                client=client,
                default_model=model,
                default_reasoning=reasoning,
                prompts=prompts,
            )
        )
        outputs_ready = True
    else:
        for req_path in request_files:
            info = batch_api.submit_batch(
                req_path,
                client=client,
                metadata=task.batch_metadata(run_name),
            )
            info["mode"] = "batch"
            info["request_path"] = str(req_path)
            info["task"] = task.name
            submitted.append(info)

    if submitted:
        print("Submission summary:")
        for entry in submitted:
            entry.setdefault("task", task.name)
            label = entry.get("task") or task.name
            mode = entry.get("mode") or "batch"
            if mode == "live":
                print(f"  [{label}] Live outputs for {entry.get('request_path')}: {entry.get('output_path')}")
            else:
                print(f"  [{label}] Batch submitted: id={entry.get('batch_id')}, request={entry.get('request_path')}")

    # Poll and download batch outputs
    batch_entries = [e for e in submitted if e.get("mode") == "batch" and e.get("batch_id")]
    if batch_entries:
        poll_interval = 90
        print(f"\nPolling {len(batch_entries)} batch(es) every {poll_interval}s until complete ...")
        pending = list(batch_entries)
        completed = []
        failed = []

        while pending:
            still_pending = []
            for entry in pending:
                status = batch_api.poll_batch(entry["batch_id"], client=client)
                state = status["status"]
                label = entry.get("task") or task.name

                if state == "completed":
                    print(f"  [{label}] Batch {entry['batch_id']} completed.")
                    entry["output_file_id"] = status.get("output_file_id")
                    completed.append(entry)
                elif state in ("failed", "cancelled", "expired"):
                    print(f"  [{label}] Batch {entry['batch_id']} {state}.")
                    failed.append(entry)
                else:
                    still_pending.append(entry)

            pending = still_pending
            if pending:
                print(f"  ... {len(pending)} batch(es) still running, waiting {poll_interval}s")
                time.sleep(poll_interval)

        if completed:
            output_dir = BATCH_OUTPUTS_DIR / run_name
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nDownloading {len(completed)} batch output(s) to {output_dir}")
            for entry in completed:
                label = entry.get("task") or task.name
                dest = output_dir / f"{label}-{entry['batch_id']}-output.jsonl"
                try:
                    batch_api.download_batch_output(entry["batch_id"], dest, client=client)
                    print(f"  [{label}] Saved {dest}")
                except Exception as exc:
                    print(f"  [{label}] Download failed: {exc}")
            outputs_ready = True

        if failed:
            print(f"\n{len(failed)} batch(es) failed/cancelled/expired:")
            for entry in failed:
                print(f"  [{entry.get('task')}] {entry['batch_id']}")

    if outputs_ready and run_evals:
        print(f"\nRunning evals for task '{task_name}' on run '{run_name}' ...")
        try:
            per_task_label_paths = None
            if task.name == "screening":
                per_task_label_paths = {task.name: str(input_dir)}
            summaries = run_tasks(
                run_name,
                tasks=[task.name],
                per_task_label_paths=per_task_label_paths,
            )
            for s in summaries:
                print(f"  [{s['task']}] rows={s['rows']} data={s['data_path']}")
                for truth, path in s["metric_paths"].items():
                    print(f"    metrics {truth}: {path}")
        except Exception as exc:
            print(f"  Evals failed: {exc}")


if __name__ == "__main__":
    main()

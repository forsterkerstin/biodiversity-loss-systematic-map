from __future__ import annotations

import colorsys

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_helpers.analysis.taxa import hierarchy as th
from data_helpers.analysis.taxa import hierarchy_plotting as thp


def _match(
    kingdom: str,
    phylum: str | None = None,
    taxon_class: str | None = None,
    order: str | None = None,
    *,
    domain: str = "Eukaryota",
    broad_group: str = "Other",
    eligible: bool = True,
) -> dict:
    lineage = [{"rank": "DOMAIN", "name": domain, "key": f"d-{domain}"}]
    if kingdom:
        lineage.append({"rank": "KINGDOM", "name": kingdom, "key": f"k-{kingdom}"})
    if phylum:
        lineage.append({"rank": "PHYLUM", "name": phylum, "key": f"p-{phylum}"})
    if taxon_class:
        lineage.append({"rank": "CLASS", "name": taxon_class, "key": f"c-{taxon_class}"})
    if order:
        lineage.append({"rank": "ORDER", "name": order, "key": f"o-{order}"})
    return {
        "broad_group": broad_group,
        "broad_group_eligible": eligible,
        "lineage": lineage,
    }


def test_research_paths_deduplicate_and_fractionalize_by_rank() -> None:
    class_a = _match(
        "Animalia",
        "Arthropoda",
        "Insecta",
        "Coleoptera",
        broad_group="Non-chordate animals",
    )
    class_b = _match(
        "Animalia",
        "Mollusca",
        "Bivalvia",
        "Venerida",
        broad_group="Non-chordate animals",
    )
    matches = pd.DataFrame(
        {
            "UT": ["u1", "u2", "u3"],
            "taxa_matches": [
                [class_a, class_a, class_b],
                [_match("Plantae", broad_group="Plants")],
                [_match("Fungi", "Ascomycota", "Sordariomycetes", eligible=False)],
            ],
        }
    )
    result = th.build_research_rank_attributions(
        matches,
        publication_ids=["u1", "u2", "u3"],
        benchmark_groups=["Non-chordate animals", "Plants", "Fungi", "Other"],
    )
    u1_class = result.loc[result["rank"].eq("class") & result["UT"].eq("u1")]
    assert len(u1_class) == 2
    assert set(u1_class["research_weight"]) == {0.5}
    u1_order = result.loc[result["rank"].eq("order") & result["UT"].eq("u1")]
    assert len(u1_order) == 2
    assert set(u1_order["research_weight"]) == {0.5}
    assert not result.loc[result["rank"].eq("phylum") & result["UT"].eq("u2")].size
    totals = result.groupby(["rank", "UT"])["research_weight"].sum()
    assert np.allclose(totals, 1.0)


def test_canonical_research_path_bridges_only_upper_containers() -> None:
    bacteria = _match(
        "Bacillati",
        "Bacillota",
        "Bacilli",
        domain="Bacteria",
    )
    assert th.canonical_research_path(bacteria) == (
        "Bacteria",
        "Bacillota",
        "Bacilli",
        None,
    )
    virus = _match("Orthornavirae", "Negarnaviricota", "Ellioviricetes")
    virus["lineage"].insert(1, {"rank": "REALM", "name": "Riboviria"})
    assert th.canonical_research_path(virus)[0] == "Viruses"


def test_stream_gbif_rank_counts_filters_and_reports_resolution(tmp_path) -> None:
    source = tmp_path / "taxon.tsv"
    pd.DataFrame(
        [
            {
                "taxonID": "1",
                "taxonRank": "species",
                "taxonomicStatus": "accepted",
                "kingdom": "Animalia",
                "phylum": "Arthropoda",
                "class": "Insecta",
                "order": "Coleoptera",
            },
            {
                "taxonID": "2",
                "taxonRank": "SPECIES",
                "taxonomicStatus": "ACCEPTED",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": None,
                "order": None,
            },
            {
                "taxonID": "3",
                "taxonRank": "species",
                "taxonomicStatus": "synonym",
                "kingdom": "Animalia",
                "phylum": "Arthropoda",
                "class": "Insecta",
                "order": "Coleoptera",
            },
            {
                "taxonID": "4",
                "taxonRank": "genus",
                "taxonomicStatus": "accepted",
                "kingdom": "Animalia",
                "phylum": "Arthropoda",
                "class": "Insecta",
                "order": "Coleoptera",
            },
        ]
    ).to_csv(source, sep="\t", index=False)
    counts, audit = th.stream_gbif_rank_counts(source, block_size=1024)
    totals = counts.groupby("rank")["gbif_described_species_count"].sum()
    assert totals.to_dict() == {
        "class": 1,
        "kingdom": 2,
        "order": 1,
        "phylum": 2,
    }
    accepted = audit.loc[
        audit["metric"].eq("Accepted species-rank rows"), "value"
    ].iloc[0]
    assert accepted == 2


def test_exact_comparison_rebalances_articles_after_common_path_filter() -> None:
    research = pd.DataFrame(
        [
            ["class", "u1", "Animalia", "Arthropoda", "Insecta", 1 / 3, 3],
            ["class", "u1", "Animalia", "Mollusca", "Bivalvia", 1 / 3, 3],
            ["class", "u1", "Animalia", "Chordata", "Teleostei", 1 / 3, 3],
            ["class", "u2", "Animalia", "Arthropoda", "Insecta", 1.0, 1],
        ],
        columns=[
            "rank",
            "UT",
            "kingdom",
            "phylum",
            "class",
            "research_weight",
            "n_article_paths",
        ],
    )
    gbif = pd.DataFrame(
        [
            ["class", "Animalia", "Arthropoda", "Insecta", 80],
            ["class", "Animalia", "Mollusca", "Bivalvia", 20],
        ],
        columns=[
            "rank",
            "kingdom",
            "phylum",
            "class",
            "gbif_described_species_count",
        ],
    )
    comparison = th.compare_exact_rank_paths(research, gbif)
    indexed = comparison.set_index("class")
    assert indexed.at["Insecta", "research_count"] == 1.5
    assert indexed.at["Bivalvia", "research_count"] == 0.5
    assert comparison["comparison_research_publications"].iloc[0] == 2
    assert np.isclose(comparison["research_share_pct"].sum(), 100)
    assert np.isclose(comparison["gbif_described_species_share_pct"].sum(), 100)


def test_order_sunburst_reconciles_and_keeps_other_terminal() -> None:
    comparison = pd.DataFrame(
        [
            ["Animalia", "Arthropoda", "Insecta", "Coleoptera", 45.0, 45],
            ["Animalia", "Arthropoda", "Insecta", "Hymenoptera", 45.0, 45],
            ["Animalia", "Arthropoda", "Insecta", "Diptera", 5.0, 5],
            ["Animalia", "Arthropoda", "Insecta", "Lepidoptera", 5.0, 5],
            ["Animalia", "Small phylum", "Small class", "Small order", 1.0, 1],
            ["Plantae", "Tracheophyta", "Magnoliopsida", "Rosales", 30.0, 25],
            ["Fungi", "Ascomycota", "Sordariomycetes", "Hypocreales", 20.0, 20],
            ["Chromista", "Ochrophyta", "Phaeophyceae", "Fucales", 2.0, 4],
        ],
        columns=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "research_count",
            "gbif_described_species_count",
        ],
    )
    comparison["comparison_research_publications"] = 153
    result = th.build_research_attention_sunburst(
        comparison,
        terminal_kingdoms=("Fungi",),
        phylum_min_within_parent_pct=2.0,
        class_min_within_parent_pct=2.0,
        order_max_children=2,
        order_max_other_within_parent_pct=5.0,
    )
    nodes = result.nodes
    roots = nodes.loc[nodes["parent_id"].isna()]
    assert np.isclose(roots["research_share_pct"].sum(), 100.0)
    assert np.isclose(roots["gbif_described_species_share_pct"].sum(), 100.0)
    for parent_id, children in nodes.dropna(subset=["parent_id"]).groupby(
        "parent_id"
    ):
        parent = nodes.set_index("node_id").loc[parent_id]
        assert np.isclose(children["research_count"].sum(), parent["research_count"])
        assert (
            children["gbif_described_species_count"].sum()
            == parent["gbif_described_species_count"]
        )
    other_ids = set(nodes.loc[nodes["is_other"], "node_id"])
    assert not nodes["parent_id"].isin(other_ids).any()
    insecta = nodes.loc[nodes["taxon"].eq("Insecta")].iloc[0]
    assert insecta["is_terminal"]
    assert "Other kingdoms" in set(nodes["taxon"])
    fungi = nodes.loc[nodes["taxon"].eq("Fungi")].iloc[0]
    assert fungi["is_terminal"]
    assert not nodes["parent_id"].eq(fungi["node_id"]).any()
    colors = thp.sunburst_node_colors(nodes)
    assert set(colors) == set(nodes["node_id"])
    target_lightness = {2: 0.58, 3: 0.70, 4: 0.83}
    target_saturation = {2: 0.64, 3: 0.52, 4: 0.40}
    for row in nodes.loc[nodes["depth"].gt(1)].itertuples():
        hue, lightness, saturation = colorsys.rgb_to_hls(
            *mcolors.to_rgb(colors[row.node_id])
        )
        assert 0 <= hue <= 1
        assert np.isclose(lightness, target_lightness[row.depth], atol=0.005)
        assert np.isclose(saturation, target_saturation[row.depth], atol=0.01)
    figure = thp.plot_research_attention_sunburst(nodes)
    assert figure.axes
    assert figure.texts == []
    assert np.allclose(
        figure.get_size_inches(),
        thp.NATURE_SUNBURST_FIGSIZE,
    )
    rendered_font_sizes = [
        text.get_fontsize()
        for axis in figure.axes
        for text in axis.texts
    ]
    assert rendered_font_sizes
    assert min(rendered_font_sizes) >= 5.0
    assert max(rendered_font_sizes) <= 7.0
    ring_bounds = figure._sunburst_ring_bounds
    assert np.isclose(ring_bounds[1][0], 0.44)
    assert np.allclose(
        [ring_bounds[depth][1] - ring_bounds[depth][0] for depth in range(1, 5)],
        [0.30, 0.34, 0.46, 0.72],
    )
    assert np.isclose(ring_bounds[4][1], 2.335)
    parent_ids = set(nodes["parent_id"].dropna().astype(str))
    terminal_above_order = nodes.loc[
        ~nodes["node_id"].astype(str).isin(parent_ids) & nodes["depth"].lt(4)
    ]
    expected_continuations = int((4 - terminal_above_order["depth"]).sum())
    continuations = figure._sunburst_terminal_continuations
    assert len(continuations) == expected_continuations
    assert {
        item["filled_rank"]
        for item in continuations
        if item["taxon"] == "Fungi"
    } == {"phylum", "class", "order"}
    assert all(item["plot_color"].lower() != "#ffffff" for item in continuations)
    plt.close(figure)


def test_large_stopped_class_combines_orders_small_on_both_global_measures() -> None:
    comparison = pd.DataFrame(
        [
            ["Animalia", "Chordata", "Mammalia", "Order A", 25.0, 25],
            ["Animalia", "Chordata", "Mammalia", "Order B", 20.0, 5],
            ["Animalia", "Chordata", "Mammalia", "Order C", 5.0, 5],
            ["Animalia", "Chordata", "Mammalia", "Order D", 5.0, 5],
            ["Plantae", "Tracheophyta", "Magnoliopsida", "Rosales", 40.0, 50],
            ["Fungi", "Ascomycota", "Sordariomycetes", "Hypocreales", 5.0, 10],
        ],
        columns=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "research_count",
            "gbif_described_species_count",
        ],
    )
    comparison["comparison_research_publications"] = 100
    result = th.build_research_attention_sunburst(
        comparison,
        terminal_kingdoms=("Fungi",),
        phylum_min_within_parent_pct=0.0,
        class_min_within_parent_pct=0.0,
        order_max_children=1,
        order_max_other_within_parent_pct=5.0,
        relaxed_order_expansion_min_class_research_share_pct=50.0,
        order_named_min_global_share_pct=10.0,
    )
    nodes = result.nodes
    mammalia_id = nodes.loc[nodes["taxon"].eq("Mammalia"), "node_id"].iloc[0]
    orders = nodes.loc[nodes["parent_id"].eq(mammalia_id)]
    assert set(orders["taxon"]) == {
        "Order A",
        "Order B",
        "Remaining Mammalia orders",
    }
    remaining = orders.loc[orders["is_order_remainder"]].iloc[0]
    assert remaining["is_other"]
    assert remaining["is_relaxed_order_remainder"]
    assert remaining["research_count"] == 10.0
    assert remaining["gbif_described_species_count"] == 10
    assert remaining["source_order_paths"] == 2
    assert thp._legend_label(remaining) == "Remaining orders"

    figure = thp.plot_research_attention_sunburst(
        nodes,
        legend_order_min_global_share_pct=10.0,
    )
    legend_text = {text.get_text() for axis in figure.axes[1:] for text in axis.texts}
    assert "Order B" in legend_text  # 20% research, despite only 5% GBIF.
    assert "Remaining orders" in legend_text
    assert "Order C" not in legend_text
    assert "Order D" not in legend_text
    assert not any(patch.get_hatch() for patch in figure.axes[0].patches)
    assert not any(
        patch.get_hatch() for axis in figure.axes[1:] for patch in axis.patches
    )
    plt.close(figure)


def test_compact_class_combines_hidden_named_and_residual_orders() -> None:
    comparison = pd.DataFrame(
        [
            ["Animalia", "Chordata", "Amphibia", "Anura", 19.0, 19],
            ["Animalia", "Chordata", "Amphibia", "Caudata", 7.0, 7],
            ["Animalia", "Chordata", "Amphibia", "Gymnophiona", 3.0, 3],
            ["Animalia", "Chordata", "Amphibia", "Minor order", 1.0, 1],
            ["Plantae", "Tracheophyta", "Magnoliopsida", "Rosales", 65.0, 65],
            ["Fungi", "Ascomycota", "Sordariomycetes", "Hypocreales", 5.0, 5],
        ],
        columns=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "research_count",
            "gbif_described_species_count",
        ],
    )
    comparison["comparison_research_publications"] = 100
    nodes = th.build_research_attention_sunburst(
        comparison,
        terminal_kingdoms=("Fungi",),
        phylum_min_within_parent_pct=0.0,
        class_min_within_parent_pct=0.0,
        order_max_children=3,
        order_max_other_within_parent_pct=5.0,
        order_named_min_global_share_pct=10.0,
    ).nodes
    amphibia_id = nodes.loc[nodes["taxon"].eq("Amphibia"), "node_id"].iloc[0]
    orders = nodes.loc[nodes["parent_id"].eq(amphibia_id)]
    assert set(orders["taxon"]) == {"Anura", "Remaining Amphibia orders"}
    remaining = orders.loc[orders["is_order_remainder"]].iloc[0]
    assert remaining["is_other"]
    assert not remaining["is_relaxed_order_remainder"]
    assert remaining["research_count"] == 11.0
    assert remaining["gbif_described_species_count"] == 11
    assert remaining["source_order_paths"] == 3
    assert thp._legend_label(remaining) == "Remaining orders"


def test_independent_gbif_benchmark_preserves_research_and_audits_remainders() -> None:
    comparison = pd.DataFrame(
        [
            ["Animalia", "P", "C1", "O1", 60.0, 50],
            ["Plantae", "T", "M", "R", 30.0, 25],
            ["Fungi", "A", "S", "H", 5.0, 20],
            ["Chromista", "X", "Y", "Z", 5.0, 5],
        ],
        columns=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "research_count",
            "gbif_described_species_count",
        ],
    )
    comparison["comparison_research_publications"] = 100
    selected = th.build_research_attention_sunburst(
        comparison,
        terminal_kingdoms=("Fungi",),
        phylum_min_within_parent_pct=0.0,
        class_min_within_parent_pct=0.0,
        order_named_min_global_share_pct=0.0,
    )
    original = selected.nodes.set_index("node_id")

    hierarchy_rows = [
        ["kingdom", "Animalia", None, None, None, 100],
        ["kingdom", "Plantae", None, None, None, 20],
        ["kingdom", "Fungi", None, None, None, 10],
        ["kingdom", "Chromista", None, None, None, 5],
        ["phylum", "Animalia", "P", None, None, 90],
        ["phylum", "Animalia", "Q", None, None, 10],
        ["phylum", "Plantae", "T", None, None, 18],
        ["phylum", "Fungi", "A", None, None, 8],
        ["phylum", "Chromista", "X", None, None, 5],
        ["class", "Animalia", "P", "C1", None, 50],
        ["class", "Animalia", "P", "C2", None, 20],
        ["class", "Animalia", "P", "C3", None, 10],
        ["class", "Animalia", "Q", "QC", None, 10],
        ["class", "Plantae", "T", "M", None, 18],
        ["order", "Animalia", "P", "C1", "O1", 20],
        ["order", "Animalia", "P", "C1", "O2", 10],
        ["order", "Plantae", "T", "M", "R", 18],
    ]
    gbif_counts = pd.DataFrame(
        hierarchy_rows,
        columns=[
            "rank",
            "kingdom",
            "phylum",
            "class",
            "order",
            "gbif_described_species_count",
        ],
    )
    broad = pd.DataFrame(
        [
            ["Chordates", 10],
            ["Non-chordate animals", 90],
            ["Plants", 20],
            ["Fungi", 10],
            ["Other", 5],
            ["Unresolved", 7],
        ],
        columns=["broad_group", "described_species_count"],
    )
    result = th.apply_independent_gbif_benchmark(
        selected,
        gbif_counts,
        broad,
        phylum_min_within_parent_pct=2.0,
        class_min_within_parent_pct=2.0,
        order_named_min_global_share_pct=1.0,
    )
    nodes = result.nodes
    rebased_original = nodes.loc[nodes["node_id"].isin(original.index)].set_index(
        "node_id"
    )
    assert np.allclose(
        rebased_original.loc[original.index, "research_count"],
        original["research_count"],
    )
    assert nodes["comparison_gbif_species"].nunique() == 1
    assert nodes["comparison_gbif_species"].iloc[0] == 135
    assert nodes.loc[nodes["parent_id"].isna(), "gbif_described_species_count"].sum() == 135

    animalia_id = nodes.loc[nodes["taxon"].eq("Animalia"), "node_id"].iloc[0]
    animalia_children = nodes.loc[nodes["parent_id"].eq(animalia_id)]
    assert dict(
        zip(
            animalia_children["taxon"],
            animalia_children["gbif_described_species_count"],
        )
    ) == {"P": 90, "Remaining Animalia phyla": 10}
    p_id = nodes.loc[nodes["taxon"].eq("P"), "node_id"].iloc[0]
    p_children = nodes.loc[nodes["parent_id"].eq(p_id)]
    assert dict(
        zip(p_children["taxon"], p_children["gbif_described_species_count"])
    ) == {"C1": 50, "Remaining P classes": 40}
    c1_id = nodes.loc[nodes["taxon"].eq("C1"), "node_id"].iloc[0]
    c1_children = nodes.loc[nodes["parent_id"].eq(c1_id)]
    assert dict(
        zip(c1_children["taxon"], c1_children["gbif_described_species_count"])
    ) == {"O1": 20, "Remaining C1 orders": 30}
    assert nodes.loc[nodes["node_id"].str.contains("Remaining"), "research_count"].eq(
        0
    ).all()

    class_assignments = result.assignments.loc[
        result.assignments["parent_node_id"].eq(p_id)
    ].set_index("actual_taxon")
    assert class_assignments.at["C2", "display_taxon"] == "Remaining P classes"
    assert class_assignments.at["C3", "display_taxon"] == "Remaining P classes"
    assert class_assignments.at["[class unavailable]", "gbif_described_species_count"] == 10
    for parent_id, children in nodes.dropna(subset=["parent_id"]).groupby(
        "parent_id"
    ):
        parent = nodes.set_index("node_id").loc[parent_id]
        assert np.isclose(children["research_count"].sum(), parent["research_count"])
        assert (
            children["gbif_described_species_count"].sum()
            == parent["gbif_described_species_count"]
        )


def test_sunburst_wedge_labels_use_angle_assets_and_rendered_fit(
    tmp_path, monkeypatch
) -> None:
    comparison = pd.DataFrame(
        [
            ["Animalia", "Chordata", "Mammalia", "Primates", 55.0, 55],
            ["Plantae", "Tracheophyta", "Magnoliopsida", "Rosales", 40.0, 40],
            ["Fungi", "Ascomycota", "Sordariomycetes", "Hypocreales", 5.0, 5],
        ],
        columns=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "research_count",
            "gbif_described_species_count",
        ],
    )
    comparison["comparison_research_publications"] = 100
    nodes = th.build_research_attention_sunburst(
        comparison,
        terminal_kingdoms=("Fungi",),
        phylum_min_within_parent_pct=2.0,
        class_min_within_parent_pct=2.0,
        order_max_children=2,
        order_max_other_within_parent_pct=5.0,
    ).nodes
    layout = thp._sunburst_layout(nodes)
    candidates = thp._sunburst_label_candidates(
        layout,
        clipart_by_taxon={
            "Animalia": "animal",
            "Plantae": "plant",
            "Fungi": "fungus",
        },
        min_angle_deg=18.0,
    )
    assert set(candidates["taxon"]) == {"Animalia", "Plantae", "Fungi"}

    from PIL import Image

    icon_path = tmp_path / "icon.png"
    Image.new("LA", (32, 32), color=(0, 255)).save(icon_path)
    monkeypatch.setattr(
        thp.taxa_clipart,
        "ensure_clipart_cached",
        lambda mapping: {
            "animal": icon_path,
            "plant": icon_path,
            "fungus": icon_path,
        },
    )
    figure = thp.plot_research_attention_sunburst(
        nodes,
        mapping={
            "sunburst_clipart": {
                "Animalia": "animal",
                "Plantae": "plant",
                "Fungi": "fungus",
            },
            "sunburst_order_clipart": {
                "Primates": "animal",
                "Rosales": "plant",
            },
        },
        in_wheel_label_min_angle_deg=18.0,
        legend_show_orders=False,
    )
    # Fungi clears the angular prefilter exactly, but its short inner-ring arc
    # cannot contain the complete icon/name pair and is therefore omitted.
    assert set(figure._sunburst_labelled_taxa) == {"Animalia", "Plantae"}
    assert set(figure._sunburst_class_labelled_taxa) == {"Mammalia"}
    assert set(figure._sunburst_order_labelled_taxa) == {"Primates", "Rosales"}
    # Omitting duplicate Order rows from the lower hierarchy gives the wheel
    # enough final-size area for both configured icon/name landmarks.
    assert set(figure._sunburst_order_icon_taxa) == {"Primates", "Rosales"}
    legend_text = {
        text.get_text()
        for axis in figure.axes[1:]
        for text in axis.texts
    }
    assert {"Animalia", "Chordata", "Mammalia"}.issubset(legend_text)
    assert {"Primates", "Rosales", "Hypocreales"}.isdisjoint(legend_text)
    ring_bounds = figure._sunburst_ring_bounds
    rendered_widths = {
        depth: top - bottom
        for depth, (bottom, top) in ring_bounds.items()
    }
    # Reallocation shrinks the centre while preserving the complete radial
    # extent used by the preceding 0.54-radius + four 0.43-ring geometry.
    assert np.isclose(ring_bounds[1][0] + sum(rendered_widths.values()), 2.26)
    assert rendered_widths[1] < rendered_widths[2] < rendered_widths[3]
    assert rendered_widths[4] > rendered_widths[3]
    polar_axis = figure.axes[0]
    label_ids = {artist.get_gid() for artist in polar_axis.texts}
    assert any(str(gid).startswith("sunburst-label-Animalia-") for gid in label_ids)
    assert any(str(gid).startswith("sunburst-label-Plantae-") for gid in label_ids)
    assert "sunburst-class-label-Mammalia" in label_ids
    assert "sunburst-order-label-Primates" in label_ids
    assert "sunburst-order-label-Rosales" in label_ids
    icon_ids = {artist.get_gid() for artist in polar_axis.artists}
    assert "sunburst-order-icon-Primates" in icon_ids
    assert "sunburst-order-icon-Rosales" in icon_ids
    rank_indents = list(thp.SUNBURST_LEGEND_RANK_INDENT.values())
    assert np.allclose(np.diff(rank_indents), rank_indents[1] - rank_indents[0])
    assert all(
        -90 <= thp._sunburst_curved_glyph_rotation(theta, 1.0) <= 90
        for theta in np.linspace(-np.pi / 2, np.pi / 2, 25)
    )
    assert all(
        -90 <= thp._sunburst_curved_glyph_rotation(theta, -1.0) <= 90
        for theta in np.linspace(np.pi / 2, 3 * np.pi / 2, 25)
    )
    assert all(
        -90 <= thp._sunburst_radial_text_rotation(theta) <= 90
        for theta in np.linspace(0, 2 * np.pi, 49)
    )
    plt.close(figure)


def _order_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kingdom": ["Animalia", "Animalia", "Animalia", "Plantae"],
            "phylum": ["Chordata", "Chordata", "Arthropoda", "Tracheophyta"],
            "class": ["Mammalia", "Aves", "Insecta", "Magnoliopsida"],
            "order": ["Rodentia", "Passeriformes", "Diptera", "Fabales"],
        }
    )


def test_distinct_path_taxa_counts_each_rank_along_compared_paths() -> None:
    counts = th.distinct_path_taxa(_order_comparison(), rank="order")
    assert counts.set_index("rank")["n_taxa"].to_dict() == {
        "kingdom": 2,
        "phylum": 3,
        "class": 4,
        "order": 4,
    }


def test_publications_on_common_paths_keeps_only_compared_paths() -> None:
    attributions = pd.DataFrame(
        {
            "UT": ["u1", "u2", "u3", "u3"],
            "rank": ["order", "order", "order", "class"],
            "kingdom": ["Animalia", "Animalia", "Fungi", "Animalia"],
            "phylum": ["Chordata", "Arthropoda", "Ascomycota", "Chordata"],
            "class": ["Mammalia", "Insecta", "Sordariomycetes", "Mammalia"],
            "order": ["Rodentia", "Diptera", "Hypocreales", None],
        }
    )
    retained = th.publications_on_common_paths(
        attributions, _order_comparison(), rank="order"
    )
    assert sorted(retained) == ["u1", "u2"]


def test_distinct_publication_labels_skips_special_values_and_filters_ids() -> None:
    publications = pd.DataFrame(
        {
            "UT": ["u1", "u2", "u3"],
            "taxa_species_labels": [
                np.array(["Mus musculus", "Not Applicable"]),
                ("Mus musculus", "Apis mellifera"),
                ["Quercus robur"],
            ],
            "taxa_genus_labels": [["Mus"], None, ["unclear"]],
        }
    )
    counts = th.distinct_publication_labels(
        publications,
        {"genus": "taxa_genus_labels", "species": "taxa_species_labels"},
        publication_ids=["u1", "u2"],
    ).set_index("rank")
    assert counts.loc["species", "n_taxa"] == 2
    assert counts.loc["genus", "n_taxa"] == 1
    assert counts.loc["species", "n_publications"] == 2

You are classifying STAGE_3: threat_l2.

The prior assistant messages contain already-decided parent threat_l0 and threat_l1 labels. Those prior outputs are context only.

# STAGE_3 RULES

- Select only from `threats_l2_candidates` or `Unclear`
- Refine the prior threat_l1 label(s) into one or more child labels from `threats_l2_candidates`
- Do not repeat or re-emit parent labels from L0 or L1 unless that exact same string appears in `threats_l2_candidates`
- Treat `threats_l2_candidates` as the full permitted label universe for this stage
- If a broader parent label seems right but is not present in `threats_l2_candidates`, do not output it
- If no candidate fits confidently at this stage, return `["Unclear"]`

# STOP RULE

- STAGE_3 is terminal, so always set `stop_reason` to `"stop"`

# OUTPUT

Return only one valid JSON object:
`{"results":[...], "stop_reason":"stop"}`

Valid examples:
- `{"results":["Intentional use"], "stop_reason":"stop"}`
- `{"results":["Unclear"], "stop_reason":"stop"}`

Invalid examples:
- `{"results":["Agricultural & Forestry Effluents"], "stop_reason":"stop"}`
- `{"results":["Pollution"], "stop_reason":"stop"}`

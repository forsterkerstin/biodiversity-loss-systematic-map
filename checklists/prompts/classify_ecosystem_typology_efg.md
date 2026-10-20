You are classifying STAGE_3: ecosystem functional group (EFG).

The prior assistant messages contain already-decided parent Realm and Biome labels. Those prior outputs are context only.

# STAGE_3 RULES

- Select only from `EFG_CANDIDATES`, `Unclear`, or `Not Applicable`
- Refine the prior parent label(s) into one or more child labels from `EFG_CANDIDATES`
- Treat `EFG_CANDIDATES` as the full permitted label universe for this stage
- Do not repeat or re-emit Realm or Biome labels unless that exact same string appears in `EFG_CANDIDATES`
- If no real ecosystem setting is present, return `["Not Applicable"]`
- If the ecosystem setting exists but you cannot confidently map it to the provided EFG candidates, return `["Unclear"]`

Choose one EFG by default. Choose multiple only if distinct assemblages are central to the study rather than mentioned in passing.

# STOP RULE

- Final stage: `stop_reason` must always be `"stop"`

# OUTPUT

Return only one valid JSON object:
`{"results":[...], "stop_reason":"stop"}`

Valid examples:
- `{"results":["Large permanent freshwater lakes"], "stop_reason":"stop"}`
- `{"results":["Unclear"], "stop_reason":"stop"}`
- `{"results":["Not Applicable"], "stop_reason":"stop"}`

Invalid examples:
- `{"results":["Rivers and streams"], "stop_reason":"stop"}`
- `{"results":["Freshwater"], "stop_reason":"stop"}`
- `{"results":["Large permanent freshwater lakes"], "stop_reason":"continue"}`

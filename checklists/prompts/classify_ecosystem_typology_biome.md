You are classifying STAGE_2: ecosystem biome.

The prior assistant message contains already-decided parent Realm label(s). That prior output is context only.

# STAGE_2 RULES

- Select only from `BIOME_CANDIDATES`, `Unclear`, or `Not Applicable`
- Refine the prior parent Realm label(s) into one or more child labels from `BIOME_CANDIDATES`
- Treat `BIOME_CANDIDATES` as the full permitted label universe for this stage
- Do not repeat or re-emit Realm labels unless that exact same string appears in `BIOME_CANDIDATES`
- If no real ecosystem setting is present, return `["Not Applicable"]`
- If the ecosystem setting exists but you cannot confidently map it to the provided biome candidates, return `["Unclear"]`

Choose one biome by default. Choose multiple only if distinct biome settings are central to the study rather than mentioned in passing.

# STOP RULE

- If results are exactly `["Unclear"]` or exactly `["Not Applicable"]`, set `stop_reason` to `"stop"`
- Otherwise set `stop_reason` to `"continue"`

Keep the existing terminality policy:
- if the article does not provide enough ecological detail to justify safe refinement to EFG, return `stop_reason:"stop"` even when a valid Biome was identified

# OUTPUT

Return only one valid JSON object:
`{"results":[...], "stop_reason":"stop|continue"}`

Valid examples:
- `{"results":["Rivers and streams"], "stop_reason":"continue"}`
- `{"results":["Marine shelf"], "stop_reason":"continue"}`
- `{"results":["Unclear"], "stop_reason":"stop"}`
- `{"results":["Not Applicable"], "stop_reason":"stop"}`

Invalid examples:
- `{"results":["Freshwater"], "stop_reason":"continue"}`
- `{"results":["Marine"], "stop_reason":"continue"}`
- `{"results":["Unclear"], "stop_reason":"continue"}`

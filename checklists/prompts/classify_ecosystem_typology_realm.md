You are classifying STAGE_1: ecosystem realm.

# STAGE_1 RULES

Allowed Realm labels:
- `Terrestrial`
- `Subterranean`
- `Subterranean-Freshwater`
- `Subterranean-Marine`
- `Freshwater-Terrestrial`
- `Freshwater`
- `Freshwater-Marine`
- `Marine`
- `Marine-Terrestrial`
- `Marine-Freshwater-Terrestrial`
- `Not Applicable`
- `All realms`

Default to one Realm unless the article clearly centers multiple distinct realms.

Use transitional Realms only when the interface process is central:
- wetlands and hydroperiod for `Freshwater-Terrestrial`
- estuarine or brackish mixing for `Freshwater-Marine`
- intertidal or shoreline gradients for `Marine-Terrestrial`
- deltaic three-way interaction for `Marine-Freshwater-Terrestrial`
- groundwater exchange for `Subterranean-Freshwater`
- anchialine or marine cave mixing for `Subterranean-Marine`

If there is no real ecological setting, use `Not Applicable`.
If the article is explicitly global across the biosphere, use `All realms`.

# STOP RULE

- If results are exactly `["Not Applicable"]` or exactly `["All realms"]`, set `stop_reason` to `"stop"`
- Otherwise set `stop_reason` to `"continue"`

Keep the existing terminality policy:
- if the article does not provide enough ecological detail to justify safe refinement to Biome, return `stop_reason:"stop"` even when a valid Realm was identified

# OUTPUT

Return only one valid JSON object:
`{"results":[...], "stop_reason":"stop|continue"}`

Valid examples:
- `{"results":["Freshwater"], "stop_reason":"continue"}`
- `{"results":["Marine-Terrestrial"], "stop_reason":"continue"}`
- `{"results":["Not Applicable"], "stop_reason":"stop"}`
- `{"results":["All realms"], "stop_reason":"stop"}`

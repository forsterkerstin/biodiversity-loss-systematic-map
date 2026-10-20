You are screening scientific records for a systematic map on the direct anthropogenic drivers of biodiversity change.

# Input:

Text consisting of the title and abstract of a scientific article.

# Task:

Assign labels for the screening steps using the provided schema.

# Broader guidelines:

1. Use only the input text and the definitions provided in this prompt.

2. Do not use external facts about the specific study, species, ecosystem, place, or driver.

3. You may make cautious, text-grounded inferences when the input plausibly indicates an anthropogenic stressor, biodiversity-relevant ecological change, or a link between them.

4. The input does not need to use the exact words "biodiversity change" for Step 1 to be positive. Count biodiversity-relevant ecological change whenever the text reports change in a biological system at genetic, species/population, community, habitat, or ecosystem level.

5. Such ecological change may be expressed directly or indirectly through ecological condition, integrity, functioning, extent, degradation, fragmentation, recovery, or similar change in the focal biological system.

6. For each step (1-4), return all required fields defined by the schema. The `results` value must be one of: `1` = yes, `0` = no, `-1` = unclear.

7. When evidence is weak but text-grounded, prefer cautious inclusion over strict exclusion. Use `-1` only when the text does not support a reasonable judgment even after cautious inference.

8. Step 4 does not require definitive causal proof. Count a linkage whenever the abstract presents a direct anthropogenic driver or stressor as related to the biodiversity-relevant ecological change in a text-grounded way, whether explicitly or through cautious inference.

# CONCEPTUAL BASIS

## Biodiversity:

Biodiversity includes diversity within species, between species, and of ecosystems.

## Essential Biodiversity Variables (EBVs) and related ecological indicators

Biodiversity-relevant change may be reported through EBVs or similar ecological indicators rather than explicit biodiversity language. Such indicators can support Step 1 when they describe change in the focal biological system at genetic, species/population, community, habitat, or ecosystem level.

## Forms of biodiversity change

Biodiversity-relevant change may be reported in one or more of the following broad categories. These categories are not fully exhaustive and may occur together.

1. Genetic change: change in genetic diversity within a species or population, including variation in genes, alleles, or inherited traits among individuals.

2. Species change: change in the presence, abundance, balance, or composition of species or populations in an area.

3. Community change: change in the composition, structure, or organization of a biological community over time.

4. Ecosystem change: change in the condition, integrity, structure, functioning, extent, connectivity, degradation state, or recovery state of an ecosystem or habitat.

## Direct anthropogenic drivers or stressors:

These are human pressures or sources linked to the biodiversity-relevant ecological change reported in the input text.

They may be broader direct drivers as defined by IPBES or more specific human activity stressors. A proximate anthropogenic stressor counts even if it is not named explicitly, as long as the text supports that inference.

Examples of direct anthropogenic driver types include but are not limited to: residential and commercial development, agriculture and aquaculture, energy production and mining, transportation and service corridors, biological resource use, human intrusions and disturbance, natural system modifications, invasive and other problematic species/genes/diseases, pollution, and climate change and severe weather.

# SCREENING STEPS

## Step 1. Biodiversity change component

Does the input text report, assess, or clearly imply at least one biodiversity-relevant ecological change in the focal biological system?

Biodiversity-relevant ecological change may occur at one or more of the following levels:

- genetic level
- species or population level
- community level
- ecosystem or habitat level

The change does not need to be described using explicit biodiversity terminology. It may be expressed through ecological condition, integrity, functioning, degradation, fragmentation, recovery, extent, connectivity, or other system-level biological change.

Do not require explicit causal attribution for Step 1. Step 1 is only about whether a biodiversity-relevant change is reported.

Do not count purely abiotic environmental change on its own. Count indirect indicators only when the text frames them as change in the focal living system, habitat, or ecosystem condition rather than only as physical environmental variation or resource accounting.

For Step 1, `results` must match the evidence of biodiversity-relevant ecological change:

- use `step1.results = 1` when biodiversity-relevant ecological change is reported, assessed, or clearly implied at one or more of the levels listed above
- use `step1.results = 0` when no biodiversity-relevant ecological change is reported
- use `step1.results = -1` when a biological system is involved but the text does not support a reliable judgment about whether biodiversity-relevant ecological change is reported

Populate `step1.biodiversity_types` using the corresponding schema labels for any detected biodiversity change types: `genetic_change`, `species_change`, `community_change`, and `ecosystem_change`. When `step1.results` is `0` or `-1`, return `step1.biodiversity_types = []`.

Prefer `1` when the text provides a reasonable, text-grounded basis to infer ecological change in the focal biological system.

Prefer `-1` when ecological change is plausible but not supported clearly enough for a reliable judgment.

## Step 2. Direction of change

Does the input text report or plausibly indicate the direction of the biodiversity-relevant ecological change?

Direction may be stated explicitly or inferred cautiously from how the ecological change is framed in the text.

Use:

- negative when the text reports or clearly implies deterioration, decline, degradation, loss, reduction, fragmentation, collapse risk, mortality increase, or other worsening ecological change
- positive when the text reports or clearly implies improvement, increase, recovery, restoration, recolonization, or other beneficial ecological change
- mixed when both positive and negative changes are reported or implied across places, times, taxa, levels, or ecological measures
- unclear when biodiversity-relevant ecological change is present but the direction cannot be determined confidently
- none only when no biodiversity-relevant ecological change is reported, or when the study explicitly reports no ecological response/change

For Step 2, `results` must match `direction`:

- if `direction` is `negative`, `positive`, or `mixed`, then `step2.results = 1`
- if `direction` is `none`, then `step2.results = 0`
- if `direction` is `unclear`, then `step2.results = -1`

Any identified direction (`negative`, `positive`, or `mixed`) is a positive Step 2 finding and must not be returned with `step2.results = 0`.

When biodiversity-relevant ecological change is present, prefer `unclear` over `none` unless the text clearly indicates absence of change.

## Step 3. Direct anthropogenic driver or stressor

Does the input text mention or plausibly imply one or more direct anthropogenic drivers or stressors?

These may be broader direct drivers of biodiversity loss or more specific human activity sources, including but not limited to the examples described in the Conceptual Basis section. The examples are illustrative, not exhaustive, and the text does not need to match them exactly.

A direct anthropogenic driver or stressor may be indicated when the abstract identifies, evaluates, compares, attributes, or meaningfully frames a human pressure, activity, land use, resource use, disturbance, extraction, infrastructure, pollution source, introduced organism, climate-related anthropogenic pressure, or other human-caused stressor as acting on the focal biological system.

A direct anthropogenic driver or stressor does not need to be explicitly named if the abstract supports a cautious, text-grounded inference.

Do not require the abstract to provide a formal threat label, a detailed mechanism, or an exact match to the reference examples. Broad but meaningful anthropogenic pressure framing can still count.

Do not count human context that is only incidental, distant, or purely background. The driver or stressor should be presented as relevant to the focal biological system, not merely mentioned in passing.

For Step 3, `results` must match the evidence of a direct anthropogenic driver or stressor:

- use `step3.results = 1` when the text mentions or clearly implies one or more direct anthropogenic drivers or stressors acting on the focal biological system
- use `step3.results = 0` when no direct anthropogenic driver or stressor is mentioned or reasonably implied
- use `step3.results = -1` when human influence may be present but the text does not support a reliable identification of a direct anthropogenic driver or stressor

Prefer `1` when the text provides a reasonable, text-grounded basis to identify a direct anthropogenic driver or stressor.

Prefer `-1` over `0` when human influence appears relevant but remains too ambiguous for reliable driver identification.

## Step 4. Linkage

Does the input text report, assess, attribute, compare, discuss, or clearly imply a link between one or more direct anthropogenic drivers/stressors and the biodiversity-relevant ecological change(s)?

A positive Step 4 finding requires more than mere co-mention. Count Step 4 as positive when the abstract presents the anthropogenic pressure as related to, contributing to, threatening, affecting, explaining, shaping, or being associated with the ecological change in a text-grounded way.

The linkage may be explicit or implicit, and does not require definitive causal proof, formal attribution, or direct experimental testing.

Do not require the abstract to state the full causal chain in a single sentence. It is enough that the driver/stressor and ecological change are meaningfully connected in the framing, analysis, interpretation, or threat context of the abstract.

For Step 4, `step4.results` must match `step4.linkage`:

- if `step4.linkage` is `direct_primary_evidence`, `modelled`, `projected`, `assessed`, `plausibly_implied`, or `other_positive`, then `step4.results = 1`
- if `step4.linkage` is `none`, then `step4.results = 0`
- if `step4.linkage` is `unclear`, then `step4.results = -1`

For Step 4, choose `step4.linkage` as follows:

- choose `direct_primary_evidence` when the paper reports observed empirical evidence linking the anthropogenic driver/stressor to the biodiversity-relevant ecological change
- choose `modelled` when the link is inferred analytically or statistically rather than directly observed
- choose `projected` when future scenarios or forward-looking projections are the basis
- choose `assessed` when the abstract evaluates, reviews, synthesizes, discusses, or risk-assesses the relationship without directly testing it as primary evidence
- choose `plausibly_implied` when the abstract meaningfully relates the anthropogenic pressure and ecological change in a way that goes beyond background acknowledgement; the driver or stressor must be actively framed as part of the study's context, analysis, or threat framing, not merely noted as a distant possibility
- choose `other_positive` only when the abstract clearly supports a positive linkage finding (`step4.results = 1`), but the linkage does not fit `direct_primary_evidence`, `modelled`, `projected`, `assessed`, or `plausibly_implied` after careful consideration
- choose `unclear` when a direct anthropogenic driver/stressor and biodiversity-relevant ecological change may both be present, but the relationship between them is too weakly framed for a reliable positive judgment
- choose `none` only when there is no text-grounded basis for linking anthropogenic pressure and ecological change

Any identified positive linkage (`direct_primary_evidence`, `modelled`, `projected`, `assessed`, `plausibly_implied`, or `other_positive`) is a positive Step 4 finding and must not be returned with `step4.results = 0`.

Prefer a positive linkage label over `unclear` when the abstract meaningfully relates the anthropogenic pressure and ecological change, even if the linkage is implicit rather than formally tested.

Prefer `unclear` over `none` when both elements may be present but the abstract does not support a reliable linkage judgment.

Do not use `plausibly_implied` for mere co-mention without a meaningful relationship.

Do not use `other_positive` when any named positive linkage label is a reasonable fit.

# OUTPUT RULES

- Return only JSON matching the schema.
- `step1.results`, `step2.results`, `step3.results`, and `step4.results` must be one of: `1`, `0`, or `-1`.
- Always return all required fields defined by the schema.
- Array fields must always be returned as arrays. Use an empty array (`[]`) when no items apply.
- Categorical fields must always be returned using one of their allowed enum values.
- `step1.biodiversity_types` is a list of detected biodiversity change types.
- `step2.direction` is the detected direction of biodiversity-relevant ecological change.
- `step3.drivers` is a list of detected direct anthropogenic drivers or stressors.
- `step4.linkage` must be one of: `direct_primary_evidence`, `modelled`, `projected`, `assessed`, `plausibly_implied`, `other_positive`, `unclear`, or `none`.

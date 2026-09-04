# RSC-PF official iTransformer adaptation

This directory is reserved for the source snapshot used by the formal-v4
`Official iTransformer-PTO` baseline. The backbone must come from the official
THUML repository:

<https://github.com/thuml/iTransformer>

The required class is `model.iTransformer.Model`. The local checkout is frozen
at Git commit `c2426e68ca13f74aaec08045c5c724d8ad328124`; the preparation script
only verifies that checkout, hashes every imported Python file and the license,
and writes `protocol/ITRANSFORMER_SOURCE_RECEIPT.json` once. It never clones,
fetches, or overwrites source files, and a different Git HEAD is rejected.

The baseline is an **official-backbone adaptation**, not a claim of reproducing
the authors' complete benchmark pipeline. RSC-PF supplies the four-channel,
24-step input and four-step output contract required by the common PTO runner;
the surrounding normalization, rolling-window split, and LP dispatch wrapper are
ours and are disclosed separately. No local inverted-token forecaster or other
substitute may be used under this baseline name.

Training and model selection use only the formal-v4 training years (2015--2018)
and selection year (2019). The 2020 evaluation windows remain inaccessible
until Gate 3 authorization. The source receipt, imported-file hashes, commit,
license hash, and adaptation configuration must be archived with the experiment
outputs.

# PDAC PRR

Result-linked inference code for the extreme survival classification (ESC) model
and the pathology, pathology–text and full multimodal prognostic risk regression
(PRR) models. This release retains the architectures and checkpoint ensemble
that produced the existing research outputs. It provides local inference, input
preparation and an optional local upload interface.

Repository: [Docurdt/PDAC_PRR](https://github.com/Docurdt/PDAC_PRR).
中文使用与上传说明：[README_zh.md](README_zh.md).

## Quick start

Use Python 3.10–3.12 (local validation used Python 3.12 and PyTorch 2.5.1).
From the repository directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Model weights are separate Release attachments and are excluded from ordinary
Git commits. Install the PRRpath archive, then run the included synthetic example:

```sh
pdac-prr-weights --bundle . --install /path/to/PDAC_PRR_prrpath_weights_v0.1.0.zip
pdac-prr-weights --bundle . --model prrpath
pdac-prr --bundle . --model prrpath --input examples/synthetic_path_features.npz --output predictions/example.json
```

The example is random numeric data, without patient records. Expected scores
are supplied in `examples/synthetic_path_expected.json`. Predictions return a
score per row, five fold scores and the number of checkpoints used. PRR scores
rank relative prognosis; they are not calibrated survival probabilities or
predicted survival times. Use `--device cuda` when the selected PyTorch installation
and hardware support CUDA; CPU is the default.

## Models and feature inputs

All NPZ files are loaded with `allow_pickle=False`. Every row must correspond to
the same patient across modalities.

| Model name | Required numeric arrays |
| --- | --- |
| `esc` | `gigapth`: N×1024×1536; `hoptimus0`: N×1024×1536; `ctranspath`: N×1024×768; `uni_v2`: N×1024×1536 |
| `prrpath` | `ls_features`: N×256 |
| `prrpath_text` | `ls_features`: N×256; `text_features`: N×768 |
| `prrfull` | `ls_features`: N×256; `text_features`: N×768; `clin_features`: N×4 |

`gigapth` preserves the source checkpoint key spelling. The clinical order is
sex, normalized age, tumor site, tumor size. The full definitions, preprocessing
and score meanings are in [MODEL_CARD.md](docs/MODEL_CARD.md).

Inference verifies checkpoint sizes and SHA-256 hashes and loads state
dictionaries strictly. It averages five retained epoch predictions within each
inner fold, then averages all five folds. A selected family requires all 25
listed checkpoints. See `model_manifest.json` for the exact files.

The source package contains all four architectures. The first local weight
assets available for upload are ESC and PRRpath. PRRpath+text and PRRfull archives
are not complete in this first handoff; their checkpoint manifests are included.
Their ensemble cannot run until all corresponding weights have been installed.

## Prepare inputs from a whole-slide image

WSI processing requires the external TRIDENT checkout and foundation weights in
a separate compatible GPU environment. Upstream software/model licenses remain
in force; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This repository
does not vendor TRIDENT or redistribute foundation weights.

```sh
git clone https://github.com/mahmoodlab/TRIDENT.git trident
git -C trident checkout c9a390563bc37b490119eded5c1d83a17a9e3ba1
```

Install TRIDENT dependencies following that pinned checkout. Run this wrapper
using the TRIDENT environment's Python. Check commands first:

```sh
python scripts/prepare_wsi.py --trident trident --wsi-dir user_data/wsis --output-dir trident_processed --dry-run
```

Remove `--dry-run` to process the slides. Optional `--weights-json` supplies a
JSON mapping of encoder names (`gigapath`, `uni_v2`, `ctranspath`, `hoptimus0`) to
local pretrained checkpoint paths. Otherwise configure the pinned TRIDENT
checkout's supported model paths. The wrapper preserves segmentation confidence
0.7, zero overlap and the retained magnification/patch extraction settings.
Raw-slide processing with external encoders has not been run end to end in the
local release environment; the exact validation scope is in
[VALIDATION.md](docs/VALIDATION.md).

Install the ESC weight archive, then select a slide's four H5 feature files:

```sh
pdac-prr-weights --bundle . --install /path/to/PDAC_PRR_esc_weights_v0.1.0.zip
python scripts/prepare_inputs.py --bundle . --features-root trident_processed --slide-stem SELECTED_SLIDE --output user_data/prepared.npz
pdac-prr --bundle . --model prrpath --input user_data/prepared.npz --output predictions/slide.json
```

`--features-root` is the directory containing the `20x_256px_0px_overlap`,
`20x_224px_0px_overlap` and `10x_256px_0px_overlap` subdirectories for that slide.
The helper requires an explicitly selected slide and preserves sampling indices
and a preprocessing sidecar. `--sampling-indices` can reuse an existing NPZ of
encoder-specific indices.

For text, install `python -m pip install -e '.[text]'` and add `--report` pointing
to a UTF-8 report file. Optional `--bert-path` uses a local Chinese BERT directory.
For raw clinical variables, add `--clinical-cohort cohort.csv --case-row 0` with
columns `Sex`, `Age`, `Tumor site`, `Tumor size`. The supplied cohort determines
the age bounds and tumor-size imputation median, matching the retained evaluation
procedure. Do not use a single row as a newly invented clinical normalization
context; a fixed prospective single-patient scaler is not provided.

## Optional local upload interface

```sh
python -m pip install -e '.[ui]'
streamlit run app.py --server.address 127.0.0.1
```

Upload a prepared numeric NPZ file and download the resulting JSON scores. WSI,
report and clinical preparation use the workflow above. The interface runs
locally; GitHub Pages is not used.

## Validation and provenance

The original four model families passed strict loading of all 100 retained
checkpoints on the source server. Extracted score computations matched the
original definitions in synthetic comparisons. The local PRRpath 25-checkpoint
ensemble and CLI matched the original class with zero observed difference on the
included synthetic input. See [VALIDATION.md](docs/VALIDATION.md) and
`source_provenance.json` for the scoped evidence and source hashes. These checks
do not constitute a rerun of every patient prediction or clinical validation.

The repository contains inference code, public provenance, synthetic examples
and documentation. Private notebooks, manuscript files, clinical tables, raw
slides, patient feature caches, server paths and credentials are not included.

## License

Project source code and documentation use the MIT License. External dependencies
and pretrained models retain their own terms. Downstream weight distribution
terms are separate from this source-only license.

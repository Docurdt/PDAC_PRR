# PDAC PRR v0.1.0 — complete model weights

All four result-linked model families are now included: ESC, PRRpath,
PRRpath+text and PRRfull. Each family has 25 original checkpoints, for 100 in
total. All checkpoint sizes and SHA-256 hashes match the retained source manifest.

## Release assets

| Model | ZIP filename | Checkpoints |
| --- | --- | ---: |
| ESC | `PDAC_PRR_esc_weights_v0.1.0.zip` | 25 |
| PRRpath | `PDAC_PRR_prrpath_weights_v0.1.0.zip` | 25 |
| PRRpath+text | `PDAC_PRR_prrpath_text_weights_v0.1.0_part1.zip` | 15, folds 1–3 |
| PRRpath+text | `PDAC_PRR_prrpath_text_weights_v0.1.0_part2.zip` | 10, folds 4–5 |
| PRRfull | `PDAC_PRR_prrfull_weights_v0.1.0_part1.zip` | 15, folds 1–3 |
| PRRfull | `PDAC_PRR_prrfull_weights_v0.1.0_part2.zip` | 10, folds 4–5 |

Each ZIP is independently installable; do not concatenate the parts.
Use the matching asset names from `SHA256SUMS_all_weights.txt` to check downloads.
GitHub Release assets must each be under 2 GiB:
[GitHub documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Installation

Install the repository environment following the README. To install both
multimodal model families from downloaded ZIPs, run in the repository root:

```sh
pdac-prr-weights --bundle . --install PDAC_PRR_prrpath_text_weights_v0.1.0_part1.zip
pdac-prr-weights --bundle . --install PDAC_PRR_prrpath_text_weights_v0.1.0_part2.zip
pdac-prr-weights --bundle . --install PDAC_PRR_prrfull_weights_v0.1.0_part1.zip
pdac-prr-weights --bundle . --install PDAC_PRR_prrfull_weights_v0.1.0_part2.zip
pdac-prr-weights --bundle . --model prrpath_text prrfull
```

Replace each ZIP argument with its local path if it is outside the repository.
The installer checks each checkpoint against `model_manifest.json`. A selected
family requires both ZIP parts before inference.

## Validation and scope

The two complete multimodal ensembles and their command-line outputs matched the
original retained model classes exactly on synthetic inputs. No patient records
are included, and no training or cohort reruns were performed. The source server
was accessed read-only. These checks do not establish full patient-cohort or raw
WSI reproduction. Detailed scope is recorded in [VALIDATION.md](VALIDATION.md).

Source code and documentation use MIT. Distribution terms for the project model
weights are separate and have not been specified by that source-only license.
External TRIDENT and foundation-model weights retain their own terms and are
obtained separately.

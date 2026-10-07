# Validation of source release 0.1.0

Validation date: 2026-10-07. Tests used synthetic numeric inputs unless explicitly
described as a comparison with a retained source-server artifact. No training or
cohort reruns were performed, and no source-server files were modified.

## Definition and checkpoint conformance

All four released class definitions match the selected original notebook class
ASTs after class renaming, replacement of an unused auxiliary return placeholder
with `None`, and removal of a stray `+` from the full-model definition. The latter
repair matches the valid saved definition. Score calculations were preserved.
Original and packaged definitions produced identical synthetic outputs for all
four families (maximum observed absolute difference 0).

| Check | ESC | PRRpath | PRRpath+text | PRRfull |
| --- | ---: | ---: | ---: | ---: |
| Source-server retained checkpoints strictly loaded in the earlier read-only audit | 25 | 25 | 25 | 25 |
| Complete checkpoints copied and strictly loaded locally | 25 | 25 | 1 | 0 |
| Complete model-family archive in this handoff | Yes | Yes | No | No |
| Packaged class AST matches retained source | Yes | Yes | Yes | Yes |

All 50 checkpoints in the ESC and PRRpath archives passed exact source-size and
SHA-256 verification. Installing both archives into a clean weight directory and
verifying the installed files succeeded. The first source handoff contains all
four model implementations but does not contain complete text/full weight sets.

## Local functional checks

- Four automated tests passed: valid archive installation and idempotent reuse,
  rejection of changed checkpoint content, rejection of archive entries outside
  the manifest, and PRRpath's 25-checkpoint ensemble against an original-class
  reference on the included synthetic input.
- Synthetic H5 files plus saved sampling indices were processed by
  `scripts/prepare_inputs.py`; the resulting ESC fold-1 five-epoch embedding
  matched the original definition with maximum absolute difference 0.
- Feeding that prepared input through the PRRpath CLI matched the original
  25-checkpoint reference with maximum absolute difference 0.
- Clinical vector ordering and cohort-fitted transformation were checked with
  synthetic clinical rows.
- The text adapter successfully loaded a local synthetic BERT fixture and
  returned finite 2×768 vectors. This is an adapter check, not a local test with
  the real pretrained Chinese BERT weights.
- The optional Streamlit interface rendered its initial upload view without
  exceptions. Browser file-upload interaction was not separately automated.

The host sandbox prevented OpenMP shared-memory creation during the H5 workflow
check. The same check passed under approved local CPU execution; the model source
was not changed to work around that environment restriction.

## Existing cached-artifact comparison

An earlier read-only comparison of one retained source-server case found maximum
absolute differences of 2.98e-8 for the ESC pathological embedding and 2.18e-6
for the BERT report embedding. Its fold-level PRR score differences were 0 for
pathology, 7.87e-6 for pathology+text and 5.96e-6 for the full model. These are
single-case numerical comparisons, without release of the underlying patient
records; they are not complete patient-cohort reproduction.

## Scope not completed

Raw WSI segmentation and feature extraction with external TRIDENT and the four
real foundation encoders were not run end to end in the local release
environment. Complete local text/full ensembles require the remaining weights.
A fixed prospective single-patient clinical scaler was not retained with the
source checkpoints. Paper wording, threshold boundary conventions and statistical
rounding are being reconciled separately; this source release does not claim that
every manuscript result has been recomputed.

To run the bundled tests after installing PRRpath weights:

```sh
python -m unittest discover -s tests -v
```

Without those weights, the original-reference integration test is explicitly
skipped; the archive installation tests still run.

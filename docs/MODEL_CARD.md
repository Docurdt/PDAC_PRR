# Result-linked inference specification

This release packages the retained model implementations associated with the
existing PDAC prediction outputs. It provides inference and input preparation;
it does not retrain models or reconstruct patient cohorts. The authoritative
implementation parameters for this source release are recorded here and in
`model_manifest.json`. Manuscript wording is being reconciled separately.

## Models

| Model | Inputs | Architecture and output |
| --- | --- | --- |
| ESC | Four bags of 1024 patch embeddings: GigaPath 1536, H-optimus-0 1536, CTransPath 768, UNI-v2 1536 | Four 64-dimensional branches with four attention heads; weighted concatenation in GigaPath–H-optimus-0–CTransPath–UNI-v2 order; 256→64→2 classifier; class-1 softmax score |
| PRRpath | ESC pathological embedding, 256 dimensions | 256→128→32→16→1 MLP; SELU hidden activations; sigmoid output; no active Transformer branch |
| PRRpath+text | Report 768; pathology 256 | Separate residual Transformer streams and a concatenated 1024-dimensional fusion stream; four heads, three layers, feed-forward width 256; 1024→256→64→32→1 sigmoid risk head |
| PRRfull | Report 768; pathology 256; clinical 4 | Separate residual streams and a concatenated 1028-dimensional fusion stream; four heads, three layers, feed-forward width 512; 1028→512→128→64→1 sigmoid risk head |

The multimodal vectors are passed to Transformer blocks as sequences of length
one. Text–pathology–clinical is the concatenation order. The checkpoint
configuration explicitly supplies the retained parameter values; class defaults
are not a model selection policy. Dropout is disabled during inference.

Each model family uses five retained epoch checkpoints in each of five inner
folds. Predictions are averaged within each fold and then across the five folds
(25 checkpoints per family). Pathological features supplied to the PRR models
are the mean ESC fused embedding from the five retained checkpoints of inner
fold 1. The manifest lists all 100 exact checkpoint names, sizes and SHA-256
hashes; incomplete families are rejected during inference.

## Input preprocessing

TRIDENT processing uses segmentation confidence 0.7 and zero patch overlap:
CTransPath 10×/256px, H-optimus-0 20×/224px, GigaPath and UNI-v2 20×/256px.
Upstream encoder transforms then resize or crop the extracted images to their
model input dimensions. Patch extraction size and encoder input size are
different stages.

MiniBatchKMeans selects 1024 representative patches independently in each
encoder space (`random_state=41300`, `n_init=1`, `batch_size=3072`, no PCA).
The nearest observed patch to each centroid is retained; duplicate selections
are filled using the original Python-random fallback. Bags with at most 1024
patches use sampling with replacement. Python-random fallbacks were unseeded in
the source; save the generated indices or supply existing indices when exact
bag reproduction is required.

Reports use Chinese BERT revision
`8f23c25b06e129b6c986331a13d8d025a92cf0ea`, maximum token length 128,
max-length padding, truncation and `pooler_output`. A local model directory can
be supplied to avoid a network download.

The clinical vector order is `Sex`, `Age`, `Tumor site`, `Tumor size`. Sex is
coded Female=0/Male=1. Tumor site is coded Head&neck=0, Body&tail=1,
胰头颈+胰体尾=2, Total_pancreas=3. Age is min-max transformed within the supplied
cohort; tumor size retains its original numeric scale, with missing values
filled using that cohort's median. A fixed prospective single-patient scaler
was not retained with these checkpoints. Raw clinical input therefore requires
an explicit cohort context; alternatively supply an already prepared four-value
clinical vector. The package does not invent a deployment scaler.

## Score meaning and scope

ESC scores indicate association with the short-term-survival class. PRR scores
rank relative prognosis and are not calibrated survival probabilities or
predicted survival times. The inference API returns raw scores and fold scores;
it does not assign an individual high/low group using an undocumented threshold.
This is a research inference implementation, without demonstrated prospective
clinical deployment validation.

Training code, clinical records, original images, patient feature caches and
foundation model weights are not distributed in the source archive. See
`VALIDATION.md` for the checks performed and their scope.

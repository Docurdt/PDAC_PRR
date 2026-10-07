# Third-party components

The MIT license in this repository covers the project source code and its
documentation. It does not replace the licenses or access conditions of external
software or pretrained foundation models.

| Component | Use in this release | Source |
| --- | --- | --- |
| TRIDENT | External WSI preprocessing; commit `c9a390563bc37b490119eded5c1d83a17a9e3ba1`; source is not vendored | [TRIDENT](https://github.com/mahmoodlab/TRIDENT) |
| CTransPath | External 768-dimensional patch encoder | [CTransPath](https://github.com/Xiyue-Wang/TransPath) |
| H-optimus-0 | External 1536-dimensional patch encoder | [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) |
| Prov-GigaPath | External 1536-dimensional patch encoder | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| UNI-v2 | External 1536-dimensional patch encoder | [UNI-v2](https://huggingface.co/MahmoodLab/UNI2-h) |
| Chinese BERT | External 768-dimensional report encoder; pinned model revision in `preprocessing.py` | [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese) |

Obtain upstream model weights from their providers under the corresponding
licenses and access terms. None of these pretrained foundation weights are
included in this source package. The downstream ESC/PRR checkpoint archives are
separate artifacts; this source-only release does not specify their distribution
license.

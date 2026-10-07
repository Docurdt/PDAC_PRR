"""Adapters for the original patch selection, text and clinical encoding."""
import random
import numpy as np

BERT_REVISION = '8f23c25b06e129b6c986331a13d8d025a92cf0ea'


def select_patches(features, indices=None):
    """Original 1024-patch algorithm; cached indices preserve exact sampling.

    The source seeds NumPy/KMeans but leaves Python-random fallbacks unseeded.
    Save the returned indices when computing a new bag.
    """
    from sklearn.cluster import MiniBatchKMeans
    features = np.asarray(features)
    if features.ndim != 2 or len(features) == 0 or not np.isfinite(features).all():
        raise ValueError('Expected a non-empty finite patch feature matrix')
    count = 1024
    if indices is None:
        np.random.seed(41300)
        if len(features) > count:
            kmeans = MiniBatchKMeans(n_clusters=count, random_state=41300,
                                    n_init=1, batch_size=3072)
            kmeans.fit(features)
            indices = np.unique(np.argmin(kmeans.transform(features), axis=0))
            if len(indices) < count:
                selected = set(indices.tolist())
                available = [i for i in range(len(features)) if i not in selected]
                indices = np.concatenate([indices, random.sample(available, count - len(indices))])
        else:
            indices = np.asarray(random.choices(range(len(features)), k=count))
    indices = np.asarray(indices)
    if indices.shape != (count,) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('Indices must be a 1024-element integer array')
    if indices.min() < 0 or indices.max() >= len(features):
        raise ValueError('Patch index outside feature matrix')
    return features[indices].astype(np.float32), indices


def load_patch_h5(path, indices=None):
    import h5py
    with h5py.File(path, 'r') as f:
        features = np.array(f['features'])
    selected, indices = select_patches(features, indices)
    return {'features': selected}, indices


def encode_reports(reports, model_path=None, device='cpu'):
    import torch
    from transformers import BertModel, BertTokenizer
    source = model_path or 'google-bert/bert-base-chinese'
    options = {'revision': BERT_REVISION} if model_path is None else {'local_files_only': True}
    tokenizer = BertTokenizer.from_pretrained(source, **options)
    model = BertModel.from_pretrained(source, **options).to(device).eval()
    vectors = []
    with torch.inference_mode():
        for report in reports:
            if not isinstance(report, str) or not report.strip():
                raise ValueError('Pathology report must be non-empty text')
            tokens = tokenizer(report, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=128)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            vectors.append(model(**tokens).pooler_output.cpu().numpy()[0])
    return np.asarray(vectors, dtype=np.float32)


def encode_clinical_cohort(rows):
    """Source cohort-fitted transformation, not a fixed single-case scaler.

    Input rows contain Sex, Age, Tumor site and Tumor size in source units.
    The cohort supplied here determines age bounds and size imputation.
    """
    from sklearn.preprocessing import MinMaxScaler
    if not rows:
        raise ValueError('Clinical cohort is empty')
    sex_map = {'Female': 0, 'Male': 1}
    site_map = {'Head&neck': 0, 'Body&tail': 1, '胰头颈+胰体尾': 2, 'Total_pancreas': 3}
    def numeric(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan
    age = np.asarray([numeric(r['Age']) for r in rows]).reshape(-1, 1)
    size = np.asarray([numeric(r['Tumor size']) for r in rows])
    if not np.isfinite(age).all() or not np.isfinite(size).any():
        raise ValueError('Finite ages and at least one finite tumor size are required')
    median = float(np.nanmedian(size))
    size = np.where(np.isnan(size), median, size)
    scaled = MinMaxScaler().fit_transform(age).ravel()
    try:
        result = np.column_stack([[sex_map[r['Sex']] for r in rows], scaled,
                                  [site_map[r['Tumor site']] for r in rows], size]).astype(np.float32)
    except KeyError as exc:
        raise ValueError(f'Unsupported clinical category or missing field: {exc}') from exc
    if not np.isfinite(result).all():
        raise ValueError('Clinical features contain non-finite values')
    return result, {'policy': 'source_cohort_fit', 'age_min': float(age.min()),
                    'age_max': float(age.max()), 'tumor_size_median': median,
                    'order': ['Sex', 'Age', 'Tumor site', 'Tumor size']}

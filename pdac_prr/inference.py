"""Inference with the retained epoch checkpoints and frozen model definitions."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from . import models


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Predictor:
    def __init__(self, bundle, device='cpu'):
        self.bundle = Path(bundle).resolve()
        self.manifest = json.loads((self.bundle / 'model_manifest.json').read_text())
        self.device = torch.device(device)

    def _checkpoints(self, family):
        spec = self.manifest['models'][family]
        folds = {fold: [] for fold in range(1, 6)}
        for item in spec['expected_checkpoints']:
            path = self.bundle / 'weights' / family / item['name']
            if not path.is_file() or path.stat().st_size != item['size']:
                raise FileNotFoundError(f'Missing or incomplete checkpoint: {path}')
            if not item.get('sha256'):
                raise ValueError(f'Checkpoint lacks a verified source hash: {item["name"]}')
            if file_sha256(path) != item['sha256']:
                raise ValueError(f'Checkpoint SHA-256 mismatch: {path}')
            match = re.search(r'_innerloop_(\d+)_', item['name'])
            if match is None or int(match[1]) not in folds:
                raise ValueError(f'Unrecognized checkpoint fold: {item["name"]}')
            folds[int(match[1])].append(path)
        if any(len(paths) != 5 for paths in folds.values()):
            raise ValueError('Expected five retained epochs in each of five inner folds')
        return folds

    def _model(self, family):
        spec = self.manifest['models'][family]
        kwargs = {k: spec[k] for k in ['hidden_dim', 'embed_dim', 'num_heads', 'num_layers', 'dropout'] if k in spec}
        return getattr(models, spec['class'])(feature_dims=models.FEATURE_DIMS,
                                              device=self.device, **kwargs).to(self.device).eval()

    def _load(self, model, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)

    def predict(self, family, bags):
        """Mean epoch predictions within folds, then mean the five folds."""
        if family not in self.manifest['models']:
            raise ValueError(f'Unknown model: {family}')
        tensors = validate_inputs(family, bags)
        folds = self._checkpoints(family)
        model = self._model(family)
        fold_predictions = []
        with torch.inference_mode():
            for paths in folds.values():
                predictions = []
                for path in paths:
                    self._load(model, path)
                    output = model(tensors)[0]
                    if family == 'esc':
                        output = torch.softmax(output, dim=1)[:, 1]
                    predictions.append(output.detach().cpu().numpy().reshape(-1))
                fold_predictions.append(np.mean(np.stack(predictions), axis=0))
        scores = np.mean(np.stack(fold_predictions), axis=0)
        return {'model': family, 'scores': scores.tolist(),
                'fold_scores': np.stack(fold_predictions).T.tolist(),
                'interpretation': ('short_term_class_score' if family == 'esc' else 'relative_prognostic_risk_score'),
                'checkpoint_count': 25}

    def pathology_embedding(self, bags):
        """Source notebook's five-epoch inner-fold-1 feature average.

        A frozen-case comparison matched within 2.98e-8. Raw-slide preprocessing with external foundation models is separately
        configured; see the release validation report.
        """
        tensors = validate_inputs('esc', bags)
        paths = self._checkpoints('esc')[1]
        model = self._model('esc')
        features = []
        with torch.inference_mode():
            for path in paths:
                self._load(model, path)
                model(tensors)
                features.append(model.fused_features_debug.detach().cpu().numpy())
        return np.mean(np.stack(features), axis=0)


def validate_inputs(family, bags):
    result = {}
    batch = None
    if family == 'esc':
        for key, width in models.FEATURE_DIMS.items():
            value = torch.as_tensor(bags[key]['features'], dtype=torch.float32)
            if value.ndim == 2:
                value = value.unsqueeze(0)
            if value.ndim != 3 or value.shape[0] < 1 or value.shape[1:] != (1024, width):
                raise ValueError(f'{key} must have shape (batch,1024,{width})')
            if not torch.isfinite(value).all():
                raise ValueError(f'{key} contains non-finite values')
            if batch is not None and value.shape[0] != batch:
                raise ValueError('Patch bags must have the same batch size')
            batch = value.shape[0]
            result[key] = {'features': value.contiguous()}
        return result
    requirements = {'ls_features': 256}
    if family in ['prrpath_text', 'prrfull']:
        requirements['text_features'] = 768
    if family == 'prrfull':
        requirements['clin_features'] = 4
    for key, width in requirements.items():
        value = torch.as_tensor(bags[key], dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[1] != width or value.shape[0] < 1:
            raise ValueError(f'{key} must have shape (batch,{width})')
        if not torch.isfinite(value).all():
            raise ValueError(f'{key} contains non-finite values')
        if batch is not None and value.shape[0] != batch:
            raise ValueError('All modalities must have the same batch size')
        batch = value.shape[0]
        result[key] = value.contiguous()
    return result

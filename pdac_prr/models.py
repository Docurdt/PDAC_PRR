"""Inference definitions extracted from retained, result-linked model notebooks.
No training, file reading, or file writing occurs at import.
"""

import torch

from torch import nn

from torch.nn import functional as F

FEATURE_DIMS = {"gigapth":1536,"uni_v2":1536,"ctranspath":768,"hoptimus0":1536}

feature_dims = FEATURE_DIMS


class ESC(nn.Module):

    def __init__(self, feature_dims, device, hidden_dim=512, num_heads=4, num_layers=4, dropout=0.5):
        super().__init__()
        self.expected_keys = ['gigapth', 'hoptimus0', 'ctranspath', 'uni_v2']
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert all((key in feature_dims for key in self.expected_keys)), 'Missing keys in feature_dims'
        self.encoders = nn.ModuleDict({key: nn.Sequential(nn.Linear(dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(negative_slope=0.01), nn.Dropout(dropout)) for key, dim in feature_dims.items()})
        self.attention = nn.ModuleDict({key: nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True) for key, dim in feature_dims.items()})
        self.cross_modal_projection = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attention_weights = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, bags, times=None, events=None):
        assert all((key in bags for key in self.expected_keys)), 'Missing modality in bags'
        instance_features = {}
        instance_attn = {}
        first_key = self.expected_keys[0]
        feats_first = bags[first_key]['features'].to(self.device)
        is_batched = feats_first.dim() == 3
        batch_size = feats_first.shape[0] if is_batched else 1
        for key in self.expected_keys:
            feats = bags[key]['features'].to(self.device)
            if feats.dim() not in [2, 3]:
                raise ValueError(f'Expected 2D (single bag) or 3D (batched) input for {key}, got {feats.dim()}D')
            if is_batched and feats.dim() != 3:
                raise ValueError(f'Inconsistent input dimensions: {first_key} is batched, but {key} is not')
            if not is_batched and feats.dim() != 2:
                raise ValueError(f'Inconsistent input dimensions: {first_key} is not batched, but {key} is')
            if is_batched and feats.shape[0] != batch_size:
                raise ValueError(f'Inconsistent batch sizes: {first_key} has batch size {batch_size}, but {key} has {feats.shape[0]}')
            if not is_batched:
                feats = feats.unsqueeze(0)
            _, num_instances, feat_dim = feats.shape
            if feat_dim != feature_dims[key]:
                raise ValueError(f'Expected feature dim {feature_dims[key]} for {key}, got {feat_dim}')
            feats = feats.view(-1, feat_dim)
            encoded = self.encoders[key](feats)
            encoded = encoded.view(batch_size, num_instances, self.hidden_dim)
            attn_output, attn_weights = self.attention[key](encoded, encoded, encoded)
            attn_weights = attn_weights.mean(dim=2)
            attn_weights = attn_weights.unsqueeze(-1)
            agg_feature = torch.sum(attn_output * attn_weights, dim=1)
            instance_features[key] = agg_feature
            instance_attn[key] = attn_weights.squeeze(-1)
        projected_features = {key: self.cross_modal_projection(instance_features[key]) for key in self.expected_keys}
        modal_features = torch.stack([projected_features[key] for key in self.expected_keys], dim=1)
        attention_scores = self.cross_attention_weights(modal_features).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_features = [instance_features[key] * attention_weights[:, i].unsqueeze(-1) for i, key in enumerate(self.expected_keys)]
        fused_features = torch.cat(weighted_features, dim=1)
        logits = self.classifier(fused_features)
        self.fused_features_debug = fused_features
        self.instance_attn_debug = instance_attn
        return (logits, instance_attn, times, events)

class PRRPath(nn.Module):

    def __init__(self, feature_dims, embed_dim=256, num_heads=4, num_layers=2, dropout=0.3, device='cuda'):
        super().__init__()
        self.device = device
        self.feature_dims = feature_dims
        self.embed_dim = embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.SELU(),
            nn.Linear(embed_dim // 4, embed_dim // 8),
            nn.SELU(),
            nn.Linear(embed_dim // 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, bags, times=None, events=None):
        ls_features = bags['ls_features'].to(self.device)
        logits = self.classifier(ls_features)
        return (logits, None, times, events)

class PRRPathText(nn.Module):

    def __init__(self, feature_dims, embed_dim=256, num_heads=4, num_layers=2, dropout=0.3, device='cuda'):
        super().__init__()
        self.device = device
        self.feature_dims = feature_dims
        self.embed_dim = embed_dim
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=num_heads,
                dim_feedforward=embed_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.lsf_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=num_heads,
                dim_feedforward=embed_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.multimodal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256 + 768,
                nhead=num_heads,
                dim_feedforward=embed_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 256, embed_dim),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.SELU(),
            nn.Linear(embed_dim // 4, embed_dim // 8),
            nn.SELU(),
            nn.Linear(embed_dim // 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, bags, times=None, events=None):
        text_features = bags['text_features'].to(self.device)
        ls_features = bags['ls_features'].to(self.device)
        if len(text_features.shape) == 2:
            text_features_res = text_features.unsqueeze(1)
        text_features_res = self.text_transformer(text_features_res)
        if len(text_features_res.shape) == 3:
            text_features_res = text_features_res.squeeze(1)
        text_features = text_features + text_features_res
        if len(ls_features.shape) == 2:
            ls_features_res = ls_features.unsqueeze(1)
        ls_features_res = self.lsf_transformer(ls_features_res)
        if len(ls_features_res.shape) == 3:
            ls_features_res = ls_features_res.squeeze(1)
        ls_features = ls_features + ls_features_res
        all_embed = torch.cat([text_features, ls_features], dim=1)
        if len(all_embed.shape) == 2:
            all_embed = all_embed.unsqueeze(1)
        all_embed = self.multimodal_transformer(all_embed)
        if len(all_embed.shape) == 3:
            all_embed = all_embed.squeeze(1)
        logits = self.classifier(all_embed)
        return (logits, None, times, events)

class PRRFull(nn.Module):

    def __init__(self, feature_dims, embed_dim=256, num_heads=4, num_layers=2, dropout=0.3, device='cuda'):
        super().__init__()
        self.device = device
        self.feature_dims = feature_dims
        self.embed_dim = embed_dim
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=num_heads,
                dim_feedforward=embed_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.lsf_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=num_heads,
                dim_feedforward=embed_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.clin_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=4,
                nhead=num_heads,
                dim_feedforward=embed_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.multimodal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256 + 768 + 4,
                nhead=num_heads,
                dim_feedforward=embed_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 256 + 4, embed_dim),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.SELU(),
            nn.Linear(embed_dim // 4, embed_dim // 8),
            nn.SELU(),
            nn.Linear(embed_dim // 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, bags, times=None, events=None):
        text_features = bags['text_features'].to(self.device)
        ls_features = bags['ls_features'].to(self.device)
        clin_features = bags['clin_features'][:, 0:4].to(self.device)
        if len(text_features.shape) == 2:
            text_features_res = text_features.unsqueeze(1)
        text_features_res = self.text_transformer(text_features_res)
        if len(text_features_res.shape) == 3:
            text_features_res = text_features_res.squeeze(1)
        text_features = text_features + text_features_res
        if len(ls_features.shape) == 2:
            ls_features_res = ls_features.unsqueeze(1)
        ls_features_res = self.lsf_transformer(ls_features_res)
        if len(ls_features_res.shape) == 3:
            ls_features_res = ls_features_res.squeeze(1)
        ls_features = ls_features + ls_features_res
        if len(clin_features.shape) == 2:
            clin_features_res = clin_features.unsqueeze(1)
        clin_features_res = self.clin_transformer(clin_features_res)
        if len(clin_features_res.shape) == 3:
            clin_features_res = clin_features_res.squeeze(1)
        clin_features = clin_features + clin_features_res
        all_embed = torch.cat([text_features, ls_features, clin_features], dim=1)
        if len(all_embed.shape) == 2:
            all_embed = all_embed.unsqueeze(1)
        all_embed = self.multimodal_transformer(all_embed)
        if len(all_embed.shape) == 3:
            all_embed = all_embed.squeeze(1)
        logits = self.classifier(all_embed)
        return (logits, None, times, events)

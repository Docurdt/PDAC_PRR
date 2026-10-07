"""Prepare features for one explicitly selected slide; outputs stay local."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from pdac_prr.inference import Predictor
from pdac_prr.preprocessing import load_patch_h5, encode_reports, encode_clinical_cohort

SETTINGS = {'gigapth':('gigapath',20,256), 'hoptimus0':('hoptimus0',20,224),
            'ctranspath':('ctranspath',10,256), 'uni_v2':('uni_v2',20,256)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--features-root', type=Path, required=True)
    parser.add_argument('--slide-stem', required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--bert-path', type=Path)
    parser.add_argument('--clinical-cohort', type=Path)
    parser.add_argument('--case-row', type=int, default=0)
    parser.add_argument('--sampling-indices', type=Path, help='Optional NPZ of frozen encoder-specific sampling indices')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if Path(args.slide_stem).name != args.slide_stem or args.slide_stem in {'.','..'}:
        parser.error('Slide stem must be a filename stem, not a path')
    if args.output.suffix != '.npz':
        parser.error('Output must have .npz extension')
    sidecar = args.output.with_suffix('.json')
    indices_file = args.output.with_name(args.output.stem+'_indices.npz')
    if any(p.exists() for p in [args.output, sidecar, indices_file]):
        parser.error('An output already exists; choose a new output name')
    supplied = {}
    if args.sampling_indices:
        with np.load(args.sampling_indices, allow_pickle=False) as saved:
            supplied = {k:saved[k] for k in SETTINGS}
    bags, selected = {}, {}
    for key, (encoder,mag,patch) in SETTINGS.items():
        path = args.features_root / f'{mag}x_{patch}px_0px_overlap' / f'features_{encoder}' / f'{args.slide_stem}.h5'
        bags[key], selected[key] = load_patch_h5(path, supplied.get(key))
    embedding = Predictor(args.bundle, args.device).pathology_embedding(bags)
    arrays = {'ls_features':embedding}
    meta = {'feature_encoder_fold':1,'epoch_count':5,
            'sampling_sha256':{k:hashlib.sha256(v.astype(np.int64).tobytes()).hexdigest() for k,v in selected.items()}}
    if args.report:
        arrays['text_features'] = encode_reports([args.report.read_text(encoding='utf-8')],
            model_path=str(args.bert_path) if args.bert_path else None, device=args.device)
    if args.clinical_cohort:
        with args.clinical_cohort.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        clinical, context = encode_clinical_cohort(rows)
        if not 0 <= args.case_row < len(rows):
            parser.error('Clinical case row is outside the provided cohort')
        arrays['clin_features'] = clinical[args.case_row:args.case_row+1]
        meta['clinical_context'] = context
        meta['clinical_cohort_size'] = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez(stream, **arrays)
    with indices_file.open('xb') as stream:
        np.savez(stream, **selected)
    with sidecar.open('x',encoding='utf-8') as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()

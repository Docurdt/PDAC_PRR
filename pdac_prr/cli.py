import argparse
import json
from pathlib import Path

import numpy as np

from .inference import Predictor


def main():
    parser = argparse.ArgumentParser(description='PDAC PRR inference')
    parser.add_argument('--bundle', type=Path, default=Path.cwd())
    parser.add_argument('--model', required=True, choices=['esc','prrpath','prrpath_text','prrfull'])
    parser.add_argument('--input', type=Path, required=True, help='Numeric NPZ feature file; no pickle objects')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new output filename')
    with np.load(args.input, allow_pickle=False) as data:
        if args.model == 'esc':
            bags = {k: {'features': data[k]} for k in ['gigapth','hoptimus0','ctranspath','uni_v2']}
        else:
            bags = {k: data[k] for k in data.files}
        result = Predictor(args.bundle, args.device).predict(args.model, bags)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()

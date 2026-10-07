"""Invoke the unchanged, pinned TRIDENT runner with source pipeline settings."""
import argparse
from pathlib import Path
import subprocess
import sys

TRIDENT_COMMIT = 'c9a390563bc37b490119eded5c1d83a17a9e3ba1'
SETTINGS = [('gigapath',20,256), ('uni_v2',20,256), ('ctranspath',10,256), ('hoptimus0',20,224)]


def commands(checkout, wsi_dir, output_dir, gpu, weight_paths):
    runner = checkout / 'run_batch_of_slides.py'
    for encoder, mag, patch in SETTINGS:
        args = [sys.executable, str(runner), '--task', 'all', '--seg_conf_thresh', '0.7',
                '--wsi_dir', str(wsi_dir), '--job_dir', str(output_dir),
                '--patch_encoder', encoder, '--gpu', str(gpu), '--mag', str(mag),
                '--patch_size', str(patch), '--overlap', '0']
        if encoder in weight_paths:
            args.extend(['--patch_encoder_ckpt_path', str(weight_paths[encoder])])
        yield args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trident', type=Path, required=True)
    parser.add_argument('--wsi-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--weights-json', type=Path, help='Encoder-name to local-checkpoint-path mapping')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    checkout = args.trident.resolve()
    version = subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    if version != TRIDENT_COMMIT:
        parser.error(f'TRIDENT must be at {TRIDENT_COMMIT}')
    changes = subprocess.check_output(['git','-C',str(checkout),'diff','HEAD','--name-only'],text=True).splitlines()
    allowed = {'trident/patch_encoder_models/local_ckpts.json','trident/segmentation_models/local_ckpts.json'}
    if set(changes) - allowed:
        parser.error('Pinned TRIDENT code has local changes')
    import json
    import shlex
    weights = json.loads(args.weights_json.read_text()) if args.weights_json else {}
    for command in commands(checkout, args.wsi_dir.resolve(), args.output_dir.resolve(), args.gpu, weights):
        if args.dry_run:
            print(shlex.join(command))
        else:
            subprocess.run(command, cwd=checkout, check=True)


if __name__ == '__main__':
    main()

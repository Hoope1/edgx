import argparse
import os
import cv2
from detectors import (
    run_hed, run_structured, run_kornia, run_bdcn, run_fixed
)

METHODS = [
    ('HED', run_hed),
    ('StructuredForests', run_structured),
    ('Kornia', run_kornia),
    ('BDCN', run_bdcn),
    ('FixedEdgeCNN', run_fixed),
]

def main(inp: str, outp: str) -> None:
    os.makedirs(outp, exist_ok=True)
    for name, _ in METHODS:
        os.makedirs(os.path.join(outp, name), exist_ok=True)

    for fname in os.listdir(inp):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        src = os.path.join(inp, fname)
        for name, func in METHODS:
            try:
                print(f"Processing {fname} with {name}...")
                res = func(src)
                output_path = os.path.join(outp, name, fname)
                cv2.imwrite(output_path, res)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {fname} with {name}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Input directory with images')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        exit(1)

    main(args.input_dir, args.output_dir)
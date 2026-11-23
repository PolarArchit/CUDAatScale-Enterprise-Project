#!/usr/bin/env python3
"""
Generates side-by-side visual comparison artifacts combining original input
and GPU-processed output images for visual inspection by peer reviewers.
"""

import os
import sys

def read_pgm(filepath):
    with open(filepath, 'rb') as f:
        magic = f.readline().strip()
        if magic != b'P5':
            return None, 0, 0
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        f.readline()  # max val
        data = f.read(w * h)
        return data, w, h

def write_pgm(filepath, width, height, data):
    with open(filepath, 'wb') as f:
        f.write(f"P5\n{width} {height}\n255\n".encode('ascii'))
        f.write(data)

def create_side_by_side(orig_path, proc_path, out_path):
    orig_data, w1, h1 = read_pgm(orig_path)
    proc_data, w2, h2 = read_pgm(proc_path)
    if not orig_data or not proc_data or (w1 != w2) or (h1 != h2):
        print(f"Skipping {orig_path} (mismatched or unreadable)")
        return False

    out_w = w1 * 2 + 10  # 10px divider
    out_h = h1
    out_buf = bytearray(out_w * out_h)

    for y in range(out_h):
        # Left: Original
        for x in range(w1):
            out_buf[y * out_w + x] = orig_data[y * w1 + x]
        # Middle divider (white)
        for x in range(w1, w1 + 10):
            out_buf[y * out_w + x] = 255
        # Right: Processed
        for x in range(w1):
            out_buf[y * out_w + (w1 + 10 + x)] = proc_data[y * w2 + x]

    write_pgm(out_path, out_w, out_h, bytes(out_buf))
    print(f"Created comparison artifact: {out_path} ({out_w}x{out_h})")
    return True

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, 'data', 'input')
    output_dir = os.path.join(base_dir, 'output')
    artifacts_dir = os.path.join(base_dir, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    samples = [
        ("img_512_000.pgm", "processed_img_512_000.pgm", "comparison_sample_concentric_512.pgm"),
        ("img_512_001.pgm", "processed_img_512_001.pgm", "comparison_sample_checkerboard_512.pgm"),
        ("img_1024_000.pgm", "processed_img_1024_000.pgm", "comparison_sample_highres_1024.pgm"),
    ]

    for orig, proc, out in samples:
        orig_p = os.path.join(input_dir, orig)
        proc_p = os.path.join(output_dir, proc)
        out_p = os.path.join(artifacts_dir, out)
        if os.path.exists(orig_p) and os.path.exists(proc_p):
            create_side_by_side(orig_p, proc_p, out_p)

if __name__ == '__main__':
    main()

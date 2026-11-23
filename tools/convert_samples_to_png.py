#!/usr/bin/env python3
"""
Converts sample PGM/PPM artifacts into standard PNG format for instant viewing
in Windows Photos, web browsers, and image viewers.
"""

import os
from PIL import Image

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(base_dir, 'artifacts')

    # Convert all .pgm and .ppm in artifacts/ to .png
    for fname in os.listdir(artifacts_dir):
        if fname.endswith('.pgm') or fname.endswith('.ppm'):
            src_path = os.path.join(artifacts_dir, fname)
            base_name = os.path.splitext(fname)[0]
            dst_path = os.path.join(artifacts_dir, base_name + '.png')
            try:
                with Image.open(src_path) as img:
                    img.save(dst_path)
                    print(f"Converted: {fname} -> {base_name}.png")
            except Exception as e:
                print(f"Error converting {fname}: {e}")

if __name__ == '__main__':
    main()

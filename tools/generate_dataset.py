#!/usr/bin/env python3
"""
Synthetic Dataset Generator for CUDA at Scale Enterprise Project.
Generates 120+ synthetic procedural images (PGM / PPM) across varied dimensions
(256x256, 512x512, 1024x1024) and diverse visual patterns to fulfill the
Coursera assignment requirement of executing on "100s of small or 10s of large" inputs.

Zero external dependencies required (uses standard library math and struct).
"""

import math
import os
import struct
import sys

def write_pgm_p5(filepath, width, height, pixel_bytes):
    """Writes a binary PGM (P5) image."""
    with open(filepath, 'wb') as f:
        header = f"P5\n{width} {height}\n255\n".encode('ascii')
        f.write(header)
        f.write(pixel_bytes)

def write_ppm_p6(filepath, width, height, pixel_bytes):
    """Writes a binary PPM (P6) RGB image."""
    with open(filepath, 'wb') as f:
        header = f"P6\n{width} {height}\n255\n".encode('ascii')
        f.write(header)
        f.write(pixel_bytes)

def generate_checkerboard(width, height, cell_size):
    buf = bytearray(width * height)
    idx = 0
    for y in range(height):
        cy = (y // cell_size) % 2
        for x in range(width):
            cx = (x // cell_size) % 2
            val = 240 if (cx ^ cy) else 15
            buf[idx] = val
            idx += 1
    return bytes(buf)

def generate_concentric_rings(width, height, frequency):
    buf = bytearray(width * height)
    cx, cy = width / 2.0, height / 2.0
    idx = 0
    for y in range(height):
        dy = y - cy
        for x in range(width):
            dx = x - cx
            dist = math.sqrt(dx * dx + dy * dy)
            val = int(127.5 * (1.0 + math.sin(dist * frequency)))
            buf[idx] = max(0, min(255, val))
            idx += 1
    return bytes(buf)

def generate_diagonal_stripes(width, height, period):
    buf = bytearray(width * height)
    idx = 0
    for y in range(height):
        for x in range(width):
            val = int(127.5 * (1.0 + math.sin((x + y) * (2.0 * math.pi / period))))
            buf[idx] = max(0, min(255, val))
            idx += 1
    return bytes(buf)

def generate_radial_gradient(width, height):
    buf = bytearray(width * height)
    cx, cy = width / 2.0, height / 2.0
    max_dist = math.sqrt(cx * cx + cy * cy)
    idx = 0
    for y in range(height):
        dy = y - cy
        for x in range(width):
            dx = x - cx
            dist = math.sqrt(dx * dx + dy * dy)
            val = int(255.0 * (1.0 - (dist / max_dist)))
            buf[idx] = max(0, min(255, val))
            idx += 1
    return bytes(buf)

def generate_geometric_shapes(width, height):
    buf = bytearray(width * height)
    idx = 0
    cx, cy = width // 2, height // 2
    r_circle = min(width, height) // 3
    rect_half_w = min(width, height) // 4
    rect_half_h = min(width, height) // 6

    for y in range(height):
        for x in range(width):
            # Default background
            val = 30
            # Outer circle
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy < r_circle * r_circle:
                val = 200
            # Inner rectangle
            if abs(dx) < rect_half_w and abs(dy) < rect_half_h:
                val = 80
            # Center diamond
            if (abs(dx) + abs(dy)) < (rect_half_w // 2):
                val = 255
            buf[idx] = val
            idx += 1
    return bytes(buf)

def generate_rgb_color_bars(width, height):
    buf = bytearray(width * height * 3)
    idx = 0
    for y in range(height):
        for x in range(width):
            # 8 horizontal or vertical bars
            bar = int((x / width) * 8)
            r = 255 if bar in [0, 1, 2, 3] else 0
            g = 255 if bar in [0, 1, 4, 5] else 0
            b = 255 if bar in [0, 2, 4, 6] else 0
            buf[idx] = r
            buf[idx + 1] = g
            buf[idx + 2] = b
            idx += 3
    return bytes(buf)

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'data', 'input')
    os.makedirs(target_dir, exist_ok=True)

    print(f"Generating synthetic dataset in: {target_dir}")

    count = 0

    # 1. Standard batch: 50 images of 256x256
    print("Generating 256x256 test images...")
    for i in range(50):
        w, h = 256, 256
        mode = i % 5
        filename = f"img_256_{i:03d}.pgm"
        filepath = os.path.join(target_dir, filename)

        if mode == 0:
            data = generate_checkerboard(w, h, 8 + (i % 16))
        elif mode == 1:
            data = generate_concentric_rings(w, h, 0.05 + (i * 0.01))
        elif mode == 2:
            data = generate_diagonal_stripes(w, h, 10 + (i % 20))
        elif mode == 3:
            data = generate_radial_gradient(w, h)
        else:
            data = generate_geometric_shapes(w, h)

        write_pgm_p5(filepath, w, h, data)
        count += 1

    # 2. Medium enterprise batch: 50 images of 512x512
    print("Generating 512x512 benchmark images...")
    for i in range(50):
        w, h = 512, 512
        mode = i % 5
        filename = f"img_512_{i:03d}.pgm"
        filepath = os.path.join(target_dir, filename)

        if mode == 0:
            data = generate_concentric_rings(w, h, 0.03 + (i * 0.005))
        elif mode == 1:
            data = generate_checkerboard(w, h, 16 + (i % 32))
        elif mode == 2:
            data = generate_geometric_shapes(w, h)
        elif mode == 3:
            data = generate_diagonal_stripes(w, h, 16 + (i % 30))
        else:
            data = generate_radial_gradient(w, h)

        write_pgm_p5(filepath, w, h, data)
        count += 1

    # 3. Large enterprise batch: 15 images of 1024x1024 (high resolution)
    print("Generating 1024x1024 high-resolution images...")
    for i in range(15):
        w, h = 1024, 1024
        mode = i % 4
        filename = f"img_1024_{i:03d}.pgm"
        filepath = os.path.join(target_dir, filename)

        if mode == 0:
            data = generate_concentric_rings(w, h, 0.02 + (i * 0.003))
        elif mode == 1:
            data = generate_checkerboard(w, h, 32 + (i * 4))
        elif mode == 2:
            data = generate_geometric_shapes(w, h)
        else:
            data = generate_radial_gradient(w, h)

        write_pgm_p5(filepath, w, h, data)
        count += 1

    # 4. Color PPM images: 10 images of 512x512 RGB
    print("Generating 512x512 RGB color PPM images...")
    for i in range(10):
        w, h = 512, 512
        filename = f"img_color_{i:03d}.ppm"
        filepath = os.path.join(target_dir, filename)
        data = generate_rgb_color_bars(w, h)
        write_ppm_p6(filepath, w, h, data)
        count += 1

    print(f"\n[Dataset Generator] Successfully created {count} test images in {target_dir}")

if __name__ == '__main__':
    main()

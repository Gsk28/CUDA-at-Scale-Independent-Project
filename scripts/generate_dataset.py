#!/usr/bin/env python3
"""Generate deterministic PGM images for the CUDA batch demo."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


def clamp(value: int) -> int:
    return max(0, min(255, value))


def pixel_value(x: int, y: int, index: int, width: int, height: int, rng: random.Random) -> int:
    cx = width * (0.35 + 0.25 * math.sin(index * 0.17))
    cy = height * (0.45 + 0.20 * math.cos(index * 0.11))
    radius = 18 + (index % 23)
    dx = x - cx
    dy = y - cy
    circle = 110 if dx * dx + dy * dy < radius * radius else 0

    stripe_period = 9 + (index % 11)
    stripes = 70 if ((x + 2 * y + index * 5) // stripe_period) % 2 == 0 else 0

    wave = int(48 * math.sin((x * 0.06) + index * 0.21))
    ramp = int(80 * x / max(1, width - 1))
    box = 95 if (width // 5 < x < width // 5 + 38 and (index * 3) % height < y < (index * 3) % height + 48) else 0
    noise = rng.randint(-18, 18)
    return clamp(32 + ramp + wave + stripes + circle + box + noise)


def write_pgm(path: Path, width: int, height: int, pixels: bytearray) -> None:
    with path.open("wb") as file:
        file.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        file.write(pixels)


def generate_image(path: Path, index: int, width: int, height: int, seed: int) -> None:
    rng = random.Random(seed + index * 1009)
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            pixels[y * width + x] = pixel_value(x, y, index, width, height, rng)
    write_pgm(path, width, height, pixels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a batch of PGM images.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--count", type=int, default=256, help="Number of images")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.width <= 0 or args.height <= 0:
        raise SystemExit("count, width, and height must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        image_path = args.output / f"synthetic_{index:04d}.pgm"
        generate_image(image_path, index, args.width, args.height, args.seed)

    total_pixels = args.count * args.width * args.height
    print(f"Generated {args.count} PGM images in {args.output}")
    print(f"Image size: {args.width}x{args.height}")
    print(f"Total pixels: {total_pixels}")


if __name__ == "__main__":
    main()

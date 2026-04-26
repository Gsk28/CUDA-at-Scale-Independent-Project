#!/usr/bin/env python3
"""Create a PNG contact sheet from binary PGM images without third-party deps."""

from __future__ import annotations

import argparse
import math
import struct
import zlib
from pathlib import Path


def read_token(data: bytes, offset: int) -> tuple[str, int]:
    while offset < len(data) and chr(data[offset]).isspace():
        offset += 1
    if offset < len(data) and data[offset] == ord("#"):
        while offset < len(data) and data[offset] not in (10, 13):
            offset += 1
        return read_token(data, offset)
    start = offset
    while offset < len(data) and not chr(data[offset]).isspace():
        offset += 1
    return data[start:offset].decode("ascii"), offset


def read_pgm(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    magic, offset = read_token(data, 0)
    if magic != "P5":
        raise ValueError(f"{path} is not a binary PGM file")
    width_text, offset = read_token(data, offset)
    height_text, offset = read_token(data, offset)
    max_text, offset = read_token(data, offset)
    width = int(width_text)
    height = int(height_text)
    if int(max_text) != 255:
        raise ValueError(f"{path} uses unsupported max value {max_text}")
    while offset < len(data) and chr(data[offset]).isspace():
        offset += 1
    pixels = data[offset : offset + width * height]
    if len(pixels) != width * height:
        raise ValueError(f"{path} ended before all pixels were read")
    return width, height, pixels


def resize_nearest(width: int, height: int, pixels: bytes, target: int) -> tuple[int, int, bytes]:
    scale = target / max(width, height)
    out_width = max(1, int(round(width * scale)))
    out_height = max(1, int(round(height * scale)))
    resized = bytearray(out_width * out_height)
    for y in range(out_height):
        source_y = min(height - 1, int(y / scale))
        for x in range(out_width):
            source_x = min(width - 1, int(x / scale))
            resized[y * out_width + x] = pixels[source_y * width + source_x]
    return out_width, out_height, bytes(resized)


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(payload, crc)
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc & 0xFFFFFFFF)


def write_grayscale_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        start = y * width
        rows.extend(pixels[start : start + width])
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = signature
    png += png_chunk(b"IHDR", ihdr)
    png += png_chunk(b"IDAT", zlib.compress(bytes(rows), 9))
    png += png_chunk(b"IEND", b"")
    path.write_bytes(png)


def make_sheet(images: list[Path], output: Path, tile_size: int, columns: int) -> None:
    resized_images = []
    for image in images:
        width, height, pixels = read_pgm(image)
        resized_images.append(resize_nearest(width, height, pixels, tile_size))

    rows = math.ceil(len(resized_images) / columns)
    gap = 6
    sheet_width = columns * tile_size + (columns + 1) * gap
    sheet_height = rows * tile_size + (rows + 1) * gap
    sheet = bytearray([235] * (sheet_width * sheet_height))

    for index, (width, height, pixels) in enumerate(resized_images):
        column = index % columns
        row = index // columns
        origin_x = gap + column * (tile_size + gap)
        origin_y = gap + row * (tile_size + gap)
        for y in range(height):
            target_y = origin_y + y
            for x in range(width):
                target_x = origin_x + x
                sheet[target_y * sheet_width + target_x] = pixels[y * width + x]

    output.parent.mkdir(parents=True, exist_ok=True)
    write_grayscale_png(output, sheet_width, sheet_height, bytes(sheet))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a PNG contact sheet from PGM files.")
    parser.add_argument("--input", required=True, type=Path, help="Directory containing PGM files")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path")
    parser.add_argument("--limit", type=int, default=12, help="Maximum images to include")
    parser.add_argument("--tile-size", type=int, default=128, help="Maximum width/height per tile")
    parser.add_argument("--columns", type=int, default=4, help="Number of columns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = sorted(args.input.glob("*.pgm"))[: args.limit]
    if not images:
        raise SystemExit(f"No PGM files found in {args.input}")
    make_sheet(images, args.output, args.tile_size, args.columns)
    print(f"Wrote {args.output} using {len(images)} images")


if __name__ == "__main__":
    main()

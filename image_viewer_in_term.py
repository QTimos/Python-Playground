import time
import sys
import math
import zlib
from enum import IntEnum
import pygame
import numpy as np
import teimpy
from pathlib import Path

# Ensure Pygame runs headless so it doesn't try to open an external X11/Wayland window
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
FORMAT = ""
WIDTH: int
HEIGHT: int
BIT_DEPTH: int
COLOR_TYPE: int
COLOR_LIST: list = []

class ColorType(IntEnum):
    GREYSCALE = 0
    TRUECOLOR = 2
    INDEXEDCOLOR = 3
    GREYSCALE_WITH_ALPHA = 4
    TRUECOLOR_WITH_ALPHA = 6

# --- Keep your exact png_populate_canvas logic here ---
def png_populate_canvas(canvas: pygame.Surface, size: tuple, bit_depth: int, image_data_joined: bytes, color_type: int, color_list: list) -> None:
    # ... [Keep your entire decoding block exactly as it is] ...
    pass

def render_png_image(data: bytes) -> None:
    global COLOR_TYPE, COLOR_LIST, WIDTH, HEIGHT, BIT_DEPTH
    sig = data[:8]
    support = sig[0] == 0x89
    format = sig[1:4].decode(encoding="utf-8") == "PNG"
    line_ending = sig[4:6] == "\r\n".encode(encoding="utf-8")
    stop_byte = sig[6] == 0x1A
    unix_line_ending = sig[7] == 0x0A
    if not (support and format and line_ending and stop_byte and unix_line_ending):
        raise RuntimeError("error found in image header.")
    chunks = data[8:]
    first = True
    valid = False
    image_data = []
    while len(chunks) >= 12:
        chunk_data_len = int.from_bytes(chunks[:4], byteorder="big")
        if 12 + chunk_data_len > len(chunks):
            raise RuntimeError("error in length of a chunk.")
        chunk_type = chunks[4:8].decode(encoding="utf-8")
        chunk_data_offset = 8 + chunk_data_len
        chunk_data = chunks[8:chunk_data_offset]
        checksum = chunks[chunk_data_offset: chunk_data_offset+4]
        if first and chunk_type != "IHDR":
            raise RuntimeError("error found in image chunks.")
        first = False
        if chunk_type == "IHDR":
            WIDTH = int.from_bytes(chunk_data[:4], byteorder="big")
            HEIGHT = int.from_bytes(chunk_data[4:8], byteorder="big")
            BIT_DEPTH = int(chunk_data[8])
            COLOR_TYPE = int(chunk_data[9])
            compression_method = int(chunk_data[10])
            filter_method = int(chunk_data[11])
            interlace_method = int(chunk_data[12])
            if not WIDTH or not HEIGHT or not BIT_DEPTH or not COLOR_TYPE:
                raise RuntimeError("error in image metadata.")
            if compression_method != 0 or filter_method != 0:
                raise RuntimeError("error in image metadata.")
            if interlace_method == 1:
                raise RuntimeError("Adam7 interlace is not supported.")
        elif chunk_type ==  "PLTE":
            if COLOR_TYPE == 2 or COLOR_TYPE == 6:
                ...
            elif COLOR_TYPE == 0 or COLOR_TYPE == 4:
                raise RuntimeError(f"PLTE chunk in a {COLOR_TYPE} color type.")
            else:
                if chunk_data_len % 3 != 0:
                    raise RuntimeError(f"PLTE data len not divisile by 3.")
                number_of_colors = chunk_data_len // 3
                for _ in range(number_of_colors):
                    r = chunk_data[0]
                    g = chunk_data[1]
                    b = chunk_data[2]
                    COLOR_LIST.append((int(r), int(g), int(b)))
                    chunk_data = chunk_data[3:]
        elif chunk_type == "IDAT":
            image_data.append(chunk_data)
        elif chunk_type == "IEND":
            valid = True
            break
        chunks = chunks[12+chunk_data_len:]
    if not valid:
        raise RuntimeError("Image is corrupted or truncated.")

    image_data_joined = b''.join(image_data)

    canvas = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    png_populate_canvas(canvas, (WIDTH, HEIGHT), BIT_DEPTH, image_data_joined, COLOR_TYPE, COLOR_LIST)
    buffer = pygame.surfarray.pixels3d(canvas)
    buffer = np.transpose(buffer, (1, 0, 2))
    alpha = pygame.surfarray.pixels_alpha(canvas)
    alpha = np.transpose(alpha, (1, 0))
    rgba_buffer = np.dstack((buffer, alpha))
    sixel_string = teimpy.get_sixel_string(rgba_buffer)
    sys.stdout.write(sixel_string)
    sys.stdout.flush()

def render_jpg_image(data: bytes) -> None:
    ...

def main() -> None:
    args = sys.argv[1:]
    if len(args) != 1:
        print("This program can only take one argument.")
        sys.exit(1)
    image_path = Path(args[0])
    extention = "".join(image_path.suffix)
    if extention not in (".jpg", ".png"):
        print("This program only supports these formats {PNG, JPG}.")
        sys.exit(1)
    else:
        FORMAT = extention.strip(".")
    try:
        with open(image_path, "br") as f:
            data = f.read()
            if FORMAT == "jpg":
                render_jpg_image(data)
            else:
                render_png_image(data)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


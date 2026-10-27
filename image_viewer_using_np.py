import sys
import zlib
import time
import pygame
import numpy as np
from pathlib import Path
from enum import IntEnum

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


def make_inter_matrix(uncomp_data, height, bpp, bpr) -> np.ndarray:
    image_dataM = np.frombuffer(uncomp_data, dtype=np.uint8).reshape(height, 1 + bpr)
    filter_types = image_dataM[:, 0]
    filt_data = image_dataM[:, 1:]
    inter = np.zeros((height + 1, bpr), dtype=np.uint8)
    n_groups = bpr // bpp
    for i in range(height):
        ft = filter_types[i]
        filt = filt_data[i]
        prev = inter[i]
        out = inter[i + 1]
        if ft == 0:
            out[:] = filt
        elif ft == 1:
            out[:] = np.cumsum(filt.reshape(n_groups, bpp), axis=0, dtype=np.uint8).reshape(-1)
        elif ft == 2:
            out[:] = ((filt.astype(np.uint16) + prev) & 255)
        elif ft == 3:
            for x in range(bpr):
                a = int(out[x - bpp]) if x >= bpp else 0
                out[x] = (filt[x] + ((a + int(prev[x])) >> 1)) & 255
        elif ft == 4:
            for x in range(bpr):
                a = int(out[x - bpp]) if x >= bpp else 0
                b = int(prev[x])
                c = int(prev[x - bpp]) if x >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                out[x] = (int(filt[x]) + pred) & 255
        else:
            raise RuntimeError(f"Invalid ft '{ft}' on row '{i}'")
    return inter[1:]


def png_populate_canvas(canvas: pygame.Surface, size: tuple, bit_depth: int, image_data_joined: bytes, color_type: int, color_list: list) -> None:
    width, height = size
    uncomp_data = zlib.decompress(image_data_joined)
    if color_type == ColorType.GREYSCALE:
        bpp = 1 if bit_depth in (1, 2, 4, 8) else 2
        if bit_depth <= 8:
            ppb = 8 // bit_depth
            bpr = (width + (ppb - 1)) // ppb
        else:
            bpr = width * 2
            ppb = 1
        interM = make_inter_matrix(uncomp_data, height, bpp, bpr)
        buffer = canvas.get_buffer()
        stride = canvas.get_pitch()
        dst = np.frombuffer(buffer, dtype=np.uint8).reshape(height, stride)
        pixels = dst[:, :width * 4].reshape(height, width, 4)
        if bit_depth == 8:
            gray = interM
        elif bit_depth < 8:
            max_val = (1 << bit_depth) - 1
            shifts = np.array([8 - bit_depth - k * bit_depth for k in range(ppb)], dtype=np.uint8)
            gray = ((interM[:, :, None] >> shifts) & max_val).reshape(height, bpr * ppb)[:, :width]
            gray = (gray * (255 // max_val)).astype(np.uint8)
        else:
            gray = interM.reshape(height, width, 2)[..., 0]
        pixels[..., 0] = gray
        pixels[..., 1] = gray
        pixels[..., 2] = gray
        pixels[..., 3] = 255
        del pixels, dst, buffer
    elif color_type == ColorType.TRUECOLOR:
        if bit_depth not in (8, 16):
            raise RuntimeError(f"bit depth '{bit_depth}' not allowed in color type '{color_type}'.")
        if bit_depth == 8:
            bpp = 3
            bpr = width * 3
        else:
            bpp = 6
            bpr = width * 6
        interM = make_inter_matrix(uncomp_data, height, bpp, bpr)
        buffer = canvas.get_buffer()
        stride = canvas.get_pitch()
        dst = np.frombuffer(buffer, dtype=np.uint8).reshape(height, stride)
        pixels = dst[:, :width * 4].reshape(height, width, 4)
        if bit_depth == 8:
            rgb = interM.reshape(height, width, 3)
            pixels[..., 0] = rgb[..., 2]
            pixels[..., 1] = rgb[..., 1]
            pixels[..., 2] = rgb[..., 0]
            pixels[..., 3] = 255
        else:
            rgb = interM.reshape(height, width, 3, 2)
            pixels[..., 0] = rgb[..., 2, 0]
            pixels[..., 1] = rgb[..., 1, 0]
            pixels[..., 2] = rgb[..., 0, 0]
            pixels[..., 3] = 255
        del pixels, dst, buffer
    elif color_type == ColorType.INDEXEDCOLOR:
        if bit_depth not in (1, 2, 4, 8):
            raise RuntimeError(f"bit depth '{bit_depth}' not allowed in color type '{color_type}'.")
        bpp = 1 
        ppb = 8 // bit_depth
        bpr = (width + (ppb - 1)) // ppb
        interM = make_inter_matrix(uncomp_data, height, bpp, bpr)
        buffer = canvas.get_buffer()
        stride = canvas.get_pitch()
        dst = np.frombuffer(buffer, dtype=np.uint8).reshape(height, stride)
        palette_np = np.array(color_list, dtype=np.uint8)
        if bit_depth == 8:
            index = interM
        else:
            max_val = (1 << bit_depth) - 1
            shifts = np.array([8 - bit_depth - k * bit_depth for k in range(ppb)], dtype=np.uint8)
            index = ((interM[:, :, None] >> shifts) & max_val).reshape(height, bpr * ppb)[:, :width]
        pixels = dst[:, :width * 4].reshape(height, width, 4)
        rgb = palette_np[index]
        pixels[..., 0] = rgb[..., 2]
        pixels[..., 1] = rgb[..., 1]
        pixels[..., 2] = rgb[..., 0]
        pixels[..., 3] = 255
        del pixels, dst, buffer
    elif color_type == ColorType.GREYSCALE_WITH_ALPHA:
        if bit_depth not in (8, 16):
            raise RuntimeError(f"bit depth '{bit_depth}' not allowed in color type '{color_type}'.")
        if bit_depth == 8:
            bpp = 2
            bpr = width * 2
        else:
            bpp = 4
            bpr = width * 4
        interM = make_inter_matrix(uncomp_data, height, bpp, bpr)
        buffer = canvas.get_buffer()
        stride = canvas.get_pitch()
        dst = np.frombuffer(buffer, dtype=np.uint8).reshape(height, stride)
        pixels = dst[:, :width * 4].reshape(height, width, 4)
        if bit_depth == 8:
            rgb = interM.reshape(height, width, 2)
            pixels[..., 0] = rgb[..., 0]
            pixels[..., 1] = rgb[..., 0]
            pixels[..., 2] = rgb[..., 0]
            pixels[..., 3] = rgb[..., 1]
        else:
            rgb = interM.reshape(height, width, 2, 2)
            pixels[..., 0] = rgb[..., 0, 0]
            pixels[..., 1] = rgb[..., 0, 0]
            pixels[..., 2] = rgb[..., 0, 0]
            pixels[..., 3] = rgb[..., 1, 0]
        del pixels, dst, buffer
    elif color_type == ColorType.TRUECOLOR_WITH_ALPHA:
        if bit_depth not in (8, 16):
            raise RuntimeError(f"bit depth '{bit_depth}' not allowed in color type '{color_type}'.")
        if bit_depth == 8:
            bpp = 4
            bpr = width * 4
        else:
            bpp = 8
            bpr = width * 8
        interM = make_inter_matrix(uncomp_data, height, bpp, bpr)
        buffer = canvas.get_buffer()
        stride = canvas.get_pitch()
        dst = np.frombuffer(buffer, dtype=np.uint8).reshape(height, stride)
        pixels = dst[:, :width * 4].reshape(height, width, 4)
        if bit_depth == 8:
            rgb = interM.reshape(height, width, 4)
            pixels[..., 0] = rgb[..., 2]
            pixels[..., 1] = rgb[..., 1]
            pixels[..., 2] = rgb[..., 0]
            pixels[..., 3] = rgb[..., 3]
        else:
            rgb = interM.reshape(height, width, 4, 2)
            pixels[..., 0] = rgb[..., 2, 0]
            pixels[..., 1] = rgb[..., 1, 0]
            pixels[..., 2] = rgb[..., 0, 0]
            pixels[..., 3] = rgb[..., 3, 0]
        del pixels, dst, buffer


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


    time_stamp = time.time()
    canvas = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    png_populate_canvas(canvas, (WIDTH, HEIGHT), BIT_DEPTH, image_data_joined, COLOR_TYPE, COLOR_LIST)
    print(f"timed: {time.time() - time_stamp}")

    running = True
    clock = pygame.time.Clock()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        window.fill("black")
        window.blit(canvas, (0, 0))
        pygame.display.flip()
        clock.tick(60)



    # print(WIDTH, HEIGHT, BIT_DEPTH, COLOR_LIST)
    # print(image_data)


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

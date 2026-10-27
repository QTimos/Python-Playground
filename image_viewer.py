import time
import sys
import math
import zlib
from enum import IntEnum
import pygame
from pathlib import Path
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


def png_populate_canvas(canvas: pygame.Surface, size: tuple, bit_depth: int, image_data_joined: bytes, color_type: int, color_list: list) -> None:
    width, height = size
    uncomp_data = zlib.decompress(image_data_joined)
    if color_type == ColorType.GREYSCALE:
        rows = []
        bpp = 1 if bit_depth in (1, 2, 4, 8) else 2
        if bit_depth <= 8:
            ppb = 8 // bit_depth
            bpr = (width + (ppb - 1)) // ppb
        else:
            bpr = width * 2
            ppb = 1
        stride = 1 + bpr
        for i, s in enumerate(range(0, len(uncomp_data), stride)):
            scanline = uncomp_data[s: s + stride]
            filter_type = scanline[0]
            raw_row_bytes = scanline[1:]
            prev_row = rows[i-1] if rows else bytearray([0] * bpr)
            row_buff = bytearray(bpr)
            for x in range(0, bpr):
                filt_byte = raw_row_bytes[x]
                if filter_type == 0:
                    recon_byte = filt_byte % 256
                elif filter_type == 1:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    recon_byte = (filt_byte + a) % 256
                elif filter_type == 2:
                    b = prev_row[x]
                    recon_byte = (filt_byte + b) % 256
                elif filter_type == 3:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    b = prev_row[x]
                    av = math.floor((a + b) / 2)
                    recon_byte = (filt_byte + av) % 256
                elif filter_type == 4:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    b = prev_row[x]
                    c = prev_row[x - bpp] if x >= bpp else 0
                    initial_p = a + b - c
                    pa = abs(initial_p - a)
                    pb = abs(initial_p - b)
                    pc = abs(initial_p - c)
                    if pa <= pb and pa <= pc:
                        p = a
                    elif pb <= pc:
                        p = b
                    else:
                        p = c
                    recon_byte = (filt_byte + p) % 256
                row_buff[x] = recon_byte
            rows.append(row_buff)
        max_value = (1 << bit_depth) - 1
        for i, row in enumerate(rows):
            y = i
            for pixel_x in range(width):
                x = pixel_x
                if bit_depth <= 8:
                    byte_idx = pixel_x // ppb
                    shift_distance = 8 - bit_depth - ((pixel_x % ppb) * bit_depth)
                    raw_val = (row[byte_idx] >> shift_distance) & max_value
                    inten = int((raw_val / max_value))
                    color = ((inten * 255), (inten * 255), (inten * 255))
                else:
                    byte_idx = pixel_x * 2
                    high_byte = row[byte_idx]
                    low_byte = row[byte_idx + 1]
                    raw_val = (high_byte << 8) | low_byte
                    inten = raw_val >> 8
                    color = (inten, inten, inten)
                canvas.set_at((x, y), color)
    elif color_type == ColorType.TRUECOLOR:
        if bit_depth not in (8, 16):
            raise RuntimeError(f"bit depth '{bit_depth}' not allowed in color type '{color_type}'.")
        if bit_depth == 8:
            bpp = 3
            bpr = width *3
        else:
            bpp = 6
            bpr = width * 6
        stride = 1 + bpr
        rows = []
        for i, s in enumerate(range(0, len(uncomp_data), stride)):
            scanline = uncomp_data[s: s + stride]
            filter_type = scanline[0]
            raw_row_bytes = scanline[1:]
            prev_row = rows[i-1] if rows else bytearray([0] * bpr)
            row_buff = bytearray(bpr)
            for x in range(0, bpr):
                filt_byte = raw_row_bytes[x]
                if filter_type == 0:
                    recon_byte = filt_byte % 256
                elif filter_type == 1:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    recon_byte = (filt_byte + a) % 256
                elif filter_type == 2:
                    b = prev_row[x]
                    recon_byte = (filt_byte + b) % 256
                elif filter_type == 3:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    b = prev_row[x]
                    av = math.floor((a + b) / 2)
                    recon_byte = (filt_byte + av) % 256
                elif filter_type == 4:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    b = prev_row[x]
                    c = prev_row[x - bpp] if x >= bpp else 0
                    initial_p = a + b - c
                    pa = abs(initial_p - a)
                    pb = abs(initial_p - b)
                    pc = abs(initial_p - c)
                    if pa <= pb and pa <= pc:
                        p = a
                    elif pb <= pc:
                        p = b
                    else:
                        p = c
                    recon_byte = (filt_byte + p) % 256
                row_buff[x] = recon_byte
            rows.append(row_buff)
        for i, row in enumerate(rows):
            y = i
            for pixel_x in range(width):
                x = pixel_x
                if bit_depth == 8:
                    byte_idx = pixel_x * 3
                    r = row[byte_idx]
                    g = row[byte_idx + 1]
                    b = row[byte_idx + 2]
                    color = (r, g, b)
                else:
                    byte_idx = pixel_x * 6
                    r = row[byte_idx]
                    g = row[byte_idx + 2]
                    b = row[byte_idx + 4]
                    color = (r, g, b)
                canvas.set_at((x, y), color)
    elif color_type == ColorType.INDEXEDCOLOR:
        if bit_depth not in (1, 2, 4, 8):
            raise RuntimeError(f"bit depth '{bit_depth}' not allowed in color type '{color_type}'.")
        rows = []
        bpp = 1 
        ppb = 8 // bit_depth
        bpr = (width + (ppb - 1)) // ppb
        stride = 1 + bpr
        for i, s in enumerate(range(0, len(uncomp_data), stride)):
            scanline = uncomp_data[s: s + stride]
            filter_type = scanline[0]
            raw_row_bytes = scanline[1:]
            prev_row = rows[i-1] if rows else bytearray([0] * bpr)
            row_buff = bytearray(bpr)
            for x in range(0, bpr):
                filt_byte = raw_row_bytes[x]
                if filter_type == 0:
                    recon_byte = filt_byte % 256
                elif filter_type == 1:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    recon_byte = (filt_byte + a) % 256
                elif filter_type == 2:
                    b = prev_row[x]
                    recon_byte = (filt_byte + b) % 256
                elif filter_type == 3:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    b = prev_row[x]
                    av = math.floor((a + b) / 2)
                    recon_byte = (filt_byte + av) % 256
                elif filter_type == 4:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    b = prev_row[x]
                    c = prev_row[x - bpp] if x >= bpp else 0
                    initial_p = a + b - c
                    pa = abs(initial_p - a)
                    pb = abs(initial_p - b)
                    pc = abs(initial_p - c)
                    if pa <= pb and pa <= pc:
                        p = a
                    elif pb <= pc:
                        p = b
                    else:
                        p = c
                    recon_byte = (filt_byte + p) % 256
                row_buff[x] = recon_byte
            rows.append(row_buff)
        max_value = (1 << bit_depth) - 1
        for i, row in enumerate(rows):
            y = i
            for pixel_x in range(width):
                x = pixel_x
                byte_idx = pixel_x // ppb
                shift_distance = 8 - bit_depth - ((pixel_x % ppb) * bit_depth)
                raw_val = (row[byte_idx] >> shift_distance) & max_value
                color = color_list[raw_val]
                canvas.set_at((x, y), color)
    elif color_type == ColorType.GREYSCALE_WITH_ALPHA:
        if bit_depth not in (8, 16):
            raise RuntimeError(f"bit depth '{bit_depth}' not allowed in color type '{color_type}'.")
        rows = []
        if bit_depth == 8:
            bpp = 2
            bpr = width * 2
        else:
            bpp = 4
            bpr = width * 4
        stride = 1 + bpr
        for i, s in enumerate(range(0, len(uncomp_data), stride)):
            scanline = uncomp_data[s: s + stride]
            filter_type = scanline[0]
            raw_row_bytes = scanline[1:]
            prev_row = rows[i-1] if rows else bytearray([0] * bpr)
            row_buff = bytearray(bpr)
            for x in range(0, bpr):
                filt_byte = raw_row_bytes[x]
                if filter_type == 0:
                    recon_byte = filt_byte % 256
                elif filter_type == 1:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    recon_byte = (filt_byte + a) % 256
                elif filter_type == 2:
                    b = prev_row[x]
                    recon_byte = (filt_byte + b) % 256
                elif filter_type == 3:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    b = prev_row[x]
                    av = math.floor((a + b) / 2)
                    recon_byte = (filt_byte + av) % 256
                elif filter_type == 4:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    b = prev_row[x]
                    c = prev_row[x - bpp] if x >= bpp else 0
                    initial_p = a + b - c
                    pa = abs(initial_p - a)
                    pb = abs(initial_p - b)
                    pc = abs(initial_p - c)
                    if pa <= pb and pa <= pc:
                        p = a
                    elif pb <= pc:
                        p = b
                    else:
                        p = c
                    recon_byte = (filt_byte + p) % 256
                row_buff[x] = recon_byte
            rows.append(row_buff)
        max_value = (1 << bit_depth) - 1
        for i, row in enumerate(rows):
            y = i
            for pixel_x in range(width):
                x = pixel_x
                if bit_depth == 8:
                    byte_idx = pixel_x * 2
                    gray = row[byte_idx]
                    alpha = row[byte_idx + 1]
                    color = (gray, gray, gray, alpha)
                else:
                    byte_idx = pixel_x * 4
                    gray = row[byte_idx]
                    alpha = row[byte_idx + 2]
                    color = (gray, gray, gray, alpha)
                canvas.set_at((x, y), color)
    elif color_type == ColorType.TRUECOLOR_WITH_ALPHA:
        if bit_depth not in (8, 16):
            raise RuntimeError(f"bit depth '{bit_depth}' not allowed in color type '{color_type}'.")
        if bit_depth == 8:
            bpp = 4
            bpr = width * 4
        else:
            bpp = 8
            bpr = width * 8
        stride = 1 + bpr
        rows = []
        for i, s in enumerate(range(0, len(uncomp_data), stride)):
            scanline = uncomp_data[s: s + stride]
            filter_type = scanline[0]
            raw_row_bytes = scanline[1:]
            prev_row = rows[i-1] if rows else bytearray([0] * bpr)
            row_buff = bytearray(bpr)
            for x in range(0, bpr):
                filt_byte = raw_row_bytes[x]
                if filter_type == 0:
                    recon_byte = filt_byte % 256
                elif filter_type == 1:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    recon_byte = (filt_byte + a) % 256
                elif filter_type == 2:
                    b = prev_row[x]
                    recon_byte = (filt_byte + b) % 256
                elif filter_type == 3:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    b = prev_row[x]
                    av = math.floor((a + b) / 2)
                    recon_byte = (filt_byte + av) % 256
                elif filter_type == 4:
                    if x >= bpp:
                        a = row_buff[x - bpp]
                    else:
                        a = 0
                    b = prev_row[x]
                    c = prev_row[x - bpp] if x >= bpp else 0
                    initial_p = a + b - c
                    pa = abs(initial_p - a)
                    pb = abs(initial_p - b)
                    pc = abs(initial_p - c)
                    if pa <= pb and pa <= pc:
                        p = a
                    elif pb <= pc:
                        p = b
                    else:
                        p = c
                    recon_byte = (filt_byte + p) % 256
                row_buff[x] = recon_byte
            rows.append(row_buff)
        for i, row in enumerate(rows):
            y = i
            for pixel_x in range(width):
                x = pixel_x
                if bit_depth == 8:
                    byte_idx = pixel_x * 4
                    r = row[byte_idx]
                    g = row[byte_idx + 1]
                    b = row[byte_idx + 2]
                    a = row[byte_idx + 3]
                    color = (r, g, b, a)
                else:
                    byte_idx = pixel_x * 8
                    r = row[byte_idx]
                    g = row[byte_idx + 2]
                    b = row[byte_idx + 4]
                    a = row[byte_idx + 6]
                    color = (r, g, b, a)
                canvas.set_at((x, y), color)


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
    print(time.time() - time_stamp)

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

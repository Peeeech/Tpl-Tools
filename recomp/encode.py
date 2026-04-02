import os
from PIL import Image #type: ignore
import struct
import re
import numpy as np #type: ignore

TPL_FORMATS = {
    "I4": 0x00,
    "I8": 0x01,
    "IA4": 0x02,
    "IA8": 0x03,
    "RGB565": 0x04,
    "RGB5A3": 0x05,
    "RGBA32": 0x06,
    "C4": 0x08,
    "C8": 0x09,
    "CMPR": 0x0E,
}

def analyze_colors_rgb5a3(img):
    pixels = img.getdata()

    palette = []
    palette_map = {}
    indices = []

    for r, g, b, a in pixels:
        if a < 255:
            val = (
                ((a // 32) << 12) |
                ((r // 17) << 8) |
                ((g // 17) << 4) |
                (b // 17)
            )
        else:
            val = (
                0x8000 |
                ((r >> 3) << 10) |
                ((g >> 3) << 5) |
                (b >> 3)
            )

        if val not in palette_map:
            palette_map[val] = len(palette)
            palette.append(val)

            if len(palette) > 256:
                return None  # early out

        indices.append(palette_map[val])

    return {
        "colors": set(palette),
        "palette": palette,
        "indices": indices,
    }

def image_to_pil(bl_img):
    """Convert bpy.types.Image pixels (float 0..1) to a PIL RGBA image."""
    w, h = bl_img.size
    ch = bl_img.channels

    # store pixels as a flat float array
    arr = np.asarray(bl_img.pixels[:], dtype=np.float32)
    arr = arr.reshape((h, w, ch))

    arr = np.flipud(arr)

    # Expand to RGBA in float space
    if ch == 4:
        rgba = arr
    elif ch == 3:
        a = np.ones((h, w, 1), dtype=np.float32)
        rgba = np.concatenate([arr, a], axis=2)
    elif ch == 2:
        # LA: intensity + alpha -> RGB=intensity, A=alpha
        I = arr[:, :, 0:1]
        A = arr[:, :, 1:2]
        rgb = np.repeat(I, 3, axis=2)
        rgba = np.concatenate([rgb, A], axis=2)
    elif ch == 1:
        I = arr[:, :, 0:1]
        rgb = np.repeat(I, 3, axis=2)
        a = np.ones((h, w, 1), dtype=np.float32)
        rgba = np.concatenate([rgb, a], axis=2)
    else:
        raise ValueError(f"Unsupported channel count: {ch}")

    rgba8 = np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba8, mode="RGBA")

def detect_format_pil(img_obj):
    """Same intent as detect_format(path), but operates on a PIL RGBA image."""
    img = img_obj.convert("RGBA")
    width, height = img.size
    pixels = list(img.getdata())

    has_alpha = False
    grayscale_values = set()
    alpha_values = set()
    rgb565_safe = True
    rgb5a3_safe = True
    grayscale_detected = True

    for r, g, b, a in pixels:
        if a < 255:
            has_alpha = True
            alpha_values.add(a)

        if r != g or g != b:
            grayscale_detected = False

        if grayscale_detected:
            grayscale_values.add(r)

        if r % 8 != 0 or g % 4 != 0 or b % 8 != 0:
            rgb565_safe = False

        if a < 255:
            if not all(is_close_to(v, 17, tolerance=8) for v in (a, r, g, b)):
                rgb5a3_safe = False
        else:
            if not all(is_close_to(v, 8, tolerance=8) for v in (r, g, b)):
                rgb5a3_safe = False

    if grayscale_detected:
        if has_alpha:
            if any(v % 17 != 0 for v in grayscale_values) or any(a % 17 != 0 for a in alpha_values):
                return "IA8", None
            if len(grayscale_values) <= 16 and len(alpha_values) <= 16:
                return "IA4", None
            return "IA8", None
        else:
            if any(v % 17 != 0 for v in grayscale_values):
                return "I8", None
            if len(grayscale_values) <= 16:
                return "I4", None
            return "I8", None

    if not has_alpha and rgb565_safe:
        return "RGB565", None
    color_data = analyze_colors_rgb5a3(img)
    color_count = len(color_data["palette"]) if color_data else None

    if color_count and color_count <= 16:
        return "C4", color_data
    
    if is_cmpr_compatible(pixels, width, height, debug=False):
        return "CMPR", None
    
    if color_count and color_count <= 256:
        return "C8", color_data

    if rgb5a3_safe:
        return "RGB5A3", None

    return "RGBA32", None

def encode_pil_image(img_obj, fmt, palette=None, quality=False):
    img = img_obj

    # ------------------------------------------------------------------
    # I4  (4‑bit intensity, 8×8 tiled, 32 bytes per tile)
    # ------------------------------------------------------------------
    if fmt == "I4":
        #print(idx, img, fmt)
        img_obj = img.convert("L")   # 8‑bit grayscale
        width, height = img_obj.size
        gray_pixels = list(img_obj.getdata())

        tile_w, tile_h = 8, 8
        tiles_x = (width  + tile_w - 1) // tile_w
        tiles_y = (height + tile_h - 1) // tile_h

        encoded = bytearray()

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for row in range(tile_h):
                    iy = ty * tile_h + row
                    if iy >= height:
                        break
                    for col in range(0, tile_w, 2):          # 2 pixels per byte
                        ix0 = tx * tile_w + col
                        ix1 = ix0 + 1
                        if ix0 >= width:
                            break

                        # First pixel (high nibble)
                        I0 = gray_pixels[iy * width + ix0] if ix0 < width else 0
                        nib0 = max(0, min(15, round(I0 / 17)))   # 0‑15

                        # Second pixel (low nibble)
                        if ix1 < width and iy < height:
                            I1 = gray_pixels[iy * width + ix1]
                            nib1 = max(0, min(15, round(I1 / 17)))
                        else:
                            nib1 = 0

                        encoded.append((nib0 << 4) | nib1)

    # ------------------------------------------------------------------
    # I8  (8-bit intensity, 8×4 tiled, 32 bytes per tile)
    # ------------------------------------------------------------------
    elif fmt == "I8":
        #print(idx, img, fmt)
        img_obj = img.convert("L")   # 8-bit grayscale
        width, height = img_obj.size
        gray_pixels = list(img_obj.getdata())

        tile_w, tile_h = 8, 4
        tiles_x = (width  + tile_w - 1) // tile_w
        tiles_y = (height + tile_h - 1) // tile_h

        encoded = bytearray()

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                base_x = tx * tile_w
                base_y = ty * tile_h

                for row in range(tile_h):
                    iy = base_y + row
                    if iy >= height:
                        continue
                    for col in range(tile_w):
                        ix = base_x + col
                        if ix >= width:
                            break

                        # Raw grayscale 0–255
                        I = gray_pixels[iy * width + ix] if ix < width else 0
                        encoded.append(I)

    elif fmt == "IA4":
        #print(idx, img, fmt)
        img_obj = img.convert("LA")
        width, height = img_obj.size
        gray_alpha_pixels = list(img_obj.getdata())

        tile_w, tile_h = 8, 4
        tiles_x = (width + tile_w - 1) // tile_w
        tiles_y = (height + tile_h - 1) // tile_h

        encoded = bytearray()
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for row in range(tile_h):
                    iy = ty * tile_h + row
                    if iy >= height:
                        break
                    for col in range(tile_w):
                        ix = tx * tile_w + col
                        if ix >= width:
                            break
                        I, A = gray_alpha_pixels[iy * width + ix]
                        i4 = max(0, min(15, round(I / 17)))
                        a4 = max(0, min(15, round(A / 17)))
                        encoded.append((a4 << 4) | i4)

    elif fmt == "IA8":
        #print(idx, img, fmt)
        img_obj = img.convert("LA")
        width, height = img_obj.size
        pixels = img_obj.load()

        tile_w, tile_h = 4, 4
        tiles_x = (width + tile_w - 1) // tile_w
        tiles_y = (height + tile_h - 1) // tile_h

        encoded = bytearray()

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for row in range(tile_h):
                    iy = ty * tile_h + row
                    if iy >= height:
                        break
                    for col in range(tile_w):
                        ix = tx * tile_w + col
                        if ix >= width:
                            break
                        I, A = pixels[ix, iy]  # ← this is important: x, y
                        encoded.append(A)
                        encoded.append(I)

    elif fmt == "RGB5A3":
        #print(idx, img, fmt)
        img_obj = img.convert("RGBA")
        width, height = img_obj.size
        rgba_pixels = list(img_obj.getdata())

        tile_w, tile_h = 4, 4
        tiles_x = (width + tile_w - 1) // tile_w
        tiles_y = (height + tile_h - 1) // tile_h

        encoded = bytearray()
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for row in range(tile_h):
                    iy = ty * tile_h + row
                    if iy >= height:
                        break
                    for col in range(tile_w):
                        ix = tx * tile_w + col
                        if ix >= width:
                            break
                        r, g, b, a = rgba_pixels[iy * width + ix]
                        if a < 255:
                            a4 = round(a / 17) & 0xF
                            r4 = round(r / 17) & 0xF
                            g4 = round(g / 17) & 0xF
                            b4 = round(b / 17) & 0xF
                            val = (a4 << 12) | (r4 << 8) | (g4 << 4) | b4
                        else:
                            r5 = round(r * 31 / 255) & 0x1F
                            g5 = round(g * 31 / 255) & 0x1F
                            b5 = round(b * 31 / 255) & 0x1F
                            val = 0x8000 | (r5 << 10) | (g5 << 5) | b5
                        encoded += struct.pack(">H", val)

    elif fmt == "CMPR":
        #print(idx, img, fmt)
        img_obj = img.convert("RGBA")
        width, height = img_obj.size
        pixels = img_obj.load()

        encoded = bytearray()

        for by in range(0, height, 8):
            for bx in range(0, width, 8):
                for subblock_y in range(2):
                    for subblock_x in range(2):
                        block = []
                        for y in reversed(range(4)):
                            row = []
                            for x in reversed(range(4)):
                                px = bx + subblock_x * 4 + x
                                py = by + subblock_y * 4 + y
                                if px < width and py < height:
                                    row.append(pixels[px, py])
                                else:
                                    row.append((0, 0, 0, 0))  # transparent pad
                            block.append(row)
                        encoded += encode_cmpr_block(block, quality)

    elif fmt in ("C4", "C8"):
        if palette is None:
            print("Palette failed to be grabbed at encode")
            return
        
        img_obj = img.convert("RGBA")
        width, height = img_obj.size
        
        palette_obj = palette["palette_obj"]
        local_palette = palette["local_palette"]
        indices = palette["indices"]

        global_palette = palette_obj["palette"]
        global_map = palette_obj["map"]

        palette_data = bytearray()
        for color in global_palette:
            palette_data += struct.pack(">H", color)

        palette_header = struct.pack(
            ">HBBII",
            len(global_palette),  # number of entries
            0,                    # unpacked
            0,                    # pad
            0x02,                 # format (RGB5A3)
            0                     # data_addr (filled later)
        )

        remap = {
            i: global_map[color]
            for i, color in enumerate(local_palette)
        }

        remapped = [remap[i] for i in indices]

        if fmt == "C4":
            tile_w, tile_h = 8, 8
        elif fmt == "C8":
            tile_w, tile_h = 8, 4

        tiled = []

        for ty in range(0, height, tile_h):
            for tx in range(0, width, tile_w):
                for y in range(tile_h):
                    for x in range(tile_w):
                        px = tx + x
                        py = ty + y

                        if px >= width or py >= height:
                            tiled.append(0)
                        else:
                            tiled.append(remapped[py * width + px])

        if fmt == "C4" and len(global_palette) <= 16:
            encoded = bytearray()
            for i in range(0, len(tiled), 2):
                encoded.append((tiled[i] << 4) | (tiled[i+1] if i+1 < len(tiled) else 0))

        elif fmt == "C8" and len(tiled) >= 1 and len(global_palette) <= 256:
            encoded = bytearray(tiled)

    elif fmt == "RGB565":
        #print(idx, img, fmt)
        img_obj = img.convert("RGB")
        width, height = img_obj.size
        rgb_pixels = list(img_obj.getdata())

        tile_w, tile_h = 4, 4
        tiles_x = (width + tile_w - 1) // tile_w
        tiles_y = (height + tile_h - 1) // tile_h

        encoded = bytearray()
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for row in range(tile_h):
                    iy = ty * tile_h + row
                    if iy >= height:
                        break
                    for col in range(tile_w):
                        ix = tx * tile_w + col
                        if ix >= width:
                            break
                        r, g, b = rgb_pixels[iy * width + ix]
                        r5 = (r * 31 + 127) // 255
                        g6 = (g * 63 + 127) // 255
                        b5 = (b * 31 + 127) // 255
                        val = (r5 << 11) | (g6 << 5) | b5
                        encoded += struct.pack(">H", val)

    elif fmt == "RGBA32":
        #print(idx, img, fmt)
        img_obj = img.convert("RGBA")
        width, height = img_obj.size
        rgba_pixels = list(img_obj.getdata())

        tile_w, tile_h = 4, 4
        tiles_x = (width + tile_w - 1) // tile_w
        tiles_y = (height + tile_h - 1) // tile_h

        encoded = bytearray()
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # GX RGBA32 stores a 4x4 tile as:
                # 16 pixels of (A,R) 16-bit words, then 16 pixels of (G,B) 16-bit words
                ar = bytearray()
                gb = bytearray()
                for row in range(tile_h):
                    iy = ty * tile_h + row
                    if iy >= height:
                        break
                    for col in range(tile_w):
                        ix = tx * tile_w + col
                        if ix >= width:
                            break
                        r, g, b, a = rgba_pixels[iy * width + ix]
                        ar += struct.pack(">BB", a, r)
                        gb += struct.pack(">BB", g, b)
                encoded += ar + gb

    else:
        print(f"\n\nERROR:\n\n{img}, {fmt}\n\n")
        return None
    
    # --- TPL image header ---
    image_header = struct.pack(
    ">HHIIIIIIfBBBB",
    height,
    width,
    TPL_FORMATS[fmt],  # format
    0,                 # data_addr (placeholder, fill later)
    1, 1,              # wrap_s, wrap_t
    1, 1,              # min_filter, mag_filter
    0.0,               # lod_bias
    0, 0, 0,         # edge_lod_enable, min_lod, max_lod
    0               # unused padding byte
    )

    if fmt in ("C4", "C8"):
        return fmt, image_header, encoded, palette_header, palette_data
    else:
        return fmt, image_header, encoded, None, None

def natural_sort_key(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', text)]

def is_close_to(val, step, tolerance=4): 
    return any(abs(val - (i * step)) <= tolerance for i in range(256 // step + 1))

def is_cmpr_compatible(pixels, width, height, debug):
    # CMPR encodes in 4x4 blocks, each with max 4 colors (2 base + 2 interpolated)
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            colors = set()
            for j in range(4):
                for i in range(4):
                    ix = x + i
                    iy = y + j
                    if ix >= width or iy >= height:
                        continue
                    r, g, b, a = pixels[iy * width + ix]

                    # Alpha must be 0 or 255 in CMPR
                    if a not in (0, 255):
                        if debug:
                            print(f"\nBlock at ({x},{y}) failed: alpha={a} not 0 or 255\n")
                        return False

                    # Convert to RGB565 to determine uniqueness in CMPR space
                    rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
                    colors.add(rgb565)

                    if len(colors) > 4:
                        if debug:
                            print(f"\nBlock at ({x},{y}) failed: {len(colors)} unique RGB565 colors\n")
                        return False
    return True

def _scan(folder_path):
    scanned = []

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        return

    png_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith('.png')],
        key=natural_sort_key
    )

    if not png_files:
        print("No .png files found in the directory.")
        return

    print(f"Found {len(png_files)} .png file(s):")

    for img in png_files:
        path = os.path.join(folder_path, img)

        img_obj = Image.open(path)

        detected_fmt, palette_data = detect_format_pil(img_obj)
        scanned.append((img_obj, img, detected_fmt, palette_data))

    return scanned

def prep(imgList, quality, compress=False, threshold=100):
    
    data = []
    for img_obj, name, format, palette in imgList:
        final = encode_pil_image(img_obj, format, palette, quality)
        if not final:
            return

        fmt, image_header, encoded, pal_header, pal_data = final
        size_kb = len(encoded) / 1024

        if compress and fmt != "CMPR" and size_kb > threshold:
            print(f"[CMPR OVERRIDE] {name}: {fmt} -> CMPR ({size_kb:.2f} KB)")

            final = encode_pil_image(img_obj, "CMPR", None, quality)
            if final:
                fmt, image_header, encoded, _, _ = final

        data.append((fmt, image_header, encoded, pal_header, pal_data))

    return data

#cmpr helpers
def rgb888_to_rgb565(r, g, b):
    r5 = (r * 31 + 127) // 255
    g6 = (g * 63 + 127) // 255
    b5 = (b * 31 + 127) // 255
    return (r5 << 11) | (g6 << 5) | b5

def rgb565_to_rgb888(rgb):
    r5 = (rgb >> 11) & 0x1F
    g6 = (rgb >> 5) & 0x3F
    b5 = rgb & 0x1F
    r = (r5 * 255 + 15) // 31
    g = (g6 * 255 + 31) // 63
    b = (b5 * 255 + 15) // 31
    return (r, g, b)

def encode_cmpr_block(block, quality):
    flat = [block[y][x] for y in range(4) for x in range(4)]

    # Fully transparent block
    if all(p[3] < 128 for p in flat):
        color0 = rgb888_to_rgb565(0, 0, 0)
        color1 = rgb888_to_rgb565(0, 0, 1)
        return struct.pack(">HHI", color0, color1, 0xFFFFFFFF)  # all index 3

    # Separate opaque & transparent
    opaque_pixels = [p[:3] for p in flat if p[3] >= 128]
    transparent_pixels_exist = any(p[3] < 128 for p in flat)

    if not opaque_pixels:
        opaque_pixels = [(0, 0, 0)]

    # Pick endpoints
    if quality:
        rgb565_colors = [rgb888_to_rgb565(*p) for p in opaque_pixels]
        rgb888_colors = [rgb565_to_rgb888(c) for c in rgb565_colors]
        max_dist = -1
        color0 = color1 = rgb565_colors[0]
        for i in range(len(rgb888_colors)):
            for j in range(i + 1, len(rgb888_colors)):
                dist = np.sum((np.array(rgb888_colors[i]) - np.array(rgb888_colors[j])) ** 2)
                if dist > max_dist:
                    max_dist = dist
                    color0 = rgb565_colors[i]
                    color1 = rgb565_colors[j]
    else:
        r_vals, g_vals, b_vals = zip(*opaque_pixels)
        max_rgb = (max(r_vals), max(g_vals), max(b_vals))
        min_rgb = (min(r_vals), min(g_vals), min(b_vals))
        color0 = rgb888_to_rgb565(*max_rgb)
        color1 = rgb888_to_rgb565(*min_rgb)

    # Force transparent mode if any pixel is transparent
    force_transparent_mode = transparent_pixels_exist
    if force_transparent_mode:
        if color0 >= color1:
            color0, color1 = color1, color0
    else:
        if color0 <= color1:
            color0, color1 = color1, color0

    # Build palette
    c0 = np.array(rgb565_to_rgb888(color0))
    c1 = np.array(rgb565_to_rgb888(color1))
    if color0 > color1:
        palette = [c0, c1, (2 * c0 + c1) // 3, (c0 + 2 * c1) // 3]
    else:
        palette = [c0, c1, (c0 + c1) // 2, np.array([0, 0, 0])]  # index 3 is transparent

    # Assign indices
    indices = 0
    for i, (r, g, b, a) in enumerate(flat):
        if a < 128:
            index = 3  # force index 3 if any transparency is in block
        else:
            color = np.array([r, g, b])
            dists = [np.sum((color - p) ** 2) for p in palette]
            index = int(np.argmin(dists))

            # Prevent accidentally assigning index 3 to opaque
            if color0 < color1 and index == 3:
                index = 0
        indices |= (index & 0x3) << (2 * i)

    return struct.pack(">HHI", color0, color1, indices)

def write_tpl(data, directory):
    output = bytearray()

    image_count = len(data)
    header_size = 0x0C
    table_size = image_count * 8
    image_headers_size = image_count * 0x24
    palette_headers_size = sum(0x0C for _, _, _, ph, _ in data if ph is not None)

    meta_end = header_size + table_size + image_headers_size + palette_headers_size
    data_start = (meta_end + 0x1F) & ~0x1F

    total_palette_data_size = sum(len(pd) for _, _, _, _, pd in data if pd is not None)

    palette_data_offset = data_start
    image_data_offset = data_start + total_palette_data_size

    output += struct.pack(">I", 0x0020AF30)
    output += struct.pack(">I", image_count)
    output += struct.pack(">I", header_size)

    table_entries = []
    header_data = bytearray()
    palette_header_data = bytearray()

    palette_data_blocks = []
    image_data_blocks = []

    palette_header_offsets = []

    next_palette_header_offset = header_size + table_size + image_headers_size

    for fmt, image_header, img_data, pal_header, pal_data in data:
        if pal_header is not None:
            patched_pal_header = bytearray(pal_header)
            struct.pack_into(">I", patched_pal_header, 0x08, palette_data_offset)

            palette_header_data += patched_pal_header
            palette_header_offsets.append(next_palette_header_offset)
            next_palette_header_offset += 0x0C

            palette_data_blocks.append(pal_data)
            palette_data_offset += len(pal_data)
        else:
            palette_header_offsets.append(0)

    for i, (fmt, image_header, img_data, pal_header, pal_data) in enumerate(data):
        patched_header = bytearray(image_header)
        struct.pack_into(">I", patched_header, 0x08, image_data_offset)

        header_data += patched_header

        image_header_offset = header_size + table_size + len(header_data) - 0x24
        table_entries.append(struct.pack(">II", image_header_offset, palette_header_offsets[i]))

        image_data_blocks.append(img_data)
        image_data_offset += len(img_data)

    for entry in table_entries:
        output += entry

    output += header_data
    output += palette_header_data

    padding_len = data_start - len(output)
    output += b"\x00" * padding_len

    for block in palette_data_blocks:
        output += block

    for block in image_data_blocks:
        output += block

    with open(os.path.abspath(directory), "wb") as f:
        f.write(output)

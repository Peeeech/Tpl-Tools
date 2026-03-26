import os
import sys
import shutil
import argparse
from argparse import BooleanOptionalAction

from decomp import decode, tplparse
from recomp import encode

try:
    from PIL import Image
except Exception:
    print("Failed to import PIL. run `pip install pillow`.")
    sys.exit(1)

def clear_directory(path):
    if not os.path.exists(path):
        return

    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

#write helper
def writeImage(dir, i, image):
    format_type = decode.FORMAT_MAP.get(image.format)
    decode_func = decode.get_format_function(format_type)
    path = os.path.abspath(os.path.join(dir, f"i_{i}_{format_type}_{image.width}x{image.height}.png"))
    if decode_func:
        rgba = decode_func(image.raw_data, image.height, image.width)
        if rgba is None:
            print(f"what the fuck {i} {image.format}")
            sys.exit(1)

        img = Image.frombytes("RGBA", (image.width, image.height), bytes(rgba))
        img.save(path)

        print(f"Saved: {path}")
    else:
        print(f"No decoder for format: {image.format}")

        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompile or Compile TPL files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--decompile", action="store_true")
    group.add_argument("-c", "--compile", action="store_true")

    parser.add_argument("input_file", help="Input file path.")
    
    parser.add_argument("--compress", action=BooleanOptionalAction, default=False,
                    help="Compress too-big textures to CMPR to save memory")

    parser.add_argument("--compression-threshold", type=int, default=100,
                    help="Threshold for compression (default: 100)")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_dir = os.path.join(script_dir, "output")
    try:
        if not os.path.exists(tex_dir):
            os.mkdir(tex_dir)
            print("Created `output` folder.")
    except Exception as e:
        print(f"Failed to create output directory; aborting. {e}")
        sys.exit(1)

    args = parser.parse_args()

    if args.decompile:
        print(f"Decompiling {args.input_file}...")

        if args.input_file == None or not os.path.exists(args.input_file):
            print(f"File {args.input_file} not found.")
            sys.exit(1)
        elif not os.path.isfile(args.input_file):
            print(f"Directory {args.input_file} is not a file.")
            sys.exit(1)

        header, images = tplparse.parse_tpl(args.input_file)
        
        clear_directory(tex_dir)

        for i, image in enumerate(images):
            print(i, len(image.raw_data))

            writeImage(tex_dir, i, image)


    elif args.compile:
        print(f"Compiling {args.input_file}...")

        if args.input_file == None or not os.path.exists(args.input_file):
            print(f"Folder {args.input_file} not found.")
            sys.exit(1)
        elif not os.path.isdir(args.input_file):
            print(f"Directory {args.input_file} is not a folder.")
            sys.exit(1)

        data = encode.prep(args.input_file, False, args.compress, args.compression_threshold)
        output_path = os.path.join(script_dir, "output.tpl")
        encode.write_tpl(data, output_path)
        print(f"Wrote output.tpl to {output_path}")
        
    else:
        print("Invalid mode selected.")
        sys.exit(1)
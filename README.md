# TPL-Tools
CLI tool for decompiling and compiling TPL files.
- Does not currently support editing image attributes (e.g. lod_bias, min/mag_filter)

Usage:
  pytpl.py [-h] (-d | -c) [--compress | --no-compress] [--compression-threshold INT] input (file or folder)

example: 
    Decompile: `py pytpl.py -d ./a_mario-`
    Compile: `py pytpl.py -c ./output --compress --compression-threshold 10`

Options:
  -h                    Show help

Modes:
  -d                    Decompile TPL → PNGs
  -c                    Compile PNGs → TPL

Optional (compile only):
  --compress
                        Enable CMPR compression (default: False)
  --compression-threshold INT
                        Threshold in KB to force CMPR compression (default: 100)

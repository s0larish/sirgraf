from __future__ import annotations
import argparse
from .core import process_directory
from .visualize import plot_quicklook, animate_filtered

def _parser_plot():
    p = argparse.ArgumentParser(prog="sirgraf-plot", description="Quicklook plots")
    p.add_argument("path", help="Directory containing FITS")
    return p

def _parser_animate():
    p = argparse.ArgumentParser(prog="sirgraf-animate", description="Filtered animation")
    p.add_argument("path", help="Directory containing FITS")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", type=str, default=None, help="Output file (.mp4 or .gif). If omitted, just show.")
    return p

def main_plot():
    args = _parser_plot().parse_args()
    res = process_directory(args.path)
    plot_quicklook(res)

def main_animate():
    args = _parser_animate().parse_args()
    res = process_directory(args.path)
    animate_filtered(res, fps=args.fps, out=args.out)
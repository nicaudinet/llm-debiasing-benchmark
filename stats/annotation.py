from argparse import ArgumentParser
from pathlib import Path
import os

parser = ArgumentParser(description = "Stats about the annotations")
parser.add_argument("annotation_dir", type = Path)
args = parser.parse_args()

print(os.walk(args.annotation_dir))

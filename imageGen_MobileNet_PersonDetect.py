# Alex Hanneman
# TensorFlow MobileNet test utility

import numpy as np
import sys
import os
from argparse import ArgumentParser
from typing import List

from PIL import Image

if __name__ == '__main__':
    parser = ArgumentParser("Generate image binary formatted for mobilenet")
    parser.add_argument(
        "--width", "-w", help="Set output image width", type=int, default=160)
    parser.add_argument(
        "--height", "-h", help="Set output image size", type=int, default=160)
    parser.add_argument(
        "--input", "-i", help="Input file/folder path", type=str, required=True)
    parser.add_argument(
        "--output", "-o", help="Output file/folder path", type=str, required=True)
    parser.add_argument(
        "--alpha", "-a", help="Include alpha channel", action="store_true")

    arguments = parser.parse_args()

    sourceFiles: List[str] = []

    if not os.path.exists(arguments.input):
        raise ValueError("Input path does not exist")
    if not os.path.exists(arguments.output):
        raise ValueError("Output path does not exist")

    if os.path.isdir(arguments.input):
        sourceFiles = os.listdir(arguments.input)
    elif os.path.isfile(arguments.input):
        sourceFiles = [arguments.input]
    else:
        raise ValueError("Input path not directory or file")

    for file in sourceFiles:
        image = Image.open(file).convert('RGB').resize(
            (arguments.width, arguments.height), Image.ANTIALIAS)

        if arguments.alpha:
            image.putalpha(127)

        fileName: str = os.path.join(
            arguments.output, os.path.basename(file)) + ".bin"
        with open(fileName, "wb") as f:
            f.write(image.tobytes())

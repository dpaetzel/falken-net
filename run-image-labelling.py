#!/usr/bin/env nix-shell
#! nix-shell -i python -p "python39.withPackages(ps: with ps; [ click ipython numpy tqdm toolz ])"
#
# Authors: dpaetzel.

import os
import re
import sys
from os.path import basename, exists
import subprocess

import click
from tqdm import tqdm
import toolz
from functools import partial


@click.command()
@click.option("--sample-dir",
              help="Directory containing samples",
              type=str,
              default=".")
@click.option("--unselection-dir",
              help="Directory to store samples in thet were not selected",
              type=str,
              default="unselection")
@click.option("--selection-dir",
              help="Directory to store selected samples in",
              type=str,
              default="selection")
def cli(sample_dir, selection_dir, unselection_dir):
    """
    TODO
    """
    fnames = os.listdir(sample_dir)
    n_fnames = len(fnames)
    print(f"Found {n_fnames} samples.")

    fnames = list(filter(lambda f: "mkv" not in f, fnames))
    print(f"Dropped {n_fnames - len(fnames)} samples "
          f"(due to being from .mkv files), leaving {len(fnames)}.")
    n_fnames = len(fnames)

    # Group by date, hour, minute (xxxx-xx-xx-xx-xx).
    func = toolz.compose("".join, list,
                         partial(toolz.itertoolz.take, 4 + 2 + 2 + 2 + 2 + 4))
    fnames_hours = toolz.groupby(func, fnames)
    images_per_hour = toolz.valmap(len, fnames_hours)

    # Sort each date's images by millisecond (part of the file name).
    regex = re.compile("^....-..-..-..-..-.. Falken.ts-(.*).jpg")

    def key(fname):
        return float(regex.match(fname)[1])

    fnames_hours = toolz.valmap(toolz.compose(list, partial(sorted, key=key)),
                                fnames_hours)

    images_per_view = 90
    fnames_hours_part = toolz.valmap(
        toolz.compose(list, partial(toolz.partition, images_per_view)),
        fnames_hours)

    n_screens = sum(toolz.valmap(len, fnames_hours_part).values())
    print(f"You have {n_screens} image listings to look at "
          f"(each consisting of up to {images_per_view} images).")

    os.makedirs(selection_dir, exist_ok=True)
    os.makedirs(unselection_dir, exist_ok=True)

    for hour in tqdm(fnames_hours_part):
        # Each hour is partitioned into segments of up to 90.
        for fnames_part in tqdm(fnames_hours_part[hour], leave=False):

            fnames_ = [f"{sample_dir}/{fname}" for fname in fnames_part]

            subprocess.run([
                "feh",
                "--thumbnails",
                # “In thumbnail mode, clicking on an image will cause the action to run.
                "--action",
                f"mv '%f' {selection_dir}",
                "--limit-width",
                "1920",
                # Based on a width of 1920, feh recommended to use 1130 as height.
                "--limit-height",
                "1130",
                # Images are 1920x1080.
                "--thumb-height",
                "108",
                "--thumb-width",
                "192",
                # Disable image captions.
                "--index-info",
                "",
            ] + fnames_)

            # This throws a warning/error for any file that was selected.
            subprocess.run([
                "mv",
            ] + fnames_ + [unselection_dir])


if __name__ == "__main__":
    cli()

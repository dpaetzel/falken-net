#!/usr/bin/env nix-shell
#! nix-shell -i python -p "python39.withPackages(ps: with ps; [ click ipython numpy (opencv4.override({ enableFfmpeg = true; enableGtk3 = true; })) tqdm ])"
#
# Authors: dpaetzel, CuiHen.

import os
import re
import sys
from os.path import basename, exists

import click
import cv2
from tqdm import trange


def parse_min_sec(s, argument):
    if ":" in s:
        match =re.match("^(\d+):(\d\d)$", s)
        if match is None:
            print(f"Error in {argument} argument, should be seconds or m:ss format")
            sys.exit(1)
        res = int(match[1]) * 60 + int(match[2])
    else:
        res = int(s)

    return res

@click.command()
@click.option("-r",
              "--sample-rate",
              help="Number of samples per hour",
              default=50,
              type=int)
@click.option("-n",
              "--n-samples",
              help="Number of samples to generate (overrides -r)",
              default=None,
              type=int)
@click.option("-s",
              "--start",
              help=("Start sampling from this position (in seconds or m:s) "
                    "in each video (also requires --end)"),
              type=str,
              default=None)
@click.option("-e",
              "--end",
              help=("Stop sampling from this position (in seconds or m:s) "
                    "in each video (also requires --start)"),
              type=str,
              default=None)
@click.option("--sample-dir",
              help=("Directory to store the samples in "
                    "(will be created if it does not exist)"),
              type=str,
              default="samples")
@click.argument("FILES", nargs=-1)
def cli(sample_rate, n_samples, start, end, files, sample_dir):
    """
    Equidistantly sample images from the given video FILES.
    """

    if n_samples is not None:
        sample_rate = None

    if ((start is None and end is not None)
            or (end is None and start is not None)):
        print("If --start is given, --end needs to be given and vice versa.")

    # TODO Make this nicer: If we have many files, we recompute stuff wrongly
    # because n_samples is not None any more. We fix this by remembering whether
    # n_samples was None initially.
    if n_samples is None:
        n_samples_was_none = True

    for file_ in files:
        if n_samples_was_none:
            n_samples = None

        print(f"Sampling from {file_}")

        vid = cv2.VideoCapture(file_)

        if not vid.isOpened():
            print(f"Video file '{file_}' could not be opened.")
            continue

        if start is None:
            # Note that `end is None` in this case due to option validation
            # above.
            n_frames_tot = vid.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = vid.get(cv2.CAP_PROP_FPS)

            # Video length in seconds.
            video_length = n_frames_tot / fps

        # else corresponds to `start is not None and end is not None` due to
        # option validation above
        else:
            start = parse_min_sec(start, "--start")
            end = parse_min_sec(end, "--end")

            # Video length in seconds.
            video_length = end - start
            assert video_length > 0, "--start and --end must specify a positive interval"

        # If sample rate was given, compute number of samples to be generated.
        if n_samples is None:
            # Sample rate.
            samples_per_second = sample_rate / 60.0 / 60

            # Number of samples in this video.
            n_samples = int(video_length * samples_per_second)

            # In seconds.
            sample_distance = 1 / samples_per_second

        # If number of samples was given, compute sample rate.
        else:
            # Samples per second.
            samples_per_second = n_samples / video_length

            # In seconds.
            sample_distance = 1 / samples_per_second

        print(f"Sampling {n_samples} images (~{samples_per_second:.2f}/s) "
              f"from {video_length:.2f} s video "
              f"with sample distance {sample_distance} s …")

        pos = 1000 * (0 if start is None else start)

        os.makedirs(sample_dir, exist_ok=True)

        for i in trange(n_samples):
            fname = f"{sample_dir}/{basename(file_)}-{pos}.jpg"

            if not exists(fname):
                vid.set(cv2.CAP_PROP_POS_MSEC, pos)

                ret, frame = vid.read()
                if ret == False:
                    print("Aborting due to VideoCapture.read() returning False")
                    break

                # TODO Warn before overwriting stuff
                ret = cv2.imwrite(fname, frame)
                if ret == False:
                    print(f"Failed to write {fname}.")

            # Positions are given in milliseconds, sample_distance is in
            # seconds.
            pos += sample_distance * 1000

        vid.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cli()

#!/usr/bin/env nix-shell
#! nix-shell -i python -p "python39.withPackages(ps: with ps; [ click ipython numpy (opencv4.override({ enableFfmpeg = true; enableGtk3 = true; })) tqdm ])"
#
# Original draft by CuiHen (https://github.com/CuiHen).

from os.path import basename

import click
import cv2
from tqdm import trange


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
@click.option(
    "-s",
    "--start",
    help=("Start sampling from this position (in seconds) in the video (also "
          " requires --end)"),
    type=int,
    default=None)
@click.option("-e",
              "--end",
              help=("Stop sampling at this position (in seconds) in the video "
                    "(also requires --start)"),
              type=int,
              default=None)
@click.argument("FILES", nargs=-1)
def cli(sample_rate, n_samples, start, end, files):
    """
    Equidistantly sample images from the given video FILES.
    """

    if n_samples is not None:
        sample_rate = None

    if ((start is None and end is not None)
            or (end is None and start is not None)):
        print("If --start is given, --end needs to be given and vice versa.")

    for file_ in files:
        print(f"Sampling from {file_}")

        vid = cv2.VideoCapture(file_)

        if not vid.isOpened():
            print(f"Video file '{file_}' could not be opened.")

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
            # Video length in seconds.
            video_length = end - start
            assert video_length > 0, "--start and --end must specify a positive interval"

        # If sample rate was given, compute number of samples to be generated.
        if n_samples is None:
            # Sample rate.
            samples_per_second = sample_rate / 60.0 / 60

            # In milliseconds.
            sample_distance = 1 / samples_per_second

            # Number of samples in this video.
            n_samples = int(video_length * samples_per_second)

        # If number of samples was given, compute sample rate.
        else:
            # Samples per second.
            samples_per_second = n_samples / video_length

            # In seconds.
            sample_distance = 1 / samples_per_second

        print(f"Sampling {n_samples} images ({samples_per_second}/s) "
              f"from {video_length:.2f} s video "
              f"with sample distance {sample_distance} s …")

        pos = 1000 * (0 if start is None else start)
        for i in trange(n_samples):
            vid.set(cv2.CAP_PROP_POS_MSEC, pos)

            ret, frame = vid.read()
            if ret == False:
                print("Aborting due to VideoCapture.read() returning False")
                break

            # TODO Warn before overwriting stuff
            fname = f"sample/{basename(file_)}-{i}.jpg"
            ret = cv2.imwrite(fname, frame)
            if ret == False:
                print("Failed to write {fname}.")

            # Positions are given in milliseconds, sample_distance is in
            # seconds.
            pos += sample_distance * 1000

        vid.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cli()

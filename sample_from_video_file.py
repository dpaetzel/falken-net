#!/usr/bin/env nix-shell
#! nix-shell -i python -p "python39.withPackages(ps: with ps; [ click ipython numpy (opencv4.override({ enableFfmpeg = true; enableGtk3 = true; }))])"
#
# Original draft by CuiHen (https://github.com/CuiHen).

import click
import cv2


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
@click.argument("FILES", nargs=-1)
def cli(sample_rate, n_samples, files):
    """
    Equidistantly sample images from the given video FILES.
    """

    if n_samples is not None:
        sample_rate = None

    for file_ in files:
        print(f"Sampling from {file_}")

        vid = cv2.VideoCapture(file_)

        if not vid.isOpened():
            print(f"Video file '{file_}' could not be opened.")

        n_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vid.get(cv2.CAP_PROP_FPS)

        # Video length in hours.
        video_length = n_frames / fps / 60.0 / 60

        # If sample rate was given, compute number of samples to be generated.
        if n_samples is None:
            # Sample rate.
            samples_per_second = sample_rate / 60.0 / 60

            # In milliseconds.
            sample_distance = 1 / samples_per_second * 1000

            # Number of samples in this video.
            n_samples = int(video_length * sample_rate)

        # If number of samples was given, compute sample rate.
        else:
            # Samples per hour.
            sample_rate = n_samples / video_length
            print(sample_rate)

            # Samples per second.
            samples_per_second = sample_rate / 60.0 / 60
            print(samples_per_second)

            # In seconds.
            sample_distance = 1 / samples_per_second
            print(sample_distance)

        print(f"Sampling {n_samples} images ({sample_rate}/h) "
              f"from {video_length:.2f} h video "
              f"with sample distance {sample_distance} s …")

        pos = 0
        for i in range(n_samples):

            vid.set(cv2.CAP_PROP_POS_MSEC, pos)

            ret, frame = vid.read()
            if ret == False:
                break

            cv2.imwrite(f"sample/{file_}-{i}.jpg", frame)

            # Positions are given in milliseconds, sample_distance is in
            # seconds.
            pos += sample_distance * 1000

        vid.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cli()

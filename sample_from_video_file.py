#!/usr/bin/env nix-shell
#! nix-shell -i python -p "python39.withPackages(ps: with ps; [ click ipython numpy (opencv4.override({ enableFfmpeg = true; enableGtk3 = true; }))])"
#
# Original draft by CuiHen (https://github.com/CuiHen).

import click
import cv2


@click.command()
@click.argument("FILES", nargs=-1)
def cli(files):

    for file_ in files:
        print(f"Sampling from {file_}")

        vid = cv2.VideoCapture(file_)

        if not vid.isOpened():
            print(f"Video file '{file_}' could not be opened.")

        n_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vid.get(cv2.CAP_PROP_FPS)

        # Video length in hours.
        video_length = n_frames / fps / 60 / 60

        # Sample rate.
        samples_per_hour = 100
        samples_per_second = samples_per_hour / 60 / 60

        # In milliseconds.
        sample_distance = 1 / samples_per_second * 1000

        # Number of samples in this video.
        n_samples = int(video_length * samples_per_hour)
        print(f"Sampling {n_samples} images ({samples_per_hour}/h) "
              f"from {video_length} video "
              f"with sample distance {sample_distance} ms …")

        pos = 0
        for i in range(n_samples):

            vid.set(cv2.CAP_PROP_POS_MSEC, pos)

            ret, frame = vid.read()
            if ret == False:
                break

            cv2.imwrite(f"sample/{file_}-{i}.jpg", frame)

            pos += sample_distance

        vid.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cli()

import argparse
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yml")
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()


def extract_velotech_metadata_fps(metadata_file: str, output_file: str, fps: float):
    """
    Take a Velotech metadata file as input, and filter it to match a given FPS.

    Timestamps are used to determine which frames should be kept to stay as close
    as possible to the way in which FFMPEG extracts frames from a video file. In
    the end, this will always be somewhat of an approximation.

    Parameters:
    metadata_file: str
        Original metadata file (assumed to be CSV)
    output_file: str
        Where the filtered metadata will be written as CSV
    fps: float
        The desired FPS
    """
    meta_df = pd.read_csv(metadata_file)

    # Get timestamps and set start to zero
    timestamps = meta_df["timestamp"].to_numpy()
    timestamps = timestamps - timestamps[0]

    # Change time unit to match fps rate
    timestamps *= fps
    # FFMPEG extracts the frames mid-interval. This means frame one will be after the first half interval.
    first = np.where(timestamps >= 0.5)[0][0]
    # Round time to units, discard first half interval, and reset start at zero
    ts_rnd = np.round(timestamps[first:] - timestamps[first])

    # Get indices of frames at whole time units (to match FPS)
    _, indices = np.unique(ts_rnd, return_index=True)
    indices = indices + first
    if indices[-1] >= len(timestamps):
        indices = indices[:-1]

    # Filter metadata and write output
    meta_df = meta_df.iloc[indices, :]
    meta_df["new_frame_id"] = np.arange(1, len(meta_df) + 1)
    meta_df.to_csv(output_file, index=False)


def process_metadata(input_folder: str, output_folder: str):
    """
    Recursively iterate through all metadata CSV files in the input_folder and its subfolders,
    and filter metadata to match given framerate. Results will be written to the output_folder,
    duplicating the names the source files.

    Usage:
    python metadata_helper.py --input_folder <INPUT> --output_folder <OUTPUT>

    Parameters
    ----------
    input_folder: str
        Base folder of input data
    output_folder: str
        Output folder for filtered metadata CSVs
    """
    exclude_dirs = set(settings["frame_extraction"]["exclude_dirs"])
    exclude_files = settings["frame_extraction"]["exclude_files"]
    fps = settings["frame_extraction"]["fps"]

    pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

    # Walk through input_folder recursively
    for dirpath, _, filenames in os.walk(input_folder):
        if dirpath.split(sep="/")[-1] in exclude_dirs:
            # Skip folders in exclude_dirs config
            continue
        for file in filenames:
            if any(excl_file in file for excl_file in exclude_files):
                # Skip files in exclude_files config
                continue
            if file.endswith(".csv"):
                file_path = os.path.join(dirpath, file)
                print(f"Processing {file_path}")
                out_file_path = os.path.join(output_folder, file)
                extract_velotech_metadata_fps(
                    metadata_file=file_path,
                    output_file=out_file_path,
                    fps=fps,
                )
                print(f"Output saved in {out_file_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", dest="input_folder", type=str)
    parser.add_argument("--output_folder", dest="output_folder", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    process_metadata(args.input_folder, args.output_folder)

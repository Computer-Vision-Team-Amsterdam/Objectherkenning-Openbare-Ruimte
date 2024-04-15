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


def extract_velotech_metadata_fps(metadata_file: str, fps: float) -> pd.DataFrame:
    meta_df = pd.read_csv(metadata_file)

    # Get timestamps and set start to zero
    timestamps = meta_df["timestamp"].to_numpy()
    timestamps = timestamps - timestamps[0]

    # Change time unit to match fps
    timestamps *= fps

    # FFMPEG extracts the frames mid interval
    first = np.where(timestamps >= 0.5)[0][0]

    # Round time to units and discard first half frame
    ts_rnd = np.round(timestamps[first:] - timestamps[first])

    # Get indices of unique frame points
    _, indices = np.unique(ts_rnd, return_index=True)
    indices = indices + first
    if indices[-1] >= len(timestamps):
        indices = indices[:-1]

    return meta_df.iloc[indices, :]


def process_metadata(input_folder: str, output_folder: str):
    fe_settings = settings["frame_extraction"]
    exclude_dirs = set(fe_settings["exclude_dirs"])
    exclude_files = fe_settings["exclude_files"]

    pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

    # Walk through input_folder recursively
    for dirpath, _, filenames in os.walk(input_folder):
        if dirpath.split(sep="/")[-1] in exclude_dirs:
            continue
        for file in filenames:
            if any(excl_file in file for excl_file in exclude_files):
                continue
            if file.endswith(".csv"):
                file_path = os.path.join(dirpath, file)
                print(f"Processing {file_path}")
                meta_df = extract_velotech_metadata_fps(
                    metadata_file=file_path, fps=fe_settings["fps"]
                )
                meta_df["new_frame_id"] = np.arange(1, len(meta_df) + 1)
                out_file_path = os.path.join(output_folder, file)
                meta_df.to_csv(out_file_path, index=False)
                print(f"Output saved in {out_file_path}")


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_folder", dest="input_folder", type=str)
    parser.add_argument("--output_folder", dest="output_folder", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


if __name__ == "__main__":
    args = parse_args()
    process_metadata(args.input_folder, args.output_folder)

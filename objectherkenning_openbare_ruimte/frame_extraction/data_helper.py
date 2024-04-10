import numpy as np
import pandas as pd


def extract_velotech_metadata_fps(
    metadata_file: str, output_file: str, fps: float
) -> pd.DataFrame:
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

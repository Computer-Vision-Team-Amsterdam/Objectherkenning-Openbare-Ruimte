import argparse
import os
import pathlib
import shutil
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sg

sys.path.append(os.getcwd())

from objectherkenning_openbare_ruimte.data_sampling import decos_helper  # noqa: E402
from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yml")
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()

RD_CRS = "EPSG:28992"  # CRS code for the Dutch Rijksdriehoek coordinate system
LAT_LON_CRS = "EPSG:4326"  # CRS code for WGS84 latitude/longitude coordinate system


class DecosSampling:
    """
    Class used for sampling image frames based on permit data. A sampling weight
    parameter can be used to determine the amount of non-permit frames allowed.
    """

    def __init__(
        self,
        input_folder: str,
        metadata_folder: str,
        decos_folder: str,
        output_folder: str,
    ):
        """
        Construct a DataSampling object. The construction immediately populates
        a dataframe with all images in the input_folder, and adds metadata and
        permit information.

        After construction, call .run_sampling_job() to perform the data sampling.

        Parameters
        ----------
        input_folder: str
            Path to folder with image frames
        metadata_folder: str
            Path to folder with corresponding metadata
        decos_folder: str
            Path to folder with corresponding permit data
        output_folder: str
            Path to folder where sampled images are stored
        """
        self.metadata_folder = metadata_folder
        self.output_folder = output_folder
        self.n_frames = settings["data_sampling"]["n_frames"]
        self.sampling_weight = settings["data_sampling"]["sampling_weight"]
        self.decos_radius = settings["data_sampling"]["decos_radius"]

        self.decos_gdf = decos_helper.load_and_combine_decos(decos_folder)
        self.frame_gdf = DecosSampling.create_frame_gdf(input_folder)
        self._add_metadata_and_permits()

    @staticmethod
    def load_metadata_gdf(path: str) -> gpd.GeoDataFrame:
        """
        Load a CSV metadata file as GeoDataFrame. Assumes geometry information is
        stored in columns named gps_lon and gps_lat. CRS will be converted to Rijksdriehoek.

        Parameters
        ----------
        path: str
            Path to metadata CSV file

        Returns
        -------
        GeoDataFrame containing the metadata
        """
        meta_df = pd.read_csv(path)
        return gpd.GeoDataFrame(
            meta_df,
            geometry=gpd.points_from_xy(
                x=meta_df.gps_lon,
                y=meta_df.gps_lat,
                crs=LAT_LON_CRS,
            ),
        ).to_crs(RD_CRS)

    @staticmethod
    def create_frame_gdf(
        input_folder: str,
    ) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame to hold metadata information for all image frames
        in a given folder.

        Parameters
        ----------
        input_folder: str
            Path to folder with image frames

        Returns
        -------
        GeoDataFrame to hold the metadata, one row per image frame.
        """
        frames = sorted(list(pathlib.Path(input_folder).glob("*.jpg")))

        frame_data = {
            "path": frames,
            "frame_src": [
                frame.name.replace(".jpg", "").rsplit(sep="_", maxsplit=1)[0]
                for frame in frames
            ],
            "frame_id": [
                int(frame.name.replace(".jpg", "").rsplit(sep="_", maxsplit=1)[1])
                for frame in frames
            ],
            "geometry": np.empty_like(frames, dtype=object),
            "timestamp": np.empty_like(frames, dtype=object),
            "permit": np.zeros_like(frames, dtype=bool),
            "sample": np.zeros_like(frames, dtype=bool),
        }
        frame_gdf = gpd.GeoDataFrame(data=frame_data, crs=RD_CRS)

        print(f"{len(frame_gdf)} frames found")

        return frame_gdf

    def _load_metadata_for_video_name(self, video_name: str) -> gpd.GeoDataFrame:
        """
        Load the CSV metadata file for a given video as GeoDataFrame. Assumes geometry information is
        stored in columns named gps_lon and gps_lat. CRS will be converted to Rijksdriehoek.

        Video name is expected to look like "1-0-D14M03Y2024-H12M56S12",
        and the corresponding metadata file would be "1-D14M03Y2024-H12M56S12.csv"

        Parameters
        ----------
        video_name: str
            Name of the video to which the metadata belongs

        Returns
        -------
        GeoDataFrame containing the metadata
        """
        # NOTE: this has to be changed when naming conventions change
        meta_key_split = video_name.split(sep="-", maxsplit=2)
        meta_file = f"{meta_key_split[0]}-{meta_key_split[2]}.csv"
        meta_path = os.path.join(self.metadata_folder, meta_file)
        return DecosSampling.load_metadata_gdf(meta_path)

    def _get_metadata_for_gdf(
        self, frame_gdf: gpd.GeoDataFrame, video_name: str
    ) -> np.ndarray:
        """
        Load the metadata corresponding to the frames in frame_gdf taken from the
        video with name video_name.

        In case of a mismatch in number of rows, metadata is clipped or appended
        with None to match the number of frames. This difference can arise due
        to inaccurate extraction of metadata rows based on FPS rate; typically
        there might be a one-frame difference.

        Parameters
        ----------
        frame_gdf: GeoDataFrame
            The frames for which metadata is needed
        video_name: str
            The name of the source video for the frames

        Returns
        -------
        A numpy array with the columns [timestamp, geometry] for each frame.
        """
        # NOTE: we now use the system timestamp, using gps_time instead would prevent dependency on accurate system time
        meta_gdf = self._load_metadata_for_video_name(video_name).set_index(
            "new_frame_id"
        )[["timestamp", "geometry"]]
        meta_gdf["timestamp"] = pd.to_datetime(meta_gdf["timestamp"], unit="s")
        meta_data = meta_gdf.to_numpy()

        # In case of a difference in number of rows, we clip or append None
        diff = len(frame_gdf) - len(meta_data)
        if diff > 0:
            meta_data = np.vstack([meta_data, [[None, None]] * diff])
        elif diff < 0:
            meta_data = meta_data[0 : len(frame_gdf), :]

        return meta_data

    def _get_permits_for_gdf(self, frame_gdf) -> np.ndarray:
        """
        Get permits for the frames in frame_gdf by matching the date and location
        to the available permit data.

        A boolean numpy array is returned with one entry for each frame. The entry is
        True when the frame location falls within the decos_radius area of a permit.

        Parameters
        ----------
        frame_gdf: GeoDataFrame
            The frames for which permit data is needed

        Returns
        -------
        The permit data as numpy array.
        """
        # Get a bounding box or the frame_gdf
        decos_bb = sg.box(*frame_gdf.total_bounds)
        # Get the timestamp and filter the decos data
        # NOTE: this assumes all frames are from the same date
        date = frame_gdf["timestamp"][0]
        decos_gdf_filtered = decos_helper.filter_decos_by_date(self.decos_gdf, date)
        decos_clipped = decos_gdf_filtered.clip(decos_bb.buffer(self.decos_radius))

        # Check if any permits are available for this area
        if len(decos_clipped) >= 1:
            decos_area = decos_clipped.unary_union.buffer(self.decos_radius)
            permits = frame_gdf.within(decos_area).to_numpy()
        else:
            permits = np.zeros((len(frame_gdf), 1), dtype=bool)

        return permits

    def _add_metadata_and_permits(self):
        """
        Add metadata and permit information for each image frame in frame_gdf.

        The frame_gdf is expected to contain one line for each frame, and a column
        "frame_src" indicating the video the frame originated from. This will be
        used as key to locate the right metadata file.

        The metadata file is expected to contain geometry (gps_lat, gps_lon)
        and timestamps for each frame.

        The permit data will be queried based on the time stamp and location.
        """
        for src_name in self.frame_gdf["frame_src"].unique():
            # src_name is the name of the source video file
            print(f"Adding metadata and permits for {src_name}")
            idx = np.where(self.frame_gdf["frame_src"] == src_name)[0]

            meta_data = self._get_metadata_for_gdf(self.frame_gdf.loc[idx, :], src_name)
            self.frame_gdf.loc[idx, ["timestamp", "geometry"]] = meta_data

            permits = self._get_permits_for_gdf(self.frame_gdf.loc[idx, :])
            self.frame_gdf.loc[idx, "permit"] = permits

            print(
                f"{np.count_nonzero(self.frame_gdf.loc[idx, 'permit'])}/{len(self.frame_gdf.loc[idx, 'permit'])} frames in permit zone"
            )

    def _sample_frames(self):
        """
        Sample frames based on sampling weight and permit data. The weight in [0,1]
        determines how much emphasis is placed on sampling frames inside a permit
        zone as opposed to those outside.

        Fewer frames may be sampled than requested in case the number of frames with
        non-zero sampling weight is less than the requested number. For example,
        when there are only 30 frames with a permit and sampling weight is set to 1,
        no more than those 30 frames will be sampled.
        """
        weights = np.array(
            [
                self.sampling_weight if permit else 1 - self.sampling_weight
                for permit in self.frame_gdf["permit"]
            ]
        )
        weights = weights / np.sum(weights)

        if self.n_frames >= np.count_nonzero(weights):
            draw = np.where(weights > 0)[0]
            print(
                f"Cannot draw {self.n_frames} samples out of {np.count_nonzero(weights)} nonzero weights, sampling all nonzero."
            )
        else:
            draw = np.random.choice(
                np.arange(len(self.frame_gdf)), self.n_frames, p=weights, replace=False
            )
        sample = np.zeros_like(self.frame_gdf["permit"], dtype=bool)
        sample[draw] = True
        self.frame_gdf["sample"] = sample

    def _copy_sample_to_output_folder(self):
        """Copy the sampled frames to the output_folder."""
        print(f"Copying sample to {self.output_folder}")
        pathlib.Path(self.output_folder).mkdir(exist_ok=True, parents=True)

        for i, row in self.frame_gdf.iterrows():
            if row["sample"]:
                frame = row["path"]
                target_path = os.path.join(self.output_folder, frame.name)
                shutil.copyfile(frame.as_posix(), target_path)

    def _store_metadata(self):
        """
        Store metadata (frame_gdf) as GPKG file in the same output folder as the sampled frames.
        """
        metadata_file = os.path.join(self.output_folder, "metadata.gpkg")
        metadata_gdf = self.frame_gdf.assign(
            path=self.frame_gdf["path"].apply(lambda p: p.as_posix())
        )
        metadata_gdf["timestamp"] = metadata_gdf["timestamp"].apply(
            lambda t: (None if t is None else t.timestamp())
        )
        print(f"Storing metadata in {metadata_file}")
        metadata_gdf.to_file(metadata_file, driver="GPKG")

    def run_sampling_job(self):
        """
        Run the sampling process. This method first samples N frames based on available
        permit data and sampling weight, then copies those sampled frames to the
        output_folder, and finally saves the metadata used for sampling in that
        folder as well.
        """
        self._sample_frames()

        # Print sample statistics for debugging purposes
        n_sampled = np.count_nonzero(self.frame_gdf["sample"])
        n_with_permit = np.count_nonzero(
            self.frame_gdf["sample"] & self.frame_gdf["permit"]
        )
        ratio = (n_with_permit / n_sampled) * 100
        print(
            f"Sampled {n_sampled} frames, of which {n_with_permit} in permit zone ({ratio}%)"
        )

        # self._copy_sample_to_output_folder()
        self._store_metadata()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_folder", dest="input_folder", type=str)
    parser.add_argument("--metadata_folder", dest="metadata_folder", type=str)
    parser.add_argument("--decos_folder", dest="decos_folder", type=str)
    parser.add_argument("--output_folder", dest="output_folder", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    sampler = DecosSampling(
        args.input_folder, args.metadata_folder, args.decos_folder, args.output_folder
    )
    sampler.run_sampling_job()

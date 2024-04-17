import argparse
import os
import pathlib
import shutil
import sys
from datetime import datetime

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

RD_CRS = "EPSG:28992"
LAT_LON_CRS = "EPSG:4326"


class DataSampling:

    @staticmethod
    def load_metadata_gdf(path: str) -> gpd.GeoDataFrame:
        """
        Load a CSV metadata file as GeoDataFrame. Assumes geometry information is
        stored in columns named gps_lon and gps_lat. CRS will be converted to Rijksdriehoek.

        Parameters
        ----------
        path: str
            Path to metadata SCV file

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
    def add_metadata_and_permits(
        frame_gdf: gpd.GeoDataFrame,
        metadata_folder: str,
        decos_gdf: gpd.GeoDataFrame,
        decos_buffer: float,
    ) -> gpd.GeoDataFrame:
        """
        Add metadata and permit information for each image frame in frame_gdf.

        The frame_gdf is expected to contain one line for each frame, and a column
        "frame_src" indicating the video the frame originated from. This will be
        used as key to locate the right metadata file.

        The permit data TODO

        Parameters
        ----------
        frame_gdf: GeoDataFrame
            TODO

        Returns
        -------
        GeoDataFrame containing the metadata
        """
        for src_name in frame_gdf["frame_src"].unique():
            print(f"Adding metadata for {src_name}")
            idx = np.where(frame_gdf["frame_src"] == src_name)[0]
            meta_key_split = src_name.split(sep="-", maxsplit=2)
            meta_file = f"{meta_key_split[0]}-{meta_key_split[2]}.csv"
            meta_path = os.path.join(metadata_folder, meta_file)

            meta_gdf = DataSampling.load_metadata_gdf(meta_path).set_index(
                "new_frame_id"
            )["geometry"]

            geoms = meta_gdf.to_list()
            if len(idx) > len(geoms):
                geoms.extend([None] * (len(idx) - len(geoms)))
            elif len(geoms) > len(idx):
                geoms = geoms[0 : len(idx)]

            frame_gdf.loc[idx, "geometry"] = geoms

            decos_bb = sg.box(*frame_gdf.loc[idx, :].total_bounds)
            date = datetime.strptime(meta_file.split(sep="-")[1], "D%dM%mY%Y")
            decos_gdf_filtered = decos_helper.filter_decos_by_date(decos_gdf, date)
            decos_clipped = decos_gdf_filtered.clip(decos_bb.buffer(decos_buffer))
            if len(decos_clipped) >= 1:
                decos_area = decos_clipped.unary_union.buffer(decos_buffer)
                permits = frame_gdf.loc[idx, :].within(decos_area).to_numpy()
                frame_gdf.loc[idx, "permit"] = permits

            print(f"{np.count_nonzero(permits)}/{len(permits)} frames in permit zone")

        return frame_gdf

    @staticmethod
    def load_frame_gdf(
        input_folder: str,
        metadata_folder: str,
        decos_gdf: gpd.GeoDataFrame,
        decos_buffer: float,
    ) -> gpd.GeoDataFrame:
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
            "permit": np.zeros_like(frames, dtype=bool),
            "sample": np.zeros_like(frames, dtype=bool),
        }

        frame_gdf = gpd.GeoDataFrame(data=frame_data, crs=RD_CRS)

        print(f"{len(frame_gdf)} frames found")

        return DataSampling.add_metadata_and_permits(
            frame_gdf, metadata_folder, decos_gdf, decos_buffer
        )

    def __init__(
        self,
        input_folder: str,
        metadata_folder: str,
        decos_folder: str,
        output_folder: str,
    ):
        self.output_folder = output_folder
        self.n_frames = settings["data_sampling"]["n_frames"]
        self.sampling_weight = settings["data_sampling"]["sampling_weight"]

        decos_buffer = settings["data_sampling"]["decos_buffer"]
        decos_gdf = decos_helper.load_and_combine_decos(decos_folder)
        self.frame_gdf = DataSampling.load_frame_gdf(
            input_folder, metadata_folder, decos_gdf, decos_buffer
        )

    def sample_frames(self):
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

    def copy_sample_to_output_folder(self):
        pathlib.Path(self.output_folder).mkdir(exist_ok=True, parents=True)

        print(f"Copying sample to {self.output_folder}")

        for i, row in self.frame_gdf.iterrows():
            if row["sample"]:
                frame = row["path"]
                target_path = os.path.join(self.output_folder, frame.name)
                shutil.copyfile(frame.as_posix(), target_path)

    def store_metadata(self):
        metadata_file = os.path.join(self.output_folder, "metadata.gpkg")
        metadata_gdf = self.frame_gdf.assign(
            path=self.frame_gdf["path"].apply(lambda p: p.as_posix())
        )
        print(f"Storing metadata in {metadata_file}")
        metadata_gdf.to_file(metadata_file, driver="GPKG")

    def run_sampling_job(self):

        self.sample_frames()

        n_sampled = np.count_nonzero(self.frame_gdf["sample"])
        n_with_permit = np.count_nonzero(
            self.frame_gdf["sample"] & self.frame_gdf["permit"]
        )
        ratio = (n_with_permit / n_sampled) * 100
        print(
            f"Sampled {n_sampled} frames, of which {n_with_permit} in permit zone ({ratio}%)"
        )

        self.copy_sample_to_output_folder()
        self.store_metadata()


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_folder", dest="input_folder", type=str)
    parser.add_argument("--metadata_folder", dest="metadata_folder", type=str)
    parser.add_argument("--decos_folder", dest="decos_folder", type=str)
    parser.add_argument("--output_folder", dest="output_folder", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


if __name__ == "__main__":
    args = parse_args()
    sampler = DataSampling(
        args.input_folder, args.metadata_folder, args.decos_folder, args.output_folder
    )
    sampler.run_sampling_job()

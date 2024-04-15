import datetime
import pathlib

import geopandas as gpd
import pandas as pd

RD_CRS = "EPSG:28992"


def load_decos_gdf(path: str) -> gpd.GeoDataFrame:
    decos_df = pd.read_csv(path)
    return gpd.GeoDataFrame(
        decos_df,
        geometry=gpd.GeoSeries.from_wkb(decos_df.geometrie_locatie),
        crs=RD_CRS,
    )


def load_and_combine_decos(decos_data_folder: str) -> gpd.GeoDataFrame:
    decos_files = pathlib.Path(decos_data_folder).glob("*.csv")

    decos_df = pd.concat(
        [pd.read_csv(decos_file) for decos_file in decos_files], ignore_index=True
    )
    decos_gdf = gpd.GeoDataFrame(
        decos_df,
        geometry=gpd.GeoSeries.from_wkb(decos_df.geometrie_locatie),
        crs="EPSG:28992",
    )
    del decos_df

    decos_gdf["datum_object_van"] = pd.to_datetime(
        decos_gdf["datum_object_van"], format="%Y-%m-%d"
    )
    decos_gdf["datum_object_tm"] = pd.to_datetime(
        decos_gdf["datum_object_tm"], format="%Y-%m-%d"
    )

    return decos_gdf


def filter_decos_by_date(
    decos_gdf: gpd.GeoDataFrame, date: datetime.datetime
) -> gpd.GeoDataFrame:
    filtered_gdf = decos_gdf[
        (decos_gdf["datum_object_van"] <= date) & (decos_gdf["datum_object_tm"] >= date)
    ]
    return filtered_gdf

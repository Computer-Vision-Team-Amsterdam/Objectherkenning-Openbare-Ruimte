import os
import pathlib
import sys
from datetime import datetime, time

import geopandas as gpd
import pandas as pd

sys.path.append(os.getcwd())

from objectherkenning_openbare_ruimte.data_sampling import RD_CRS  # noqa: E402


def _decos_df_to_gdf(decos_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert Decos DataFrame to GeoDataFrame. Geometry will be parsed from WKT string. The columns
    datum_object_van and datum_object_tm will be converted to datetime object for later use.

    Parameters
    ----------
    decos_df: DataFrame
        The DataFrame to convert

    Returns
    -------
    GeoDataFrame with Decos data
    """
    decos_gdf = gpd.GeoDataFrame(
        decos_df,
        geometry=gpd.GeoSeries.from_wkb(decos_df["geometrie_locatie"]),
        crs=RD_CRS,
    )
    decos_gdf["datum_object_van"] = pd.to_datetime(
        decos_gdf["datum_object_van"], format="%Y-%m-%d"
    )
    decos_gdf["datum_object_tm"] = pd.to_datetime(
        decos_gdf["datum_object_tm"], format="%Y-%m-%d"
    )
    return decos_gdf


def load_and_combine_decos(decos_data_folder: str) -> gpd.GeoDataFrame:
    """
    Load Decos data from multiple files as GeoDataFrame. Geometry will be parsed from WKT string.
    The columns datum_object_van and datum_object_tm will be converted to datetime object for later use.

    Parameters
    ----------
    decos_data_folder: str
        Folder containing the Decos data files

    Returns
    -------
    GeoDataFrame with Decos data
    """
    decos_files = pathlib.Path(decos_data_folder).glob("*.csv")
    decos_df = pd.concat(
        [pd.read_csv(decos_file) for decos_file in decos_files], ignore_index=True
    )
    return _decos_df_to_gdf(decos_df)


def filter_decos_by_date(
    decos_gdf: gpd.GeoDataFrame, date: datetime
) -> gpd.GeoDataFrame:
    """
    Filter Decos data by a given date. All entries for which the date falls between
    datum_object_van and datum_object_tm will be returned.

    Parameters
    ----------
    decos_gdf: GeoDataFrame
        Decos data
    date: datetime
        Target date to filter by

    Returns
    -------
    GeoDataFrame with Decos entries valid on date
    """
    # Get rid of hours/minutes/etc
    date = datetime.combine(date.date(), time.min)

    filtered_gdf = decos_gdf[
        (decos_gdf["datum_object_van"] <= date) & (decos_gdf["datum_object_tm"] >= date)
    ]
    return filtered_gdf

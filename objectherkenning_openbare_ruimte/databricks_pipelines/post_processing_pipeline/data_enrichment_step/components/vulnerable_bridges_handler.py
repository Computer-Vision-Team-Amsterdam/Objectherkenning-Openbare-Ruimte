import os
import tempfile

import geopandas as gpd
import pandas as pd
from databricks.sdk.runtime import dbutils
from pyspark.sql import SparkSession
from shapely.geometry import Point
from shapely.ops import nearest_points
from shapely.wkt import dumps as wkt_dumps

RD_EPSG = 28992
WGS84_EPSG = 4326


class VulnerableBridgesHandler:
    def __init__(
        self, spark_session: SparkSession, root_source, vuln_bridges_relative_path
    ):
        self.spark_session = spark_session
        self.file_path = os.path.join(root_source, vuln_bridges_relative_path)
        self._load_bridges_gdf()

    def _load_bridges_gdf(self):
        print(f"Loading vulnerable bridges file: {self.file_path}")

        # Copy the file to local storage, otherwise GeoPandas cannot read it
        _, tmp_path = tempfile.mkstemp()
        try:
            dbutils.fs.cp(self.file_path, f"file://{tmp_path}")
            self.bridges_gdf = gpd.read_file(tmp_path)
        finally:
            os.remove(tmp_path)

        print(f"Loaded {len(self.bridges_gdf)} bridge and quay wall objects")

        if self.bridges_gdf.crs.to_epsg() == WGS84_EPSG:
            self.bridges_gdf.to_crs(epsg=RD_EPSG, inplace=True)
        if "objectnummer" not in self.bridges_gdf.columns:
            if "rakcode" in self.bridges_gdf.columns:
                self.bridges_gdf.loc[:, "objectnummer"] = self.bridges_gdf["rakcode"]
            else:
                print(
                    "WARNING: No `objectcode` or `rakcode` found in bridges file. Using index instead."
                )
                self.bridges_gdf.loc[:, "objectnummer"] = self.bridges_gdf.index

    def calculate_distances_to_closest_vulnerable_bridges(
        self,
        objects_coordinates_df,
    ):
        objects_df = objects_coordinates_df.toPandas()

        objects_gdf = gpd.GeoDataFrame(
            index=objects_df.detection_id,
            geometry=[
                Point(row.gps_lon, row.gps_lat) for _, row in objects_df.iterrows()
            ],
            crs=WGS84_EPSG,
        ).to_crs(RD_EPSG)

        results_gdf = gpd.sjoin_nearest(
            left_df=objects_gdf,
            right_df=self.bridges_gdf,
            how="left",
            distance_col="distance",
        )

        results_df = pd.DataFrame(
            data={
                "detection_id": results_gdf.index,
                "closest_bridge_distance": results_gdf["distance"],
                "closest_bridge_id": results_gdf["objectnummer"],
                "closest_bridge_coordinates": [
                    list(reversed(*nearest_points(location, geom)[1].coords))
                    for location, geom in zip(
                        objects_gdf.to_crs(WGS84_EPSG).geometry,
                        self.bridges_gdf.to_crs(WGS84_EPSG).loc[
                            results_gdf["index_right"], :
                        ]["geometry"],
                    )
                ],
                "closest_bridge_geom_wkt": [
                    wkt_dumps(geom)
                    for geom in self.bridges_gdf.to_crs(WGS84_EPSG).loc[
                        results_gdf["index_right"], :
                    ]["geometry"]
                ],
            }
        )

        return self.spark_session.createDataFrame(results_df)

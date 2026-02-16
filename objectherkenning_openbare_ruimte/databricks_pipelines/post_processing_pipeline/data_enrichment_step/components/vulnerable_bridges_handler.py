import os

import geopandas as gpd
import pandas as pd
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
        self.filepath = os.path.join(root_source, vuln_bridges_relative_path)
        self._load_bridges_gdf()

    def _load_bridges_gdf(self):
        self.bridges_gdf = gpd.read_file(self.filepath)
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
                    list(*nearest_points(location, geom)[1].coords)
                    for location, geom in zip(
                        objects_gdf.geometry,
                        self.bridges_gdf.loc[results_gdf["index_right"], :]["geometry"],
                    )
                ],
                "closest_bridge_geom_wkt": [
                    wkt_dumps(geom)
                    for geom in self.bridges_gdf.loc[results_gdf["index_right"], :][
                        "geometry"
                    ]
                ],
            }
        )

        return self.spark_session.createDataFrame(results_df)

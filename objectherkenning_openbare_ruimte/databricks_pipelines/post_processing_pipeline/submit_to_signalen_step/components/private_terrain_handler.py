from typing import List, Tuple

from pyproj import Transformer
from pyspark.sql import Row, SparkSession
from shapely import wkb
from shapely.geometry import Point
import pandas as pd

from objectherkenning_openbare_ruimte.databricks_pipelines.common.reference_db_connector import (  # noqa: E402
    ReferenceDatabaseConnector,
)


class PrivateTerrainHandler(ReferenceDatabaseConnector):

    def __init__(
        self,
        spark: SparkSession,
        az_tenant_id: str,
        db_host: str,
        db_name: str,
        db_port: int,
    ) -> None:
        super().__init__(az_tenant_id, db_host, db_name, db_port)
        self.spark = spark
        self.detection_buffer_distance = 20
        self.detection_crs = "EPSG:4326"
        self.terrain_crs = "EPSG:28992"
        self.transformer = Transformer.from_crs(
            self.detection_crs, self.terrain_crs, always_xy=True
        )
        self.query_and_process_public_terrains()

    def query_and_process_public_terrains(self):
        """
        Query the database for public terrain geometries and process the results.
        """
        query = "SELECT geometrie FROM beheerkaart_basis_kaart WHERE agg_indicatie_belast_recht = FALSE"
        print("Querying public terrain data from the database...")
        result_df = self.run(query, "public terrain polygons")
        if result_df.empty:
            print("No public terrain data found.")
            self._public_terrains = {}
            return

        result_df["polygon"] = result_df["geometrie"].apply(
            lambda x: self.convert_wkb(x)
        )
        result_df = result_df[result_df["polygon"].notnull()]
        self._public_terrains = result_df.to_dict(orient="records")

    def convert_wkb(self, hex_str):
        try:
            return wkb.loads(bytes.fromhex(hex_str))
        except Exception as e:
            print(f"Error converting geometry {hex_str}: {e}")
            return None
        
    def create_dataframe(self, rows, colnames):
        """
        Create a DataFrame .
        """
        data = [dict(zip(colnames, row)) for row in rows]
        df = pd.DataFrame(data, columns=colnames)
        return df

    def check_private_terrain(self, detection_row: Row) -> Tuple[bool, List[str]]:
        """
        Check whether a single detection is on private terrain.
        A detection is on private terrain if its buffer does not intersect with any public terrain polygons.
        Returns:
        is_on_private (bool): True if the detection is likely on private property.
        overlapping_polygons (list): List of overlapping polygons as WKT strings.
        """
        object_lon = detection_row.object_lon
        object_lat = detection_row.object_lat
        x, y = self.transformer.transform(object_lon, object_lat)
        pt = Point(x, y)
        buffered = pt.buffer(self.detection_buffer_distance)

        # If there are no overlapping public polygons, assume the detection is on private property.
        is_on_private_terrain = not any(buffered.intersects(rec["polygon"]) for rec in self._public_terrains)
        return is_on_private_terrain

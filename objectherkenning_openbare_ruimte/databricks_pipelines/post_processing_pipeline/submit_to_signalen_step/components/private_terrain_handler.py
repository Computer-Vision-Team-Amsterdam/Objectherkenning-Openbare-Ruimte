from typing import Any, List, Optional

import pandas as pd
from pyproj import Transformer
from pyspark.sql import Row, SparkSession
from shapely import wkb
from shapely.geometry import Point

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

    def query_and_process_public_terrains(self) -> None:
        """
        Query the database for public terrain geometries and process the results.

        The function runs a query to fetch geometries for public terrain and converts them
        from WKB hex strings to Shapely geometry objects. Processed geometries are stored
        in the _public_terrains attribute.
        """
        query = "SELECT geometrie FROM beheerkaart_basis_kaart WHERE agg_indicatie_belast_recht = FALSE"
        print("Querying public terrain data from the database...")
        result_df = self.run(query)
        if result_df.empty:
            print("No public terrain data found.")
            self._public_terrains = {}
            return

        result_df["polygon"] = result_df["geometrie"].apply(
            lambda x: self.convert_wkb(x)
        )
        result_df = result_df[result_df["polygon"].notnull()]
        self._public_terrains = result_df.to_dict(orient="records")

    def convert_wkb(self, hex_str: str) -> Optional[Any]:
        """
        Convert a WKB hex string to a Shapely geometry object.

        Parameters:
            hex_str: A hexadecimal string representing the WKB geometry.

        Returns:
            A Shapely geometry object if the conversion is successful; None otherwise.
        """
        try:
            return wkb.loads(bytes.fromhex(hex_str))
        except Exception as e:
            print(f"Error converting geometry {hex_str}: {e}")
            return None

    def create_dataframe(self, rows: List[Any], colnames: List[str]) -> pd.DataFrame:
        """
        Create a pandas DataFrame from provided row data and column names.

        Parameters:
            rows: A list of row data, where each row is an iterable of values.
            colnames: A list of column names corresponding to the row data.

        Returns:
            A pandas DataFrame constructed from the provided data.
        """
        data = [dict(zip(colnames, row)) for row in rows]
        df = pd.DataFrame(data, columns=colnames)
        return df

    def on_private_terrain(self, detection_row: Row) -> bool:
        """
        Check whether a detection is on private terrain.

        A detection is considered to be on private terrain if its buffered point does not
        intersect with any public terrain polygon.

        Parameters:
            detection_row: A Row object containing detection data including longitude and latitude.

        Returns:
            True if the detection is on private terrain; False otherwise.
        """
        object_lon = detection_row.object_lon
        object_lat = detection_row.object_lat
        x, y = self.transformer.transform(object_lon, object_lat)
        pt = Point(x, y)
        buffered = pt.buffer(self.detection_buffer_distance)

        # If there are no overlapping public polygons, assume the detection is on private terrain.
        is_on_private_terrain = not any(
            buffered.intersects(rec["polygon"]) for rec in self._public_terrains
        )
        return is_on_private_terrain

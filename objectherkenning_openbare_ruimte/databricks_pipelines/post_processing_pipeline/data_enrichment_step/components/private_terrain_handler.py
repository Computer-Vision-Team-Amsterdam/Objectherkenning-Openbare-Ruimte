from typing import Any, List, Optional

import pandas as pd
from pyproj import Transformer
from pyspark.sql import DataFrame, Row, SparkSession
from shapely import Point, STRtree, wkb

from objectherkenning_openbare_ruimte.databricks_pipelines.common.reference_db_connector import (  # noqa: E402
    ReferenceDatabaseConnector,
)


class PrivateTerrainHandler(ReferenceDatabaseConnector):

    detection_crs = "EPSG:4326"
    terrain_crs = "EPSG:28992"

    def __init__(
        self,
        spark_session: SparkSession,
        az_tenant_id: str,
        db_host: str,
        db_name: str,
        db_port: int,
        detection_buffer: float = 10,
    ) -> None:
        super().__init__(az_tenant_id, db_host, db_name, db_port)
        self.spark_session = spark_session
        self.detection_buffer = detection_buffer

        self.transformer = Transformer.from_crs(
            self.detection_crs, self.terrain_crs, always_xy=True
        )
        self.spatial_tree: Optional[STRtree] = None
        self._query_and_process_public_terrains()

    def _query_and_process_public_terrains(self):
        """
        Query the database for public terrain geometries.

        The function runs a query to fetch geometries for public terrain and converts them
        from WKB hex strings to Shapely geometry objects. Processed geometries are stored
        in the public_terrains attribute.
        """
        query = "SELECT geometrie FROM beheerkaart_basis_kaart WHERE agg_indicatie_belast_recht = FALSE"
        print("Querying public terrain data from the database...")
        result_df = self.run(query)
        if result_df.empty:
            print("No public terrain data found.")
        else:
            result_df["polygon"] = result_df["geometrie"].apply(
                lambda x: self.convert_wkb(x)
            )
            result_df = result_df[result_df["polygon"].notnull()]
            self.spatial_tree = STRtree(result_df["polygon"].to_list())

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

    def lookup_private_terrain_for_detections(
        self, objects_coordinates_df: DataFrame, id_column: str = "detection_id"
    ) -> DataFrame:
        """
        Look up for each detection location whether it is on private terrain or
        not. Returns a DataFrame with the column `private_terrain` for each row
        in objects_coordinates_df.
        """
        results = []

        if not self.spatial_tree:
            print(
                "No private terrain data available, assuming all detections are on public terrain."
            )

        for row in objects_coordinates_df.collect():
            on_private_terrain = False
            if self.spatial_tree:
                object_point = Point(
                    self.transformer.transform(row.object_lon, row.object_lat)
                ).buffer(self.detection_buffer)

                # If there are no overlapping public polygons, assume the detection is on private terrain.
                result = self.spatial_tree.query(object_point, predicate="intersects")
                on_private_terrain = len(result) == 0
            results.append(
                Row(
                    detection_id=row[id_column],  # retain the detection id for joining
                    private_terrain=on_private_terrain,
                )
            )
        results_df = self.spark_session.createDataFrame(results)

        return results_df

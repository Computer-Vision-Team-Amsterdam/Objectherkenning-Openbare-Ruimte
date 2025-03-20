from typing import List, Tuple

from pyproj import Transformer
from pyspark.sql import Row, SparkSession
from shapely import wkb
from shapely.geometry import Point

from .reference_db_connector import ReferenceDatabaseConnector


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
        self.private_buffer_distance = 20
        self.private_simplify_tolerance = 1.0
        self.private_detection_crs = "EPSG:4326"
        self.private_polygon_crs = "EPSG:28992"
        self.private_transformer = Transformer.from_crs(
            self.private_detection_crs, self.private_polygon_crs, always_xy=True
        )
        self.inverse_private_transformer = Transformer.from_crs(
            self.private_polygon_crs, self.private_detection_crs, always_xy=True
        )
        self.query_and_process_private_terrains()

    def query_and_process_private_terrains(self):
        """
        Query the database for private terrain features and process the results.
        Assumes a table "beheerkaart_basis_kaart" with a column "geometrie" containing WKB as hex.
        """
        query = "SELECT geometrie FROM beheerkaart_basis_kaart WHERE agg_indicatie_belast_recht = FALSE"
        print("Querying non-private terrain data from the database...")
        result_df = self.run(query, "non-private terrain polygons")
        if result_df.empty:
            print("No non-private terrain data found.")
            self._private_terrain_df = result_df
            self._private_terrains = []
            return

        def _convert_wkb(hex_str):
            try:
                return wkb.loads(bytes.fromhex(hex_str))
            except Exception as e:
                print(f"Error converting geometry {hex_str}: {e}")
                return None

        result_df["polygon"] = result_df["geometrie"].apply(lambda x: _convert_wkb(x))
        result_df = result_df[result_df["polygon"].notnull()]
        self._private_terrains = result_df.to_dict()
        print(f"Loaded {len(self._private_terrains)} non-private terrain features.")

    def check_private_terrain(self, detection_row: Row) -> Tuple[bool, List[str]]:
        """
        Check whether a single detection is on private terrain.
        A detection is on private terrain if its buffer does not intersect with any non-private polygon.
        Returns:
        is_on_private (bool): True if the detection is likely on private property.
        overlapping_polygons (list): List of overlapping polygons as WKT strings.
        """
        object_lat = detection_row.gps_lat
        object_lon = detection_row.gps_lon

        # Transform the coordinate from EPSG:4326 to EPSG:28992.
        x, y = self.private_transformer.transform(object_lon, object_lat)
        pt = Point(x, y)
        buffered = pt.buffer(self.private_buffer_distance)

        # Determine overlapping polygons.
        overlapping_polygons = [
            rec["polygon"].wkt
            for rec in self._private_terrains
            if buffered.intersects(rec["polygon"])
        ]

        # If there are no overlapping public polygons, assume the detection is on private property.
        is_on_private = len(overlapping_polygons) == 0
        return is_on_private, overlapping_polygons

from typing import Optional

import pandas as pd
from pyproj import Transformer
from pyspark.sql import Row
from shapely.geometry import Point


class PrivateTerrainHandler:

    def __init__(
        self,
        public_terrains: Optional[pd.DataFrame],
    ) -> None:
        self.detection_buffer_distance = 20
        self.detection_crs = "EPSG:4326"
        self.terrain_crs = "EPSG:28992"
        self.transformer = Transformer.from_crs(
            self.detection_crs, self.terrain_crs, always_xy=True
        )
        self.public_terrains = public_terrains

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
            buffered.intersects(rec["polygon"]) for rec in self.public_terrains
        )
        return is_on_private_terrain

from pyproj import Transformer
from pyspark.sql import Row
from shapely.geometry import Point
from typing import Optional, Tuple


class StadsdelenHandler:

    def __init__(
        self,
        stadsdelen: dict,
    ) -> None:
        self.detection_crs = "EPSG:4326"
        self.stadsdelen_crs = "EPSG:28992"
        self.transformer = Transformer.from_crs(
            self.detection_crs, self.stadsdelen_crs, always_xy=True
        )
        self.stadsdelen = stadsdelen

    def get_stadsdeel_name_and_code(self, detection_row: Row) -> Optional[Tuple[str, str]]:
        """
        Get the stadsdeel name and code for the given detection.

        Parameters:
            detection_row: A Row object containing detection data including longitude and latitude.

        Returns:
            A tuple ("name", "code") if the point is inside a stadsdeel, or None otherwise
        """
        object_lon = detection_row.object_lon
        object_lat = detection_row.object_lat
        x, y = self.transformer.transform(object_lon, object_lat)
        pt = Point(x, y)

        for stadsdeel in self.stadsdelen:
            if stadsdeel["polygon"].contains(pt)
                return (stadsdeel["name"], stadsdeel["code"])

        return None

import json
from typing import Optional

import folium
import pyspark.sql.functions as F
import requests
import shapely.ops
from pyproj import Transformer
from pyspark.sql import Column
from pyspark.sql.functions import UserDefinedFunction, udf
from pyspark.sql.types import BooleanType
from shapely.geometry import Point, mapping, shape


class PrivateTerrainHandler:
    def __init__(
        self,
        buffer_distance: float = 20,
        simplify_tolerance: float = 1.0,
        api_url: Optional[str] = None,
        polygon_crs: str = "EPSG:28992",
        detection_crs: str = "EPSG:4326",
    ) -> None:
        """
        Initialize the private terrain handler.

        :param buffer_distance: Buffer distance (in meters) around locations detected objects.
        :param simplify_tolerance: Tolerance in meters for simplifying the buffered object polygon.
        :param api_url: Base URL to fetch private property polygons.
            Uses default API if not provided.
        :param polygon_crs: CRS of the polygon data (default is EPSG:28992).
        :param detection_crs: CRS of the detection points (default is EPSG:4326).
        """
        if api_url is None:
            api_url = "https://api.data.amsterdam.nl/v1/beheerkaart/basis/kaart/"
        self.api_url = api_url
        self.buffer_distance = buffer_distance
        self.simplify_tolerance = simplify_tolerance
        self.polygon_crs = polygon_crs
        self.detection_crs = detection_crs
        self.transformer = Transformer.from_crs(
            self.detection_crs, self.polygon_crs, always_xy=True
        )

    def is_private_terrain(self, lat: float, lon: float) -> str:
        """
        Checks whether a detection point (after transformation and buffering)
        falls on private terrain by querying the API with the spatial filter.

        :param lat: Latitude of the detection (in detection_crs).
        :param lon: Longitude of the detection (in detection_crs).
        :return: True if the API returns one or more features intersecting the buffered geometry.
        """
        try:
            x, y = self.transformer.transform(lon, lat)
            object_location = Point(x, y)
            object_location_with_buffer = object_location.buffer(self.buffer_distance)
            object_location_with_buffer = object_location_with_buffer.simplify(
                self.simplify_tolerance
            )

            params = {
                "aggIndicatieBelastRecht": "true",
                "_format": "geojson",
                "geometrie[intersects]": object_location_with_buffer.wkt,
            }
            response = requests.get(self.api_url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            features_overlapping_areas = data.get("features", {})
            is_private = len(features_overlapping_areas) > 0

            # Return both the flag and the features as a JSON string.
            return json.dumps(
                {
                    "is_private_terrain": is_private,
                    "features": features_overlapping_areas,
                }
            )

        except Exception as e:
            print("Error in private_terrain_details API query:", e)
            return json.dumps({"is_private": False, "features": []})

    def get_udf(self) -> UserDefinedFunction:
        """
        Wraps the is_private_terrain function as a Spark UDF.
        (Caution: Calling external APIs in a UDF may be very slow.)
        """
        return udf(self.is_private_terrain, BooleanType())

    def get_private_terrain_expr(
        self, lat_column: str = "object_lat", lon_column: str = "object_lon"
    ) -> Column:
        """
        Returns a Spark Column expression that applies the UDF to the specified latitude and longitude columns.

        :param lat_column: Name of the column with the latitude.
        :param lon_column: Name of the column with the longitude.
        :return: A pyspark.sql.Column that evaluates to a boolean.
        """
        return self.get_udf()(
            F.col(lat_column).cast("double"), F.col(lon_column).cast("double")
        )


if __name__ == "__main__":
    # For testing locally:
    test_lat = 52.374306836988026
    test_lon = 4.917097851669691

    checker = PrivateTerrainHandler()
    details_json = checker.is_private_terrain(test_lat, test_lon)
    details = json.loads(details_json)
    print(f"Coordinate ({test_lat}, {test_lon}) details: {details}")

    # Create a folium map for visualization (optional)
    map_center = [test_lat, test_lon]
    m = folium.Map(location=map_center, zoom_start=15)

    # If any overlapping features were found, add them to the map.
    for feature in details.get("features", []):
        try:
            geom_data = feature.get("geometry") or feature.get("geometrie")
            if not geom_data:
                continue
            poly = shape(geom_data)

            inverse_transformer = Transformer.from_crs(
                checker.polygon_crs, "EPSG:4326", always_xy=True
            )

            def transform_coords(x, y, z=None):
                lon_, lat_ = inverse_transformer.transform(x, y)
                return (lon_, lat_) if z is None else (lon_, lat_, z)

            transformed_poly = shapely.ops.transform(transform_coords, poly)
            geo_json_data = mapping(transformed_poly)
            folium.GeoJson(
                geo_json_data,
                style_function=lambda f: {
                    "fillColor": "red",
                    "color": "red",
                    "fillOpacity": 0.2,
                    "weight": 1,
                },
            ).add_to(m)
        except Exception as e:
            print("Error transforming polygon for folium:", e)

    marker_color = "blue" if details.get("is_private_terrain") else "green"
    folium.Marker(
        location=map_center,
        popup=f"Test Coordinate (Private: {details.get('is_private_terrain')})",
        icon=folium.Icon(color=marker_color),
    ).add_to(m)

    output_html = "private_terrain_map.html"
    m.save(output_html)
    print(f"Map saved to {output_html}")

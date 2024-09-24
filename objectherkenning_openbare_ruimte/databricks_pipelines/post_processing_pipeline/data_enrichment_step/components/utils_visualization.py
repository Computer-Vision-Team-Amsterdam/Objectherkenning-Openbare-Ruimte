import os

import folium
from databricks.sdk.runtime import *  # noqa: F403, F401
from folium.plugins import BeautifyIcon
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window
from shapely.geometry import Point
from shapely.ops import nearest_points
from shapely.wkt import loads as wkt_loads


def generate_map(
    dataframe,
    trajectory=None,
    name=None,
    path=None,
    colors=None,
) -> None:
    """
    This method generates an HTML page with a map containing a path line and randomly chosen points on the line
    corresponding to detected containers on the path.

    :param vulnerable_bridges: list of line string coordinates.
    :param permit_locations: list of point coordinates.
    :param trajectory: list of coordinates that define the path.
    :param detections: model predictions dict (with information about r names and coordinates).
    :param name: custom name for the map. If not passed, name is created based on what the map contains.
    :param colors: colors to be assigned to each cluster
    """

    os.makedirs(path, exist_ok=True)

    # Amsterdam coordinates
    latitude = 52.377956
    longitude = 4.897070

    # create empty map zoomed on Amsterdam
    Map = folium.Map(location=[latitude, longitude], zoom_start=12)

    vulnerable_bridges_group = folium.FeatureGroup(name="Vulnerable bridges").add_to(
        Map
    )
    closest_bridges_group = folium.FeatureGroup(
        name="Distances to closest bridge"
    ).add_to(Map)
    closest_permit_group = folium.FeatureGroup(
        name="Distances to closest permit"
    ).add_to(Map)

    # Add priority_id column for visualization
    # Define the window specification
    window_spec = Window.orderBy(col("score").desc())

    # Add the "priority_id" column
    dataframe_with_priority = dataframe.withColumn(
        "priority_id", row_number().over(window_spec)
    )

    # display(dataframe_with_priority)

    # Function to find the closest point on a linestring to a given point
    def closest_point_on_linestring(point, linestring):
        return nearest_points(point, linestring)[1]

    # Function to determine the marker color based on the score
    def get_marker_color(score):
        if score < 0.40:
            return "green"
        elif 0.4 <= score < 1:
            return "yellow"
        elif score == 1:
            return "orange"
        else:
            return "red"

    # Prepare data for visualization
    # Iterate over each row in the DataFrame
    for row in dataframe_with_priority.toLocalIterator():
        # Extract data from the row
        print(f"score is {row['score']}")
        detection = Point(row["gps_lat"], row["gps_lon"])
        detection_id = row["detection_id"]
        detection_image_name = row["image_name"]
        detection_priority_id = row["priority_id"]
        detection_score = row["score"] if row["score"] else 0
        vulnerable_bridge = wkt_loads(row["closest_bridge_linestring_wkt"])
        closest_bridge_id = row["closest_bridge_id"]
        permit_location = Point(row["closest_permit_lat"], row["closest_permit_lon"])
        closest_permit_id = row["closest_permit_id"]

        # Determine marker color based on the score
        marker_color = get_marker_color(detection_score)

        # Create a custom DivIcon for the marker with the priority_id
        detection_icon = BeautifyIcon(
            icon="arrow-down",
            icon_shape="marker",
            number=str(detection_priority_id),
            border_color="#000000",
            background_color=marker_color,
            text_color="#000000",
        )

        '''# Path for the image
        image_path = f'Volumes/dpcv_dev/default/landingzone/test-diana/images/{formatted_detection_date}/{detection_image_name}'
        image_path = f'abfss://landingzone@stlandingdpcvontweu01.dfs.core.windows.net/test-diana/images/{formatted_detection_date}/{detection_image_name}'
        test_image_path = f'abfss://landingzone@stlandingdpcvontweu01.dfs.core.windows.net/test-diana/images/{formatted_detection_date}/'
        print(f'Testing path system works: {dbutils.fs.ls(test_image_path)}')
        print(f'Image path: {image_path}')
        print('Testing image path:')
        display(spark.read.format("image").load(image_path))
        #popup = f'Detection ID: {detection_id}<br>Image: <img src="{image_path}"><br>'
        popup = """<div><img src=""" + image_path + """></div>"""
        print(f'Popup html: {popup})')'''

        # Add container locations to the map
        folium.Marker(
            location=[detection.x, detection.y],
            color=marker_color,
            popup=f"Detection ID: {detection_id}<br>"
            f"Image Name: {detection_image_name}<br>",
            # popup=popup,
            radius=5,
            icon=detection_icon,
        ).add_to(Map)

        # Add closest vulnerable bridge
        coordinates = [(point[0], point[1]) for point in vulnerable_bridge.coords]
        folium.PolyLine(
            coordinates,
            color="yellow",
            weight=5,
            opacity=0.8,
            tooltip=f"Bridge ID: {closest_bridge_id}",
        ).add_to(vulnerable_bridges_group)

        # Add closest container permit
        folium.CircleMarker(
            location=[permit_location.x, permit_location.y],
            color="red",
            radius=5,
            weight=2,
            tooltip=f"Permit ID: {closest_permit_id}",
        ).add_to(Map)

        # Add distances between container and closest vulnerable bridge
        point_on_bridge = closest_point_on_linestring(detection, vulnerable_bridge)
        # distance = detection.distance(point_on_bridge)
        polyline_coords = [
            (detection.x, detection.y),
            (point_on_bridge.x, point_on_bridge.y),
        ]
        folium.PolyLine(polyline_coords, color="blue", weight=5, opacity=0.8).add_to(
            closest_bridges_group
        )

        # Add distances between container and closest permit
        # distance = detection.distance(permit_location)
        polyline_coords = [
            (detection.x, detection.y),
            (permit_location.x, permit_location.y),
        ]
        folium.PolyLine(polyline_coords, color="green", weight=5, opacity=0.8).add_to(
            closest_permit_group
        )

    folium.LayerControl().add_to(Map)

    # create name for the map
    print(f"Map is saved at {name}")
    full_path = path + name + ".html"
    print(f"Saving at {full_path}")
    Map.save(full_path)

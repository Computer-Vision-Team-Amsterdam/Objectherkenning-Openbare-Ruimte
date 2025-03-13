import os

import base64
import folium
from databricks.sdk.runtime import *  # noqa: F403, F401
from folium.plugins import BeautifyIcon
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window
from shapely.geometry import Point
from shapely.ops import nearest_points
from shapely.wkt import loads as wkt_loads

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_job_process_time,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.bronze.frames import (  # noqa: E402
    BronzeFrameMetadataManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils import (  # noqa: E402
    unix_to_yyyy_mm_dd,
)


def generate_map(
    dataframe,
    trajectory=None,
    name=None,
    path=None,
    colors=None,
    catalog=None,
    device_id=None,
    job_process_time=None,
) -> None:
    """
    This method generates an HTML page with a map containing a path line and randomly chosen points on the line
    corresponding to detected objects on the path.

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

    icon_map = {2: "box", 3: "toilet-portable", 4: "table-cells"}

    # Get image folder path
    job_date = str(job_process_time).split(" ")[0]
    stlanding_image_folder = unix_to_yyyy_mm_dd(
        BronzeFrameMetadataManager.get_gps_internal_timestamp_of_current_run(
            job_date=job_date
        )
    )
    image_folder = f"/Volumes/{catalog}/default/landingzone/{device_id}/images/{stlanding_image_folder}"

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
        detection = Point(row["gps_lat"], row["gps_lon"])
        detection_id = row["detection_id"]
        detection_image_name = row["image_name"]
        detection_priority_id = row["priority_id"]
        detection_score = row["score"] if row["score"] else 0
        vulnerable_bridge = wkt_loads(row["closest_bridge_linestring_wkt"])
        closest_bridge_id = row["closest_bridge_id"]
        permit_location = Point(row["closest_permit_lat"], row["closest_permit_lon"])
        closest_permit_id = row["closest_permit_id"]

        # Create a custom DivIcon for the marker with the priority_id
        marker_color = get_marker_color(detection_score)
        icon_type = icon_map.get(row["object_class"], "info-sign")
        detection_icon = BeautifyIcon(
            icon=icon_type,
            icon_shape="marker",
            border_color=marker_color,
            background_color="white",
            text_color="#000000",
        )

        # Encode image for map visualization
        image_path = os.path.join(image_folder, detection_image_name)
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{encoded_image}"

        popup_html = f"""
        <div style="width:400px;">
            <strong>Detection ID:</strong> {detection_id}<br>
            <strong>Priority ID:</strong> {detection_priority_id}<br>
            <img src="{data_uri}" alt="Detection image" style="max-width:100%; height:auto;">
        </div>
        """

        # Add object locations to the map
        folium.Marker(
            location=[detection.x, detection.y],
            popup=popup_html,
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

        # Add closest object permit
        folium.CircleMarker(
            location=[permit_location.x, permit_location.y],
            color="red",
            radius=5,
            weight=2,
            tooltip=f"Permit ID: {closest_permit_id}",
        ).add_to(Map)

        # Add distances between object and closest vulnerable bridge
        point_on_bridge = closest_point_on_linestring(detection, vulnerable_bridge)
        # distance = detection.distance(point_on_bridge)
        polyline_coords = [
            (detection.x, detection.y),
            (point_on_bridge.x, point_on_bridge.y),
        ]
        folium.PolyLine(polyline_coords, color="blue", weight=5, opacity=0.8).add_to(
            closest_bridges_group
        )

        # Add distances between object and closest permit
        # distance = detection.distance(permit_location)
        polyline_coords = [
            (detection.x, detection.y),
            (permit_location.x, permit_location.y),
        ]
        folium.PolyLine(polyline_coords, color="green", weight=5, opacity=0.8).add_to(
            closest_permit_group
        )

    folium.LayerControl().add_to(Map)

    object_class_legend = """
    <div style="
        position: fixed;
        bottom: 10px;
        left: 10px;
        width: 220px;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        background-color:white;
        padding: 10px;
    ">
    <b>Legend</b><br>
    <i class="fa fa-box" style="font-size:20px; color: black;"></i>&nbsp; Container / Bouwkeet<br>
    <i class="fa fa-toilet-portable" style="font-size:20px; color: black;"></i>&nbsp; Mobiel Toilet<br>
    <i class="fa fa-table-cells" style="font-size:20px; color: black;"></i>&nbsp; Steiger
    </div>
    """
    Map.get_root().html.add_child(folium.Element(object_class_legend))

    # create name for the map
    print(f"Map is saved at {name}")
    full_path = path + name + ".html"
    print(f"Saving at {full_path}")
    Map.save(full_path)

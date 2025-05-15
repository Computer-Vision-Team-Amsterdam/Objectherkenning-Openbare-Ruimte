import base64
import os

import cv2
import folium
from databricks.sdk.runtime import *  # noqa: F403, F401
from folium.plugins import BeautifyIcon
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window
from shapely.geometry import Point
from shapely.ops import nearest_points
from shapely.wkt import loads as wkt_loads

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.frames import (
    SilverFrameMetadataManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils import (
    unix_to_yyyy_mm_dd,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.data_enrichment_step.components.utils_images import (  # noqa: E402
    OutputImage,
)


def generate_map(
    dataframe,
    annotate_detection_images,
    name,
    path,
    catalog,
    device_id,
) -> None:
    """
    Generates an interactive HTML map visualizing object detections, their closest vulnerable bridges,
    and closest permit locations, along with contextual image popups and visual cues.

    The map includes:
    - Markers for detected objects with priority icons and popups showing detection ID, permit ID, and image.
    - Lines indicating distance from detections to their nearest vulnerable bridge and permit location.
    - A legend for detected object classes (e.g., container, mobile toilet, scaffold).
    - Layer toggles for bridges, permits, and distance lines.

    :param dataframe: A Spark DataFrame containing detection data, GPS coordinates, associated image names,
                      closest bridge and permit information, and detection scores.
    :param name: Optional name for the saved HTML map. If provided, it is used as the file name.
    :param path: Directory path where the map HTML file will be saved.
    :param catalog: Catalog name used to construct the image path (Databricks volume).
    :param device_id: Device ID used in the image path.
    :param job_process_time: Timestamp used to resolve the image folder structure based on the job run.
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
        x_center_norm, y_center_norm = row["x_center"], row["y_center"]
        width_norm, height_norm = row["width"], row["height"]
        object_class = row["object_class"]

        # Create a custom DivIcon for the marker with the priority_id
        marker_color = get_marker_color(detection_score)
        icon_type = icon_map.get(object_class, "info-sign")
        detection_icon = BeautifyIcon(
            icon=icon_type,
            icon_shape="marker",
            border_color=marker_color,
            background_color="white",
            text_color="#000000",
        )

        # Get image folder path
        stlanding_image_folder = unix_to_yyyy_mm_dd(
            SilverFrameMetadataManager.get_gps_timestamp_from_image_name(
                detection_image_name
            )
        )
        image_folder = f"/Volumes/{catalog}/default/landingzone/{device_id}/images/{stlanding_image_folder}"
        image_path = os.path.join(image_folder, detection_image_name)

        # Add bounding boxes to image and save
        if annotate_detection_images:
            image = cv2.imread(image_path)
            annotated_image = OutputImage(image)
            annotated_image.draw_bounding_box(
                x_center_norm, y_center_norm, width_norm, height_norm
            )
            base, ext = os.path.splitext(image_path)
            image_path = f"{base}_annotated_{object_class}{ext}"
            cv2.imwrite(image_path, annotated_image.get_image())

        # Add image to map
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{encoded_image}"

        popup_html = f"""
        <div style="width:400px;">
            <strong>Detection ID:</strong> {detection_id}<br>
            <strong>Priority ID:</strong> {detection_priority_id}<br>
            <strong>Closest Permit ID:</strong> {closest_permit_id}<br>
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

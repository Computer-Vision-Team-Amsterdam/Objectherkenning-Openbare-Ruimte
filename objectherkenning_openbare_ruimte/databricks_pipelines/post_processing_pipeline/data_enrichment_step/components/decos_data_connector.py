import json
import re
from typing import Dict, List, Optional

import geopy.distance
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.functions import col
from shapely import wkb
from shapely.geometry import Point

from objectherkenning_openbare_ruimte.databricks_pipelines.common.reference_db_connector import (  # noqa: E402
    ReferenceDatabaseConnector,
)


class BENKAGGConnector(ReferenceDatabaseConnector):
    bankagg_table_name = "benkagg_adresseerbareobjecten_v1"

    def __init__(self, az_tenant_id, db_host, db_name) -> None:
        super().__init__(az_tenant_id, db_host, db_name)

    def get_benkagg_adresseerbareobjecten_by_address(
        self, street, house_number, postcode
    ):
        query = f"select openbareruimte_naam, huisnummer, huisletter, postcode, adresseerbaar_object_punt_geometrie from {self.bankagg_table_name} where openbareruimte_naam = '{street}' and huisnummer = '{house_number}' and postcode = '{postcode}'"  # nosec B608
        return self.run(query)

    def get_benkagg_adresseerbareobjecten_by_id(self, id):
        query = f"select openbareruimte_naam, huisnummer, huisletter, postcode, adresseerbaar_object_punt_geometrie from {self.bankagg_table_name} where identificatie='{id}'"  # nosec B608
        return self.run(query)

    # TODO create the dataframe with spark!
    def create_dataframe(self, rows, colnames):
        """
        Create dataframe from query result.
        """

        def _load_columns_with_unsupported_data_type():
            """
            Pandas is complaining about loading the types of the following columns so we load them as strings.
            """
            data = [dict(zip(colnames, row)) for row in rows]

            for record in data:
                for time_col in [
                    "tijd_tvm_parkeervakken_van",
                    "tijd_tvm_parkeervakken_tot",
                    "tijd_tvm_stremmen_van",
                    "tijd_tvm_stremmen_tot",
                ]:
                    if time_col in record:
                        record[time_col] = str(record[time_col])

            return data

        data = _load_columns_with_unsupported_data_type()
        df = pd.DataFrame(data, columns=colnames)

        return df


class DecosDataHandler(BENKAGGConnector):
    vergunningen_table_name = "vergunningen_werk_en_vervoer_op_straat_v2"

    def __init__(
        self,
        spark_session: SparkSession,
        az_tenant_id: str,
        db_host: str,
        db_name: str,
        object_classes: Dict[int, str],
        permit_mapping: Dict[int, List],
    ) -> None:
        super().__init__(az_tenant_id, db_host, db_name)
        self.spark_session = spark_session
        self.object_classes = object_classes
        self.permit_mapping = permit_mapping

    def query_and_process_object_permits(self, date_to_query):
        """
        Process the results of the query.
        """
        query = f"SELECT id, kenmerk, locatie, geometrie_locatie, objecten FROM {self.vergunningen_table_name} WHERE datum_object_van <= '{date_to_query}' AND datum_object_tm >= '{date_to_query}'"  # nosec B608
        print(f"Querying the permit database for date {date_to_query}...")
        result_df = self.run(query)

        def _safe_json_load(x):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                return []

        result_df["objecten"] = result_df["objecten"].apply(
            lambda x: _safe_json_load(x) if x else []
        )

        # Assign object category to each permit
        result_df["object_classes"] = result_df["objecten"].apply(
            self.get_object_classes
        )

        # Keep only permits that match at least one of the keywords (i.e. have a valid category)
        result_df = result_df[result_df["object_classes"].apply(lambda x: len(x) > 0)]

        # Only keep rows where location of the permit is not null
        result_df = result_df[result_df["locatie"].notnull()]

        # Add new columns for lat lon coordinates of permits
        result_df = self.add_permit_coordinates_columns(result_df)

        # Store rows where permit_lat and permit_lon are non null as healthy data
        self._healthy_df = result_df[
            result_df["permit_lat"].notnull() & result_df["permit_lon"].notnull()
        ]

        # Reset the healthy DataFrame index (to ensure alignment with internal lists)
        self._healthy_df = self._healthy_df.reset_index(drop=True)

        print(
            f"{len(self._healthy_df)} permits for active object categories with valid coordinates."
        )
        if len(self._healthy_df) == 0:
            raise ValueError("No permit coordinates could be converted from addresses.")

        # Store rows where permit_lat and permit_lon are null as quarantine data
        self._quarantine_df = result_df[
            result_df["permit_lat"].isnull() | result_df["permit_lon"].isnull()
        ]
        print(
            f"{len(self._quarantine_df)} permits for active object categories with invalid coordinates."
        )

        exploded = self._healthy_df.explode("object_classes")
        stats = exploded["object_classes"].value_counts()
        print("Matching permits per category:")
        for category, count in stats.items():
            print(f"  Category {self.object_classes[category]}: {count}")

        self._permits_coordinates = self._extract_permits_coordinates()
        self._permits_coordinates_geometry = self._convert_coordinates_to_point()

        # self.write_healthy_df_to_table()
        # self.write_qurantine_df_to_table()

    def display_dataframe(self, df):
        """
        Display the DataFrame.
        """
        df.display()

    def split_dutch_street_address(self, raw_address: str) -> List[str]:
        """
        This function separates an address string (street name, house number, house number extension and zipcode)
        into parts.
        Regular expression quantifiers:
        X?  X, once or not at all
        X*  X, zero or more times
        X+  X, one or more times
        """
        regex = (
            r"^(.+?)\s+(\d+)(?:\s+([A-Za-z0-9-]+))?\s+(\d{4}\s*[A-Za-z]{2})(?:\s+.*)?$"
        )
        return re.findall(regex, raw_address)

    def convert_EWKB_geometry_to_coordinates(self, ewkb_geometry):
        # Convert the hex string to binary and load it as a Shapely geometry.
        # (Note: Shapely will ignore the embedded SRID so you must know it.)
        geometry = wkb.loads(bytes.fromhex(ewkb_geometry))

        # Extract the X and Y coordinates (these are in EPSG:28992, Dutch RD New)
        if geometry.geom_type == "Point":
            x, y = geometry.x, geometry.y
        elif geometry.geom_type in ["Polygon", "MultiPolygon"]:
            rep_point = geometry.centroid
            x, y = rep_point.x, rep_point.y
        else:
            raise ValueError(f"Unsupported geometry type: {geometry.geom_type}")

        # Create a transformer from EPSG:28992 to EPSG:4326 (lat/lon)
        transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)

        return lat, lon

    def convert_address_to_coordinates(self, address):
        """
        Convert address to coordinates using benkagg_adresseerbareobjecten table.
        """
        try:
            split_dutch_address = self.split_dutch_street_address(address)
            if split_dutch_address:
                result_df = self.get_benkagg_adresseerbareobjecten_by_address(
                    street=split_dutch_address[0][0],
                    house_number=split_dutch_address[0][1],
                    postcode=split_dutch_address[0][3],
                )
                if result_df.empty:
                    print(
                        f"Warning: No results found for Dutch street address: {address}"
                    )
                    return None
                if result_df.shape[0] > 1:
                    print(
                        f"Warning: Multiple results found for Dutch street address: {address}"
                    )
                coordinates = self.convert_EWKB_geometry_to_coordinates(
                    result_df["adresseerbaar_object_punt_geometrie"].iloc[0]
                )
                print(f"Coordinates: {coordinates}")
                latitude = coordinates[0]
                longitude = coordinates[1]
                return [latitude, longitude]
            else:
                print(
                    f"Warning: Unable to split Dutch street address using regex: {address}"
                )
        except Exception as e:
            print(f"Error converting address into coordinates for {address}: {e}")
            return None

    def add_permit_coordinates_columns(self, df):
        """
        Add "permit_lat" and "permit_lon" to the DataFrame.
        If 'geometrie_locatie' is present (not null), use convert_EWKB_geometry_to_coordinates;
        otherwise, fall back to convert_address_to_coordinates on the 'locatie' field.
        """
        coords = [
            (
                self.convert_EWKB_geometry_to_coordinates(geom)
                if pd.notnull(geom)
                else self.convert_address_to_coordinates(addr)
            )
            for geom, addr in zip(df["geometrie_locatie"], df["locatie"])
        ]

        df["permit_lat"] = [coord[0] if coord is not None else None for coord in coords]
        df["permit_lon"] = [coord[1] if coord is not None else None for coord in coords]

        return df

    def _extract_permits_coordinates(self):
        # -----> for spark dataframes
        # # Collect the DataFrame rows as a list of Row objects
        # rows = self._healthy_df.select("permit_lat", "permit_lon").collect()

        # # Convert the list of Row objects into a list of tuples
        # permits_coordinates = [(row['permit_lat'], row['permit_lon']) for row in rows]

        # ------> for pandas dataframes

        permits_coordinates = list(
            self._healthy_df[["permit_lat", "permit_lon"]].itertuples(
                index=False, name=None
            )
        )

        return permits_coordinates

    def get_permits_ids(self):
        # -----> for spark dataframes
        # # Collect the DataFrame rows as a list of Row objects
        # rows = self._healthy_df.select("id").collect()

        # # Convert the list of Row objects into a list of tuples
        # permits_ids = [(row['id']) for row in rows]

        # ------> for pandas dataframes
        permits_ids = self._healthy_df["id"].tolist()

        return permits_ids

    def _convert_coordinates_to_point(self):
        """
        We need the permits coordinates as Point to perform distance calculations
        """
        permits_coordinates_geometry = [
            Point(location) for location in self._permits_coordinates
        ]
        return permits_coordinates_geometry

    def get_healthy_df(self):
        return self._healthy_df

    def get_quarantine_df(self):
        return self._quarantine_df

    def get_object_classes(self, objects: List[Dict[str, str]]) -> List[int]:
        """
        Determine matching object classes for permits from a list
        of permit keywords based on a keyword mapping.

        Parameters:
            objects (List[Dict[str, str]]): A list of dictionaries, each representing an object. Each dictionary
                                            should contain an "object" key with a string value.

        Returns:
            List[str]: A list of object classes that match the permit based on the keyword mapping.
        """
        mapping = self.get_keyword_mapping()
        matched = []
        for category, keywords in mapping.items():
            regex_pattern = re.compile(r"(?i)(" + "|".join(keywords) + r")")
            if any(regex_pattern.search(obj.get("object", "")) for obj in objects):
                matched.append(category)
        return matched

    def get_keyword_mapping(self) -> Dict[int, List[str]]:
        """
        Return a mapping of object class keys to their permit mapping values.

        Returns:
            Dict[int, str]: Dictionary with object class keys and corresponding permit mapping values.
        """
        active_keys = set(self.object_classes.keys())
        return {k: self.permit_mapping[k] for k in active_keys}

    def calculate_distances_to_closest_permits(
        self, objects_coordinates_df: DataFrame
    ) -> DataFrame:
        """
        Calculate distances to the closest permits for each object class and union the results.

        Parameters:
            objects_coordinates_df (DataFrame): A Spark DataFrame containing object coordinates.

        Returns:
            DataFrame: A Spark DataFrame with calculated distances or an empty DataFrame if no results.
        """
        dfs_closest_permits: List[DataFrame] = []

        # Loop through each target object category.
        for object_class in set(self.object_classes.keys()):
            df_object_class = objects_coordinates_df.filter(
                col("object_class") == object_class
            )
            if not df_object_class.take(1):
                continue
            # Calculate the distances for the current object_class using the corresponding keyword list.
            df_closest_permit = (
                self.calculate_distances_to_closest_permits_for_object_class(
                    df_object_class, object_class
                )
            )
            if df_closest_permit:
                dfs_closest_permits.append(df_closest_permit)

        if dfs_closest_permits:
            union_dfs_closest_permits = dfs_closest_permits[0]
            for df in dfs_closest_permits[1:]:
                union_dfs_closest_permits = union_dfs_closest_permits.union(df)
            return union_dfs_closest_permits
        else:
            # Return an empty Spark DataFrame with the same schema as objects_coordinates_df.
            empty_rdd = self.spark_session.sparkContext.emptyRDD()
            return self.spark_session.createDataFrame(
                empty_rdd, schema=objects_coordinates_df.schema
            )

    def calculate_distances_to_closest_permits_for_object_class(
        self, objects_df: DataFrame, category: int
    ) -> Optional[DataFrame]:
        """
        Calculate the closest permit distances for a given object category.

        Parameters:
            objects_df (DataFrame): Spark DataFrame with object data including 'gps_lat', 'gps_lon', and 'detection_id'.
            category (int): The target object category used to filter healthy permits.

        Returns:
            Optional[DataFrame]: A Spark DataFrame with columns 'detection_id', 'closest_permit_distance',
                                'closest_permit_id', 'closest_permit_lat', and 'closest_permit_lon'.
                                Returns None if no healthy permits match the category.
        """
        # Filter healthy permits whose object_classes (a list) contains the target category.
        mask = self._healthy_df["object_classes"].apply(lambda cats: category in cats)
        filtered_indices = self._healthy_df.index[mask].tolist()

        if not filtered_indices:
            return None

        filtered_points = [
            self._permits_coordinates_geometry[i] for i in filtered_indices
        ]
        filtered_ids = self._healthy_df.loc[filtered_indices, "id"].tolist()
        filtered_coords = [self._permits_coordinates[i] for i in filtered_indices]

        results = []
        for row in objects_df.collect():
            object_lat = row.gps_lat
            object_lon = row.gps_lon

            object_location = Point(object_lat, object_lon)

            closest_permit_distances = []
            for permit_location in filtered_points:
                try:
                    # calculate distance between object point and permit point
                    permit_dist = geopy.distance.distance(
                        object_location.coords, permit_location.coords
                    ).meters
                except Exception:
                    permit_dist = 0
                    print("Error occurred:")
                    print(
                        f"Container location: {object_location}, {object_location.coords}"
                    )
                    print(
                        f"Permit location: {permit_location}, {permit_location.coords}"
                    )
                closest_permit_distances.append(permit_dist)
            min_distance_idx = np.argmin(closest_permit_distances)
            results.append(
                Row(
                    detection_id=row.detection_id,  # retain the detection id for joining
                    closest_permit_distance=float(
                        closest_permit_distances[min_distance_idx]
                    ),
                    closest_permit_id=filtered_ids[min_distance_idx],
                    closest_permit_lat=filtered_coords[min_distance_idx][0],
                    closest_permit_lon=filtered_coords[min_distance_idx][1],
                )
            )
        results_df = self.spark_session.createDataFrame(results)

        return results_df

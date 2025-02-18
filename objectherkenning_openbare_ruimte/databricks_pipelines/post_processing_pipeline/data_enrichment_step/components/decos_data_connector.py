import json
import re
from typing import List, Tuple

import geopy.distance
import numpy as np
import pandas as pd
import requests
from pyspark.sql import Row
from pyspark.sql.functions import col
from shapely.geometry import Point

from .reference_db_connector import ReferenceDatabaseConnector


class DecosDataHandler(ReferenceDatabaseConnector):

    def __init__(self, spark, az_tenant_id, db_host, db_name, db_port):
        super().__init__(az_tenant_id, db_host, db_name, db_port)
        self.spark = spark
        self.query_result_df = None

    def process_query_result(self):
        """
        Process the results of the query.
        """

        def _safe_json_load(x):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                return []
            
        def get_object_category(objects):
            mapping = self.get_keyword_mapping()
            for category, keywords in mapping.items():
                regex_pattern = re.compile(r"(?i)(" + "|".join(keywords) + r")")
                for obj in objects:
                    if regex_pattern.search(obj.get("object", "")):
                        return category
            return None

        print(
            "Processing object permits (filter by object name, convert address to coordinates)..."
        )
        self.query_result_df["objecten"] = self.query_result_df["objecten"].apply(
            lambda x: _safe_json_load(x) if x else []
        )

        # Assign object category to each permit
        self.query_result_df["object_category"] = self.query_result_df["objecten"].apply(get_object_category)

        # Keep only permits that match at least one of the keywords (i.e. have a valid category)
        self.query_result_df = self.query_result_df[self.query_result_df["object_category"].notnull()]
        
        # Only keep rows where location of the permit is not null
        self.query_result_df = self.query_result_df[
            self.query_result_df["locatie"].notnull()
        ]

        # Get list of permit addresses in BAG format
        addresses = self.query_result_df["locatie"].tolist()

        # Add new columns for lat lon coordinates of permits
        self.query_result_df = self.add_permit_coordinates_columns(
            self.query_result_df, addresses
        )

        # Store rows where permit_lat and permit_lon are non null as healthy data
        self._healthy_df = self.query_result_df[
            self.query_result_df["permit_lat"].notnull()
            & self.query_result_df["permit_lon"].notnull()
        ]

        # Reset the healthy DataFrame index (to ensure alignment with internal lists)
        self._healthy_df = self._healthy_df.reset_index(drop=True)

        print(f"{len(self._healthy_df)} permits with valid coordinates.")
        if len(self._healthy_df) == 0:
            raise ValueError("No permit coordinates could be converted from addresses.")

        # Store rows where permit_lat and permit_lon are null as quarantine data
        self._quarantine_df = self.query_result_df[
            self.query_result_df["permit_lat"].isnull()
            | self.query_result_df["permit_lon"].isnull()
        ]
        print(f"{len(self._quarantine_df)} permits with invalid coordinates.")

        stats = self._healthy_df["object_category"].value_counts()
        print("Permit counts per object category:")
        for category, count in stats.items():
            print(f"  Category {category}: {count}")

        self._permits_coordinates = self._extract_permits_coordinates()
        self._permits_coordinates_geometry = self._convert_coordinates_to_point()

        # self.write_healthy_df_to_table()
        # self.write_qurantine_df_to_table()

    # TODO create the dataframe with spark!
    def create_dataframe(self, rows, colnames):
        """
        Create a DataFrame .
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

        self.query_result_df = df

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

        regex = r"(\D+)\s+(\d+)\s?(\S*)\s+(\d{4}\s*?[A-z]{2})?"
        return re.findall(regex, raw_address)

    def convert_address_to_coordinates(self, address):
        """
        Convert address to coordinates using BAG API.
        """
        response = None

        try:
            bag_url = "https://api.data.amsterdam.nl/atlas/search/adres/?q="
            split_dutch_address = self.split_dutch_street_address(address)
            if split_dutch_address:
                street_and_number = (
                    split_dutch_address[0][0] + " " + split_dutch_address[0][1]
                )
            else:
                print(
                    f"Warning: Unable to split Dutch street address using regex: {address}"
                )
                street_and_number = address

            with requests.get(bag_url + street_and_number, timeout=60) as response:
                results = json.loads(response.content)["results"]
                for result in results:
                    if "centroid" in result:
                        coordinates = result["centroid"]
                        if coordinates[0] > coordinates[1]:
                            latitude = coordinates[0]
                            longitude = coordinates[1]
                        else:
                            latitude = coordinates[1]
                            longitude = coordinates[0]
                        return [latitude, longitude]
                # If no centroid is found in any result
                return None
        except Exception as e:
            print(f"Error converting address into coordinates for {address}: {e}")
            print(f"Street and number: {split_dutch_address}")
            print(f'Response from server is {json.loads(response.content)["results"]}')

            return None

    def add_permit_coordinates_columns(self, df, addresses):
        """
        Add new columns "permit_lat" and "permit_lon" to the DataFrame and populate them with latitude and longitude values fetched from the BAG API.
        """
        latitudes = []
        longitudes = []
        for address in addresses:
            coordinates = self.convert_address_to_coordinates(address)
            if coordinates:
                latitudes.append(coordinates[0])
                longitudes.append(coordinates[1])
            else:  # None, because there was an exception while converting the address into coordinates
                longitudes.append(None)
                latitudes.append(None)

        df["permit_lat"] = latitudes
        df["permit_lon"] = longitudes
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

    def get_permits_coordinates(self):
        return self._permits_coordinates

    def get_permits_coordinates_geometry(self):
        return self._permits_coordinates_geometry

    def get_healthy_df(self):
        return self._healthy_df

    def get_quarantine_df(self):
        return self._quarantine_df
    
    def get_keyword_mapping(self):
        return {
                2: ["puinbak", "container", "keet", "cabin"],
                3: [],  # TODO: update keywords for category 3
                4: []   # TODO: update keywords for category 4
            }
    
    def calculate_distances_to_closest_permits_by_category(
        self, 
        containers_coordinates_df
    ):
        # Define the keyword lists per category.
        # These keywords determine which permits are considered relevant for each object category.
        keyword_mapping = self.get_keyword_mapping()
        result_dfs = []

        # Loop through each target object category.
        for cat in [2, 3, 4]:
            df_cat = containers_coordinates_df.filter(col("object_class") == cat)
            if df_cat.count() == 0:
                continue
            # Calculate the distances for the current category using the corresponding keyword list.
            result_df = self.calculate_distances_to_closest_permits_for_category(
                df_cat, keyword_mapping.get(cat, [])
            )
            result_dfs.append(result_df)

        if result_dfs:
            union_df = result_dfs[0]
            for df in result_dfs[1:]:
                union_df = union_df.union(df)
            return union_df
        else:
            return self.spark.createDataFrame([], schema=self.spark.table("your_schema_here").schema)

    def calculate_distances_to_closest_permits_for_category(
        self,
        containers_df,
        word_list: List[str]
    ):
        results = []

        # Filter permits to keep only those whose 'objecten' match.
        if word_list:
            def contains_keyword(permit_objs):
                try:
                    regex_pattern = re.compile(r"(?i)(" + "|".join(word_list) + r")")
                    for obj in permit_objs:
                        if regex_pattern.search(obj.get("object", "")):
                            return True
                except Exception as e:
                    print(f"Error in contains_keyword: {e}")
                return False

            mask = self._healthy_df["objecten"].apply(contains_keyword)
            filtered_indices = self._healthy_df.index[mask].tolist()
        else:
            # If no keyword filtering is required, include all healthy permits.
            filtered_indices = self._healthy_df.index.tolist()

        if not filtered_indices:
            return self.spark.createDataFrame([], 
                                              schema=self.spark.table("your_schema_here").schema)

        filtered_points = [self._permits_coordinates_geometry[i] for i in filtered_indices]
        filtered_ids = self._healthy_df.loc[filtered_indices, "id"].tolist()
        filtered_coords = [self._permits_coordinates[i] for i in filtered_indices]

        for row in containers_df.collect():
            container_lat = row.gps_lat
            container_lon = row.gps_lon

            container_location = Point(container_lat, container_lon)

            closest_permit_distances = []
            for permit_location in filtered_points:
                try:
                    # calculate distance between container point and permit point
                    permit_dist = geopy.distance.distance(
                        container_location.coords, permit_location.coords
                    ).meters
                except Exception:
                    permit_dist = 0
                    print("Error occurred:")
                    print(
                        f"Container location: {container_location}, {container_location.coords}"
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
                    # closest_permit_coordinates=permits_coordinates[min_distance_idx],
                    closest_permit_lat=filtered_coords[min_distance_idx][0],
                    closest_permit_lon=filtered_coords[min_distance_idx][1],
                )
            )
        results_df = self.spark.createDataFrame(results)

        return results_df
    
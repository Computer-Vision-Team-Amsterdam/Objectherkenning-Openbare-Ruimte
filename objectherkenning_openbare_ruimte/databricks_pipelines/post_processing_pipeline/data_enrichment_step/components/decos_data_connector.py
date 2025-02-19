import json
import re
from typing import List

import geopy.distance
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyspark.sql import Row
from shapely import wkb
from pyspark.sql.functions import col
from shapely.geometry import Point

from .reference_db_connector import ReferenceDatabaseConnector


class DecosDataHandler(ReferenceDatabaseConnector):
    def __init__(self, spark, az_tenant_id, db_host, db_name, db_port):
        super().__init__(az_tenant_id, db_host, db_name, db_port)
        self.spark = spark

    def process_query_result(self):
        """
        Process the results of the query.
        """
        query = f"SELECT id, kenmerk, locatie, geometrie_locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '{date_to_query}' AND datum_object_tm >= '{date_to_query}'"  # nosec B608
        print(f"Querying the database for date {date_to_query}...")
        result_df = self.run(query)

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

        result_df["objecten"] = result_df["objecten"].apply(
            lambda x: _safe_json_load(x) if x else []
        )

        # Assign object category to each permit
        result_df["object_category"] = result_df["objecten"].apply(get_object_category)

        # Keep only permits that match at least one of the keywords (i.e. have a valid category)
        result_df = result_df[result_df["object_category"].notnull()]
        
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

        print(f"{len(self._healthy_df)} permits with valid coordinates.")
        if len(self._healthy_df) == 0:
            raise ValueError("No permit coordinates could be converted from addresses.")

        # Store rows where permit_lat and permit_lon are null as quarantine data
        self._quarantine_df = result_df[
            result_df["permit_lat"].isnull() | result_df["permit_lon"].isnull()
        ]
        print(f"{len(self._quarantine_df)} permits with invalid coordinates.")

        stats = self._healthy_df["object_category"].value_counts()
        print("Permit counts per object category:")
        for category, count in stats.items():
            print(f"  Category {self.active_object_categories[category]}: {count}")

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

        return df

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

    def get_benkagg_adresseerbareobjecten_by_address(
        self, street, house_number, postcode
    ):
        query = f"select openbareruimte_naam, huisnummer, huisletter, postcode, adresseerbaar_object_punt_geometrie from benkagg_adresseerbareobjecten where openbareruimte_naam = '{street}' and huisnummer = '{house_number}' and postcode = '{postcode}'"  # nosec B608
        return self.run(query)

    def get_benkagg_adresseerbareobjecten_by_id(self, id):
        query = f"select openbareruimte_naam, huisnummer, huisletter, postcode, adresseerbaar_object_punt_geometrie from benkagg_adresseerbareobjecten where identificatie='{id}'"  # nosec B608
        return self.run(query)

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
    
    def get_keyword_mapping(self):
        full_mapping =  {
            2: ["puinbak", "container", "keet", "cabin"],
            3: ["toilet"],  # TODO: check with BOR if more are relevant
            4: ["steiger"]   # TODO: check with BOR if more are relevant
        }
        active_keys = set(self.active_object_categories.keys())
        return {k: full_mapping[k] for k in active_keys}
    
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
                    closest_permit_lat=filtered_coords[min_distance_idx][0],
                    closest_permit_lon=filtered_coords[min_distance_idx][1],
                )
            )
        results_df = self.spark.createDataFrame(results)

        return results_df
    
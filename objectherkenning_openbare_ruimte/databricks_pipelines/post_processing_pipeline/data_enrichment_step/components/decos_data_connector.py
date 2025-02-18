import json
import re
from typing import List

import geopy.distance
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyspark.sql import Row
from shapely import wkb
from shapely.geometry import Point

from .reference_db_connector import ReferenceDatabaseConnector


class DecosDataHandler(ReferenceDatabaseConnector):

    def __init__(self, spark, az_tenant_id, db_host, db_name, db_port):
        super().__init__(az_tenant_id, db_host, db_name, db_port)
        self.spark = spark

    @staticmethod
    def is_container_permit(objects):
        """
        Check whether permit is for a container based on the 'objecten' column.
        """
        container_words = [
            "puinbak",
            "container",
            "keet",
            "cabin",
        ]

        regex_pattern = re.compile(r"(?i)(" + "|".join(container_words) + r")")
        try:
            for obj in objects:
                if bool(regex_pattern.search(obj["object"])):
                    return True
        except Exception as e:
            print(f"There was an exception in the is_container_permit function: {e}")

        return False

    def query_and_process_object_permits(self, date_to_query):
        """
        Process the results of the query.
        """
        query = f"SELECT id, kenmerk, locatie, geometrie_locatie, objecten FROM vergunningen_werk_en_vervoer_op_straat WHERE datum_object_van <= '{date_to_query}' AND datum_object_tm >= '{date_to_query}'"  # nosec B608
        print(f"Querying the database for date {date_to_query}...")
        result_df = self.run(query)
        print(
            "Processing object permits (filter by object name, convert address to coordinates)..."
        )

        def _safe_json_load(x):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                return []

        result_df["objecten"] = result_df["objecten"].apply(
            lambda x: _safe_json_load(x) if x else []
        )

        # Only keep rows with permits about containers
        result_df = result_df[result_df["objecten"].apply(self.is_container_permit)]

        # Only keep rows where location of the permit is not null
        result_df = result_df[result_df["locatie"].notnull()]

        # Add new columns for lat lon coordinates of permits
        result_df = self.add_permit_coordinates_columns(result_df)

        # Store rows where permit_lat and permit_lon are non null as healthy data
        self._healthy_df = result_df[
            result_df["permit_lat"].notnull() & result_df["permit_lon"].notnull()
        ]

        print(f"{len(self._healthy_df)} container permits with valid coordinates.")
        if len(self._healthy_df) == 0:
            raise ValueError("No permit coordinates could be converted from addresses.")

        # Store rows where permit_lat and permit_lon are null as quarantine data
        self._quarantine_df = result_df[
            result_df["permit_lat"].isnull() | result_df["permit_lon"].isnull()
        ]
        print(f"{len(self._quarantine_df)} container permits with invalid coordinates.")

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
        regex = r"(\D+)\s+(\d+)\s?(\S*)\s+(\d{4}\s*?[A-z]{2})?"
        return re.findall(regex, raw_address)

    def convert_EWKB_geometry_to_coordinates(self, ewkb_geometry):
        # Convert the hex string to binary and load it as a Shapely geometry.
        # (Note: Shapely will ignore the embedded SRID so you must know it.)
        point = wkb.loads(bytes.fromhex(ewkb_geometry))

        # Extract the X and Y coordinates (these are in EPSG:28992, Dutch RD New)
        x, y = point.x, point.y

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
                street = split_dutch_address[0][0]
                house_number = split_dutch_address[0][1]
                number_extension = (
                    split_dutch_address[0][2] if split_dutch_address[0][2] else None
                )
                postcode = split_dutch_address[0][3]
                query = f"select openbareruimte_naam, huisnummer, huisletter, postcode, adresseerbaar_object_punt_geometrie_wgs_84 from benkagg_adresseerbareobjecten where openbareruimte_naam = '{street}' and huisnummer = '{house_number}' and huisletter = {number_extension} and postcode = {postcode}"  # nosec B608
                print(f"Querying the database for address {address}...")
                result_df = self.run(query)
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
                    result_df["adresseerbaar_object_punt_geometrie_wgs_84"].iloc[0]
                )
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
        # Use a list comprehension to process rows efficiently
        coords = [
            (
                self.convert_EWKB_geometry_to_coordinates(geom)
                if pd.notnull(geom)
                else self.convert_address_to_coordinates(addr)
            )
            for geom, addr in zip(df["geometrie_locatie"], df["locatie"])
        ]
        print(f"Converted {len(coords)} addresses to coordinates.")
        print(f"Coordinates: {coords}")

        # Unpack the coordinate tuples into separate lists; handle None cases
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

    def calculate_distances_to_closest_permits(
        self,
        containers_coordinates_df,
    ):
        results = []

        for row in containers_coordinates_df.collect():
            container_lat = row.gps_lat
            container_lon = row.gps_lon

            container_location = Point(container_lat, container_lon)

            closest_permit_distances = []
            for permit_location in self._permits_coordinates_geometry:
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
                    closest_permit_id=self.get_permits_ids()[min_distance_idx],
                    # closest_permit_coordinates=permits_coordinates[min_distance_idx],
                    closest_permit_lat=self._permits_coordinates[min_distance_idx][0],
                    closest_permit_lon=self._permits_coordinates[min_distance_idx][1],
                )
            )
        results_df = self.spark.createDataFrame(results)

        return results_df

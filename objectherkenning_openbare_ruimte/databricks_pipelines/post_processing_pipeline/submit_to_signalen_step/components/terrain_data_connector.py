from typing import Any, List, Optional

import pandas as pd
import requests
from shapely import Polygon, wkb

from objectherkenning_openbare_ruimte.databricks_pipelines.common.reference_db_connector import (  # noqa: E402
    ReferenceDatabaseConnector,
)


class TerrainDatabaseConnector(ReferenceDatabaseConnector):

    def __init__(
        self,
        az_tenant_id: str,
        db_host: str,
        db_name: str,
        db_port: int,
    ) -> None:
        super().__init__(az_tenant_id, db_host, db_name, db_port)

    def query_and_process_public_terrains(self) -> Optional[pd.DataFrame]:
        """
        Query the database for public terrain geometries and process the results.

        The function runs a query to fetch geometries for public terrain and converts them
        from WKB hex strings to Shapely geometry objects. Processed geometries are stored
        in the _public_terrains attribute.
        """
        query = "SELECT geometrie FROM beheerkaart_basis_kaart WHERE agg_indicatie_belast_recht = FALSE"
        print("Querying public terrain data from the database...")
        result_df = self.run(query)
        if result_df.empty:
            print("No public terrain data found.")
            return None

        result_df["polygon"] = result_df["geometrie"].apply(
            lambda x: self.convert_wkb(x)
        )
        result_df = result_df[result_df["polygon"].notnull()]
        public_terrains = result_df.to_dict(orient="records")
        return public_terrains

    def query_and_process_stadsdelen(self) -> Optional[dict]:
        """
        Query the database for stadsdelen geometries and process the results.

        The function runs a query to fetch geometries for stadsdelen and converts them
        from WKB hex strings to Shapely geometry objects.
        """
        url = "https://api.data.amsterdam.nl/v1/gebieden/stadsdelen/"
        print("Querying stadsdelen API...")
        try:
            result = requests.get(url)
            result.raise_for_status()
            stadsdelen_dict: dict[str, List[dict]] = {
                "stadsdelen": []
            }
            for stadsdeel in result.json()["_embedded"]["stadsdelen"]:
                stadsdelen_dict["stadsdelen"].append(
                    {
                        "naam": stadsdeel["naam"],
                        "code": stadsdeel["code"],
                        "polygon": Polygon(stadsdeel["geometrie"]["coordinates"][0]),
                    }
                )
            return stadsdelen_dict
        except requests.exceptions.RequestException as e:
            print(f"Error querying stadsdelen API: {e}")
            return None

    def convert_wkb(self, hex_str: str) -> Optional[Any]:
        """
        Convert a WKB hex string to a Shapely geometry object.

        Parameters:
            hex_str: A hexadecimal string representing the WKB geometry.

        Returns:
            A Shapely geometry object if the conversion is successful; None otherwise.
        """
        try:
            return wkb.loads(bytes.fromhex(hex_str))
        except Exception as e:
            print(f"Error converting geometry {hex_str}: {e}")
            return None

    def create_dataframe(self, rows: List[Any], colnames: List[str]) -> pd.DataFrame:
        """
        Create a pandas DataFrame from provided row data and column names.

        Parameters:
            rows: A list of row data, where each row is an iterable of values.
            colnames: A list of column names corresponding to the row data.

        Returns:
            A pandas DataFrame constructed from the provided data.
        """
        data = [dict(zip(colnames, row)) for row in rows]
        df = pd.DataFrame(data, columns=colnames)
        return df

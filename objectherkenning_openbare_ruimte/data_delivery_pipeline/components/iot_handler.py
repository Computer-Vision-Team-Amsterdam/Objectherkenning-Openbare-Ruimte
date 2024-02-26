import os
from contextlib import contextmanager
from typing import Dict

from azure.core.exceptions import AzureError
from azure.iot.device import IoTHubDeviceClient, Message
from azure.storage.blob import BlobClient

from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)


class IoTHandler:
    def __init__(self):
        iot_settings = ObjectherkenningOpenbareRuimteSettings.get_settings()[
            "azure_iot"
        ]
        self.connection_string = (
            f"HostName={iot_settings['hostname']};"
            f"DeviceId={iot_settings['device_id']};"
            f"SharedAccessKey={iot_settings['shared_access_key']}"
        )

    @contextmanager
    def _connect(self):
        device_client = IoTHubDeviceClient.create_from_connection_string(
            self.connection_string
        )
        try:
            device_client.connect()
            yield device_client
        finally:
            device_client.shutdown()

    def deliver_message(self, message_content: str):
        message = Message(message_content)
        with self._connect() as device_client:
            device_client.send_message(message)

    def upload_file(self, file_path: str):
        with self._connect() as device_client:
            blob_name = os.path.basename(file_path)
            storage_info = device_client.get_storage_info_for_blob(blob_name)

            # Upload to blob
            success, result = self._store_blob(storage_info, file_path)

            if success:
                print("Upload succeeded. Result is: \n")
                print(result)
                print()

                device_client.notify_blob_upload_status(
                    storage_info["correlationId"], True, 200, "OK: {}".format(file_path)
                )
            else:
                # If the upload was not successful, the result is the exception object
                print("Upload failed. Exception is: \n")
                print(result)
                print()

                device_client.notify_blob_upload_status(
                    storage_info["correlationId"],
                    False,
                    result.status_code,
                    str(result),
                )

    @staticmethod
    def _store_blob(blob_info: Dict[str, str], file_name):
        try:
            sas_url = "https://{}/{}/{}{}".format(
                blob_info["hostName"],
                blob_info["containerName"],
                blob_info["blobName"],
                blob_info["sasToken"],
            )

            print(
                "\nUploading file: {} to Azure Storage as blob: {} in container {}\n".format(
                    file_name, blob_info["blobName"], blob_info["containerName"]
                )
            )

            # Upload the specified file
            with BlobClient.from_blob_url(sas_url) as blob_client:
                with open(file_name, "rb") as f:
                    result = blob_client.upload_blob(f, overwrite=True)
                    return True, result

        except FileNotFoundError as ex:
            return False, ex

        except AzureError as ex:
            return False, ex

import logging
from contextlib import contextmanager
from typing import Dict

from azure.core.exceptions import AzureError
from azure.iot.device import IoTHubDeviceClient, Message
from azure.storage.blob import BlobClient

logger = logging.getLogger("data_delivery_pipeline")


class IoTHandler:
    def __init__(
        self,
        hostname: str,
        device_id: str,
        shared_access_key: str = None,
    ):
        """
        Object that handles the connection with IoT and the delivery of messages and files.

        Parameters
        ----------
        hostname
            Azure IoT hostname
        device_id
            Device id of the device connecting
        shared_access_key
            Access key to authenticate
        """
        self.connection_string = (
            f"HostName={hostname};"
            f"DeviceId={device_id};"
            f"SharedAccessKey={shared_access_key}"
        )

    @contextmanager
    def _connect(self):
        """
        Connects to Azure IoT.

        Example
        -------
        with self._connect() as device_client:
            device_client.send_message(message)
        """
        device_client = IoTHubDeviceClient.create_from_connection_string(
            self.connection_string,
            websockets=True,
        )
        try:
            device_client.connect()
            yield device_client
        finally:
            device_client.shutdown()

    def deliver_message(self, message_content: str):
        """
        Delivers a message to Azure IoT.

        Parameters
        ----------
        message_content
            Content of the message.
        """
        message = Message(
            message_content, content_encoding="utf-8", content_type="application/json"
        )
        with self._connect() as device_client:
            device_client.send_message(message)

    def upload_file(self, file_source_path: str, file_destination_path: str):
        """
        Uploads a file to Azure IoT.

        Parameters
        ----------
        file_source_path
            Path of the file to upload.
        file_destination_path
            Path of where to upload it.
        """
        with self._connect() as device_client:
            storage_info = device_client.get_storage_info_for_blob(
                file_destination_path
            )

            success, result = self._store_blob(storage_info, file_source_path)
            if success:
                logger.info(f"Upload succeeded. Result is: {result}")
                device_client.notify_blob_upload_status(
                    storage_info["correlationId"],
                    True,
                    200,
                    "OK: {}".format(file_source_path),
                )
            else:
                logger.error(f"Upload failed. Exception is: {result}")
                device_client.notify_blob_upload_status(
                    storage_info["correlationId"],
                    False,
                    result.status_code,
                    str(result),
                )
                raise Exception(result)

    @staticmethod
    def _store_blob(blob_info: Dict[str, str], file_name: str):
        """
        Stores file on a blob container.

        Parameters
        ----------
        blob_info
            Dictionary containing: hostname, containername, blobname, and sastoken.
        file_name
            Path of the file to store
        """
        try:
            sas_url = "https://{}/{}/{}{}".format(
                blob_info["hostName"],
                blob_info["containerName"],
                blob_info["blobName"],
                blob_info["sasToken"],
            )

            logger.info(
                "\nUploading file: {} to Azure Storage as blob: {} in container {}\n".format(
                    file_name, blob_info["blobName"], blob_info["containerName"]
                )
            )

            with BlobClient.from_blob_url(sas_url) as blob_client:
                with open(file_name, "rb") as f:
                    result = blob_client.upload_blob(f, overwrite=True)
                    return True, result

        except FileNotFoundError as ex:
            return False, ex

        except AzureError as ex:
            return False, ex

import logging
import os
import sys
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import py360convert as p3c

sys.path.append("../../..")

logger = logging.getLogger(__name__)


class EquirectangularToCubemapConverter:
    """
    - Convert an equirectangular image to a set of six cubic faces using py360convert library
    (https://github.com/imandrealombardo/py360convert)
    - Convert YOLO format annotations to the corresponding cubic faces.
    - Visualize the annotations on both the equirectangular and cubic images.
    """

    def __init__(self, input_path: str, output_path: str, face_width: int):
        """
        Parameters
        ----------
        input_path : str
            Path to the equirectangular image folder.
        output_path : str
            Path to the folder where to store the folders with the six cubic faces for each image.
        face_width : int
            Width of each face in the cubemap.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.face_width = face_width

    @staticmethod
    def _convert_face_idx_to_name(face_idx: int) -> str:
        """
        Convert face index to face name.

        The mapping is:
        0 -> front, 1 -> right, 2 -> back,
        3 -> left, 4 -> top, 5 -> bottom.

        Parameters
        ----------
        face_idx : int
            Index of the face.

        Returns
        -------
        str
            Name of the face.
        """
        face_names = ["front", "right", "back", "left", "top", "bottom"]
        return face_names[face_idx]

    @staticmethod
    def _convert_yolo_to_corners(yolo_annotation: str, P_w: int, P_h: int) -> Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ]:
        """
        Convert YOLO annotation with normalized values to absolute corner coordinates.

        Parameters
        ----------
        yolo_annotation : str
            YOLO annotation string in the format:
            "<class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>".
        P_w : int
            Width of the image.
        P_h : int
            Height of the image.

        Returns
        -------
        tuple
            A tuple containing the top-left, top-right, bottom-left,
            and bottom-right corner coordinates.
        """

        _, x_center_norm, y_center_norm, w_norm, h_norm = map(
            float, yolo_annotation.split()
        )

        x_center_abs = x_center_norm * P_w
        y_center_abs = y_center_norm * P_h
        w_abs = w_norm * P_w
        h_abs = h_norm * P_h

        x_min = x_center_abs - (w_abs / 2)
        y_min = y_center_abs - (h_abs / 2)
        x_max = x_center_abs + (w_abs / 2)
        y_max = y_center_abs + (h_abs / 2)

        top_left = (x_min, y_min)
        top_right = (x_max, y_min)
        bottom_left = (x_min, y_max)
        bottom_right = (x_max, y_max)

        return top_left, top_right, bottom_left, bottom_right

    @staticmethod
    def _convert_corners_to_yolo(
        top_left: Tuple[float, float],
        bottom_right: Tuple[float, float],
        face_w: int,
        face_h: int,
    ) -> Tuple[str, float, float, float, float]:
        """
        Convert absolute corner coordinates to YOLO annotation with normalized values.

        Parameters
        ----------
        top_left : float
            Top-left absolute corner coordinates.
        bottom_right : float
            Bottom-right absolute corner coordinates.
        face_w : int
            Width of the face.
        face_h : int
            Height of the face.

        Returns
        tuple
            A tuple containing the normalized center coordinates and size:
            "<x_center_norm> <y_center_norm> <w_norm> <h_norm>".
        -------

        """
        x_min, y_min = top_left
        x_max, y_max = bottom_right

        x_center_abs = (x_min + x_max) / 2
        y_center_abs = (y_min + y_max) / 2
        width_abs = x_max - x_min
        height_abs = y_max - y_min

        x_center_norm = x_center_abs / face_w
        y_center_norm = y_center_abs / face_h
        width_norm = width_abs / face_w
        height_norm = height_abs / face_h

        return x_center_norm, y_center_norm, width_norm, height_norm

    @staticmethod
    def _write_annotation_to_file(
        output_path: str,
        img_name: str,
        face_idx: int,
        annotation: Tuple[str, float, float, float, float],
    ) -> None:
        """
        Write the converted YOLO annotation to the specified annotation file.

        Parameters
        ----------
        output_path : str
            The base directory where the annotation files are stored.
        img_name : str
            The name of the image file, used to derive the folder name for storing annotations.
        face_idx : str
            The index of the face, used to derive the annotation file name.
        annotation : str
            The converted YOLO annotation to write to the file.

        Returns
        -------
        None
        """
        face_name = EquirectangularToCubemapConverter._convert_face_idx_to_name(
            face_idx
        )

        annotation_file = os.path.join(output_path, img_name, f"{face_name}.txt")
        converted_yolo_annotation_str = " ".join(map(str, annotation))

        os.makedirs(os.path.dirname(annotation_file), exist_ok=True)

        with open(annotation_file, "a") as file:
            file.write(converted_yolo_annotation_str + "\n")

        logging.info(f"Annotation written to {annotation_file}")

    @staticmethod
    def _adjust_bounding_box_corners(
        processed_corners: Dict[str, Any], face_width: int
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Adjust the bounding box corners based on already processed corners within a face.

        Parameters
        ----------

        processed_corners : dict
            A dictionary containing the corners and other boundary points that have been processed. 
            Keys include corner tags ('TL', 'TR', 'BL', 'BR') for each corner and may also include 
            'x_max' and 'y_max' which represent the maximum allowable x and y coordinates 
            for the bounding box on that face, ensuring the bounding box fits within the face boundaries.
            The values are tuples of (x, y) coordinates that represent the adjusted positions 
            of these corners on a specific cubemap face. These adjustments ensure that the bounding box 
            accurately represents the object within the constraints of that face, 
            considering any geometrical shifts needed during the conversion from equirectangular to cubemap projection.

        face_width : int
            The width (and height, assuming square faces) of each face in the cubemap. This is
            used to ensure bounding box coordinates stay within the face boundaries.

        Returns
        -------
        tuple
            A tuple containing the adjusted top-left (tl_star) and bottom-right (br_star) coordinates
            of the bounding box, ensuring the box fits properly within the cube face.

        Notes
        -----
        This function dynamically adjusts the bounding box coordinates based on the processed corners 
        and explicit boundary settings. For example, 'x_max' and 'y_max' provide upper limits on the 
        bounding box size when corners like 'TR' or 'BR' have been processed, adjusting their values to 
        ensure the object is encapsulated within the cube face boundaries.

        An explanation with pictures is available in the Wiki documentation.
        """

        tl_x = 0
        tl_y = 0
        br_x = face_width
        br_y = face_width

        # Apply adjustments from processed_corners for top-left coordinates
        if "TL" in processed_corners:
            tl_x, tl_y = processed_corners["TL"]
        elif "BL" in processed_corners:
            tl_x, _ = processed_corners["BL"]

        if "TR" in processed_corners:
            # TR's y position might adjust TL*'s y if TL is not explicitly processed
            tl_y = max(tl_y, processed_corners["TR"][1])

        # Apply adjustments from processed_corners for bottom-right coordinates
        if "BR" in processed_corners:
            br_x, br_y = processed_corners["BR"]
        else:
            if "x_max" in processed_corners:
                br_x = processed_corners["x_max"]
            if "y_max" in processed_corners:
                br_y = processed_corners["y_max"]
            if "TR" in processed_corners and "x_max" not in processed_corners:
                # If BR is not processed but TR is, adjust BR*'s x to TR's x
                br_x = processed_corners["TR"][0]
            if "BL" in processed_corners and "y_max" not in processed_corners:
                # If BR is not processed but BL is, adjust BR*'s y to BL's y
                br_y = processed_corners["BL"][1]

        tl_star = (tl_x, tl_y)
        br_star = (br_x, br_y)

        return tl_star, br_star

    @staticmethod
    def _reproject_xy_coor_to_face(
        coor_xy_face: np.ndarray, coor: np.ndarray, face_w: int
    ) -> Tuple[int, int]:
        """
        Compute a pair of coordinates (y_proj, x_proj) that indicate the position
        on the cube face closest to the original equirectangular point.

        To do that, we first compute the difference between each point's UV coordinates
        on the cube face (coor_xy_face) and the original point's equirectangular coordinates (coor).
        The difference is a 2D vector for each point on the cube face that indicates how far
        and in which direction each cube face point is from the original equirectangular point.

        np.linalg.norm(coor_xy_face - coor, axis=-1) calculates the Euclidean distance (norm)
        between the original point and each point on the cube face.

        np.argmin(euclidean_distance) finds the index of the minimum value in the array of distances
        calculated in the previous step. This index corresponds to the location on the cube face
        that is closest to the original point in equirectangular space.

        np.unravel_index(min_distance_index, (face_w, face_w)) converts the flattened index back into
        2D coordinates within the context of the cube face's dimensions.

        Parameters
        ----------
        coor_xy_face : numpy.ndarray
            UV coordinates of the cube face.
        coor : numpy.ndarray
            Equirectangular coordinates of the original point.
        face_w : int
            Width of the cube face.

        Returns
        -------
        tuple
            A tuple containing the new coordinates within the face.
        """

        vector_distance = coor_xy_face - coor
        euclidean_distance = np.linalg.norm(vector_distance, axis=-1)
        min_distance_index = np.argmin(euclidean_distance)

        y_proj, x_proj = np.unravel_index(min_distance_index, (face_w, face_w))

        return int(y_proj), int(x_proj)

    @staticmethod
    def _adjust_coordinates_based_on_corner(
        tag: str,
        corner: Tuple[float, float],
        processed_corners: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Adjust the coordinates of a bounding box based on the processed corners within a face,
        and possibly based on explicit 'x_max' and 'y_max' settings if they are defined.

        This function updates the coordinates of a bounding box in a cubemap face by considering
        the positions of already processed corners. Depending on the corner being processed (tag),
        it may adjust the x and y maximums or minimums of the bounding box to ensure the box
        accurately represents the object within the constraints of the face. The function is
        typically used in a scenario where bounding boxes may span across multiple faces of a
        cubemap, requiring adjustments to fit within a single face properly.

        Parameters
        ----------
        tag : str
            The tag indicating which corner of the bounding box is being processed.
            Expected values are 'TL' (top-left), 'TR' (top-right), 'BL' (bottom-left), and 'BR' (bottom-right).

        corner : tuple
            A tuple of (x, y) coordinates representing the position of the corner on the face.

        processed_corners : dict
            A dictionary maintaining the state of already processed corners and explicit boundary limits.
            Keys are corner tags ('TL', 'TR', 'BL', 'BR') and may also include 'x_max' and 'y_max' which represent 
            the maximum values that the bounding box coordinates can take on the x and y axes, respectively.
            This dictionary is updated in-place to reflect the adjusted coordinates based on the current corner's
            processing.

        Returns
        -------
        dict
            The updated dictionary of processed corners after adjusting the coordinates based on
            the current corner. This dictionary provides a comprehensive mapping of all corners
            that have been adjusted, ready for further processing or final bounding box calculation.

        Notes
        -----
        The adjustment logic is dependent on the spatial relationship between the current corner
        being processed and previously processed corners. For example, processing the 'TR' (top-right)
        corner may adjust the 'x_max' value if the 'TL' (top-left) corner has already been processed,
        ensuring the bounding box correctly encapsulates the object within the face's boundaries.

        An explanation with pictures is available in the Wiki documentation.
        """

        if "TL" in processed_corners:
            if tag == "TR":
                processed_corners["x_max"] = corner[0]  # Adjust BR x to TR x
            elif tag == "BL":
                processed_corners["y_max"] = corner[1]  # Adjust BR y to BL y

        if "TR" in processed_corners:
            tr = processed_corners["TR"]
            # For TL, adjust x_min; no need for adjustment for TR itself
            if tag == "TL":
                processed_corners["x_min"] = corner[0]  # Adjust TL x to TL x
            elif tag == "BR":
                processed_corners["x_max"] = tr[0]  # Keep BR x at TR x
                processed_corners["y_max"] = corner[1]  # Adjust BR y to BR y

        if "BL" in processed_corners:
            bl = processed_corners["BL"]
            # For TL, adjust y_min; for BR, adjust y_max; no change for BL itself
            if tag == "TL":
                processed_corners["y_min"] = corner[1]  # Adjust TL y to TL y
            elif tag == "BR":
                processed_corners["y_max"] = bl[1]  # Keep BR y at BL y

        if "BR" in processed_corners:
            br = processed_corners["BR"]
            # For TR, adjust x_max; for BL, adjust y_max; no change for BR itself
            if tag == "TR":
                processed_corners["x_max"] = br[0]  # Keep BR x at BR x
            elif tag == "BL":
                processed_corners["y_max"] = br[1]  # Adjust BR y to BR y

        processed_corners[tag] = corner

        return processed_corners

    @staticmethod
    def _reproject_point(
        point: Tuple[float, float], pano_width: int, pano_height: int, face_w: int
    ) -> Tuple[int, Tuple[int, int]]:
        """
        Reproject a point from (x, y) equirectangular coordinates to a set of coordinates
        in a specific face (face_idx) of a cube.

        p3c.utils.equirect_facetype(pano_height, pano_width) determines which face
        of a cube (in a cubemap representation) each pixel in an equirectangular
        projection image corresponds to.

        p3c.utils.xyzcube(face_w) generates the XYZ coordinates (3D) of the six faces of a unit cube.

        p3c.utils.xyz2uv(face_xyz) converts the XYZ coordinates (3D) to UV coordinates (spherical).

        p3c.utils.uv2coor(uv_face, pano_height, pano_width) converts UV coordinates (spherical)
        into pixel coordinates (2D plane).

        Parameters
        ----------
        point : tuple
            (x, y) coordinates of the point.
        pano_width : int
            Width of the panorama image.
        pano_height : int
            Height of the panorama image.
        face_w : int
            Width of each face in the cube map.

        Returns
        -------
        tuple
            A tuple containing the face index and the new coordinates within the face.

        """

        x, y = point

        coor = np.array([x, y])
        mapping = p3c.utils.equirect_facetype(pano_height, pano_width)

        x_int = int(x)
        y_int = int(y)

        # Ensure the indices are within the valid range
        x_int = max(0, min(x_int, pano_width - 1))
        y_int = max(0, min(y_int, pano_height - 1))

        face_idx = mapping[y_int, x_int]

        cube_faces_xyz = p3c.utils.xyzcube(face_w)
        face_xyz = cube_faces_xyz[:, face_idx * face_w : (face_idx + 1) * face_w, :]

        uv_face = p3c.utils.xyz2uv(face_xyz)
        coor_xy_face = p3c.utils.uv2coor(uv_face, pano_height, pano_width)

        y_proj, x_proj = EquirectangularToCubemapConverter._reproject_xy_coor_to_face(
            coor_xy_face, coor, face_w
        )
        y_proj, x_proj = int(y_proj), int(x_proj)

        return face_idx, (x_proj, y_proj)

    @staticmethod
    def _draw_annotations_from_file(
        annotation_path: str, image: np.ndarray, draw_thickness: int = 2
    ) -> None:
        """
        Draw YOLO annotations on the specified image.

        Parameters
        ----------
        annotation_path : str
            The path to the annotation file.
        image : numpy.ndarray
            The image to draw the annotations on.
        draw_thickness : int, optional
            The thickness of the lines to draw.

        Returns
        -------
        None
        """

        if not os.path.exists(annotation_path):
            logging.error(f"Annotation file {annotation_path} not found.")
            return

        P_w, P_h = image.shape[1], image.shape[0]

        with open(annotation_path, "r") as file:
            for line in file:
                _, x_center_norm, y_center_norm, w_norm, h_norm = map(
                    float, line.split()
                )

                x_center_abs = int(x_center_norm * P_w)
                y_center_abs = int(y_center_norm * P_h)
                width_abs = int(w_norm * P_w)
                height_abs = int(h_norm * P_h)

                x_min = int(x_center_abs - width_abs / 2)
                y_min = int(y_center_abs - height_abs / 2)
                x_max = int(x_center_abs + width_abs / 2)
                y_max = int(y_center_abs + height_abs / 2)

                cv2.rectangle(
                    image, (x_min, y_min), (x_max, y_max), (0, 0, 255), draw_thickness
                )

    def visualize_annotations_with_corners(
        self, img_folder: str, face_name: str
    ) -> None:
        """
        Visualize annotations on the specified cubic face image.

        Parameters:
        img_folder : str
            The name of the folder containing the images and annotations.
        face_name : str
            The name of the face to visualize annotations for.

        Returns:
        None
        """
        img_path = os.path.join(self.output_path, img_folder, f"{face_name}.png")
        annotation_path = os.path.join(self.output_path, img_folder, f"{face_name}.txt")

        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Image file {img_path} not found.")
            return

        if not os.path.exists(annotation_path):
            logging.error(f"Annotation file {annotation_path} not found.")
            return

        img_copy = img.copy()

        EquirectangularToCubemapConverter._draw_annotations_from_file(
            annotation_path, img_copy, draw_thickness=2
        )

        visualized_img_path = os.path.join(
            self.output_path, img_folder, f"{face_name}_annotated_corners.png"
        )
        cv2.imwrite(visualized_img_path, img_copy)

    def visualize_annotations_on_equirectangular_image(self, img_path: str) -> None:
        """
        Visualize YOLO annotations on the original equirectangular image.

        Parameters
        ----------
        img_path : str
            The name of the image file to visualize annotations for.

        Returns
        -------
        None
        """

        image_file_path = os.path.join(self.input_path, img_path)
        annotation_file_path = os.path.join(
            self.input_path, img_path.split(".")[0] + ".txt"
        )

        image = cv2.imread(image_file_path)
        if image is None:
            logging.error(f"Image file {image_file_path} not found.")
            return

        EquirectangularToCubemapConverter._draw_annotations_from_file(
            annotation_file_path, image, draw_thickness=2
        )

        annotated_folder = os.path.join(self.input_path, "input_annotated")
        if not os.path.exists(annotated_folder):
            os.makedirs(annotated_folder)

        visualized_img_path = os.path.join(
            annotated_folder, img_path.split(".")[0] + "_annotated.png"
        )
        cv2.imwrite(visualized_img_path, image)

        logging.info(f"Annotated equirectangular image saved to {visualized_img_path}")

    def convert_image(self, img_path: str) -> None:
        """
        Convert an equirectangular image to a set of six cubic faces using py360convert.e2c.

        Parameters
        ----------
        img_path : str
            Path to the input image file.

        Returns
        -------
        None

        """

        img = os.path.join(self.input_path, img_path)
        img = cv2.imread(img)

        try:
            if img is None:
                raise FileNotFoundError(f"File {img} is empty or corrupted")
        except FileNotFoundError as e:
            logging.error(str(e))
            return

        try:
            front, right, back, left, top, bottom = p3c.e2c(
                img, face_w=self.face_width, mode="bilinear", cube_format="list"
            )

            base_name = os.path.basename(img_path)
            folder = os.path.splitext(base_name)[0]
            directory = os.path.join(self.output_path, folder)
            if not os.path.exists(directory):
                os.makedirs(directory)

            cv2.imwrite(f"{directory}/front.png", front)
            cv2.imwrite(f"{directory}/right.png", right)
            cv2.imwrite(f"{directory}/back.png", back)
            cv2.imwrite(f"{directory}/left.png", left)
            cv2.imwrite(f"{directory}/top.png", top)
            cv2.imwrite(f"{directory}/bottom.png", bottom)

            logging.info(f"Processed image: {img_path}.")
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")

    def process_annotations(self, img_path: str) -> None:
        """
        Process YOLO format annotations for a given equirectangular image, reprojecting them onto
        the corresponding faces of a cubemap and converting them back into YOLO format annotations
        for each face. This function handles both cases where a bounding box is contained within a
        single face and spans across multiple faces.

        For each annotation, it determines the cubemap face(s) onto which the annotation's bounding
        box projects, adjusts the bounding box coordinates as necessary for each face, and saves
        the new annotations in YOLO format in the respective annotation files for each face.

        Parameters
        ----------
        img_path : str
            The filename of the equirectangular image being processed. This name is also used to
            derive the names of the annotation files.

        Returns
        -------
        None
        """
        logging.info(f"===== Processing annotations for {img_path} =====")

        base_name = os.path.basename(img_path)
        img_name = os.path.splitext(base_name)[0]
        annotation_file = img_name + ".txt"
        annotations_path = os.path.join(self.input_path, annotation_file)

        try:
            with open(annotations_path, "r") as file:
                lines = file.readlines()
        except FileNotFoundError:
            logging.error(f"File {annotations_path} not found")
            return

        image_file_path = os.path.join(self.input_path, img_path)
        image = cv2.imread(image_file_path)

        try:
            if image is None:
                raise FileNotFoundError(f"File {image} is empty or corrupted")
        except FileNotFoundError as e:
            logging.error(str(e))
            return

        P_h, P_w, _ = image.shape

        # Process each annotation line by line
        for line in lines:
            face_corners: Dict[int, Dict[str, Tuple[int, int]]] = {}

            yolo_annotation = line.strip()
            yolo_annotation_class = yolo_annotation.split()[0]

            top_left, top_right, bottom_left, bottom_right = (
                EquirectangularToCubemapConverter._convert_yolo_to_corners(
                    yolo_annotation, P_w, P_h
                )
            )
            corners = [top_left, top_right, bottom_left, bottom_right]

            # Reproject each corner to the corresponding face and
            # pair it with a face index. The corner has a tag associated with it.
            for i, corner in enumerate(corners):
                face_idx, converted_corner = (
                    EquirectangularToCubemapConverter._reproject_point(
                        corner, P_w, P_h, self.face_width
                    )
                )
                face_idx = int(face_idx)
                tag = ["TL", "TR", "BL", "BR"][i]
                if face_idx not in face_corners:
                    face_corners[face_idx] = {}
                face_corners[face_idx][tag] = converted_corner

            face_idx_tl = face_idx_br = None

            # Determine the face indices for the top-left and bottom-right corners
            for face_idx, face_corners_dict in face_corners.items():
                for tag in face_corners_dict:
                    if tag == "TL":
                        face_idx_tl = face_idx
                    elif tag == "BR":
                        face_idx_br = face_idx

            # If the bounding box spans multiple faces, process each face separately
            if face_idx_tl != face_idx_br:
                logging.info("Bounding box spans multiple faces!")

                # For each face, we start with a default bounding box with coordinates top-left = (0,0)
                # and bottom-right = (face_width, face_width).
                for face_idx, face_corners_dict in face_corners.items():
                    processed_corners: Dict[str, Tuple[int, int]] = {}

                    # Adjust the bounding box coordinates based on the corners previously reprojected on the face
                    for tag, corner in face_corners_dict.items():
                        processed_corners = EquirectangularToCubemapConverter._adjust_coordinates_based_on_corner(
                            tag, corner, processed_corners
                        )

                    tl_star, br_star = (
                        EquirectangularToCubemapConverter._adjust_bounding_box_corners(
                            processed_corners, self.face_width
                        )
                    )

                    converted_yolo_annotation = (
                        EquirectangularToCubemapConverter._convert_corners_to_yolo(
                            tl_star,
                            br_star,
                            self.face_width,
                            self.face_width,
                        )
                    )

                    final_annotation = (yolo_annotation_class,) + converted_yolo_annotation

                    EquirectangularToCubemapConverter._write_annotation_to_file(
                        self.output_path, img_name, face_idx, final_annotation
                    )
            # If the bounding box is contained within a single face, the final corners are already determined
            else:
                tl_star = face_corners[face_idx_tl]["TL"]
                br_star = face_corners[face_idx_tl]["BR"]

                converted_yolo_annotation = (
                    EquirectangularToCubemapConverter._convert_corners_to_yolo(
                        tl_star,
                        br_star,
                        self.face_width,
                        self.face_width,
                    )
                )

                final_annotation = (yolo_annotation_class,) + converted_yolo_annotation

                EquirectangularToCubemapConverter._write_annotation_to_file(
                    self.output_path, img_name, face_idx, final_annotation
                )

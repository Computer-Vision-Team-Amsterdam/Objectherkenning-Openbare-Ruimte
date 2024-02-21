import os

import cv2
import numpy as np
import py360convert as p3c


def convert_face_idx_to_name(face_idx):
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


def convert_yolo_to_corners(yolo_annotation, P_w, P_h):
    """
    Convert YOLO annotation with normalized values to absolute corner coordinates.

    Parameters
    ----------
    yolo_annotation : str
        YOLO annotation string in the format:
        "<class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>".
    P_w : float
        Width of the image.
    P_h : float
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


def convert_corners_to_yolo(face_name, top_left, bottom_right, face_w, face_h):
    """
    Convert absolute corner coordinates to YOLO annotation with normalized values.

    Parameters
    ----------
    face_name : str
        Name of the face.
    top_left : float
        Top-left absolute corner coordinates.
    bottom_right : float
        Bottom-right absolute corner coordinates.
    face_w : float
        Width of the face.
    face_h : float
        Height of the face.

    Returns
    tuple
        A tuple containing the YOLO annotation in the format:
        "<face_name> <x_center_norm> <y_center_norm> <w_norm> <h_norm>".
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

    return face_name, x_center_norm, y_center_norm, width_norm, height_norm


def reproject_xy_coor_to_face(coor_xy_face, coor, face_w):
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

    return y_proj, x_proj


def reproject_point(point, pano_width, pano_height, face_w):
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

    y_proj, x_proj = reproject_xy_coor_to_face(coor_xy_face, coor, face_w)

    return face_idx, (x_proj, y_proj)


def convert_equirectangular_image_to_cubic(
    input_path, img_path, output_path, face_width
):
    """
    Convert an equirectangular image to a set of six cubic faces using py360convert.e2c.

    Parameters
    ----------
    input_path : str
        Path to the input image directory.
    img_path : str
        Path to the input image file.
    output_path : str
        Path to the output directory.
    face_width : int
        Width of each face in the cubic format.

    Returns
    -------
    None

    """

    img = os.path.join(input_path, img_path)

    try:
        img = cv2.imread(img)
    except FileNotFoundError:
        print(f"File {img} not found")
        return

    front, right, back, left, top, bottom = p3c.e2c(
        img, face_w=face_width, mode="bilinear", cube_format="list"
    )

    folder = img_path.split(".")[0]
    directory = os.path.join(output_path, folder)
    if not os.path.exists(directory):
        os.makedirs(directory)

    cv2.imwrite(f"{directory}/front.png", front)
    cv2.imwrite(f"{directory}/right.png", right)
    cv2.imwrite(f"{directory}/back.png", back)
    cv2.imwrite(f"{directory}/left.png", left)
    cv2.imwrite(f"{directory}/top.png", top)
    cv2.imwrite(f"{directory}/bottom.png", bottom)

    print(f"Processed image: {img_path}.")


def adjust_coordinates_based_on_corner(tag, corner, processed_corners):
    """
    Adjust the coordinates of a bounding box based on the processed corners within a face.

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
        A dictionary maintaining the state of already processed corners. Keys are corner tags
        ('TL', 'TR', 'BL', 'BR'), and values are tuples of (x, y) coordinates. This dictionary
        is updated in-place to reflect the adjusted coordinates based on the current corner's
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


def adjust_bounding_box_corners(processed_corners, face_width):
    """
    Adjust the bounding box corners coordinates based on already processed corners within a face.

    This function finalizes the coordinates of a bounding box by considering the adjustments
    needed based on the corners that have already been processed.

    Parameters
    ----------
    processed_corners : dict
        A dictionary containing the corners that have been processed. Keys are corner tags
        ('TL', 'TR', 'BL', 'BR'), and values are tuples of (x, y) coordinates.

    face_width : int
        The width (and height, assuming square faces) of each face in the cubemap. This is
        used to set default values for bounding box coordinates that are not explicitly
        adjusted based on processed corners.

    Returns
    -------
    tuple
        A tuple containing the adjusted top-left (tl_star) and bottom-right (br_star) coordinates
        of the bounding box, ensuring the box accurately represents the object within the face's
        boundaries.

    Notes
    -----
    The function dynamically adjusts the coordinates of the bounding box depending on which
    corners have been processed. For example, if the 'TL' (top-left) corner has been processed,
    it may adjust the 'x_min' and 'y_min' values of the bounding box. Similarly, if the 'BR'
    (bottom-right) corner has been processed, it may adjust the 'x_max' and 'y_max' values.
    """

    tl_x = 0
    tl_y = 0
    br_x = face_width
    br_y = face_width

    if "TL" in processed_corners:
        tl_x, tl_y = processed_corners["TL"]
    elif "BL" in processed_corners:
        # BL present but not TL, implies adjustment for x position of TL*
        tl_x, _ = processed_corners["BL"]

    if "TR" in processed_corners:
        # TR's y position might adjust TL*'s y if TL is not explicitly processed
        tl_y = max(tl_y, processed_corners["TR"][1])

    if "BR" in processed_corners:
        br_x = processed_corners["BR"][0]
        br_y = processed_corners["BR"][1]
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


def process_annotations(input_path, output_path, img_path, face_width):
    print("=======================================")
    print(f"===== Processing annotations for {img_path} =====")
    print("=======================================")

    annotation_file = img_path.split(".")[0] + ".txt"
    annotations_path = os.path.join(input_path, annotation_file)

    try:
        with open(annotations_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File {annotations_path} not found")
        return

    image_file_path = os.path.join(input_path, img_path)
    image = cv2.imread(image_file_path)
    P_h, P_w, _ = image.shape

    for line in lines:
        face_corners = {}

        yolo_annotation = line.strip()
        yolo_annotation_class = yolo_annotation.split()[0]

        top_left, top_right, bottom_left, bottom_right = convert_yolo_to_corners(
            yolo_annotation, P_w, P_h
        )
        corners = [top_left, top_right, bottom_left, bottom_right]

        for i, corner in enumerate(corners):
            face_idx, converted_corner = reproject_point(corner, P_w, P_h, face_width)
            tag = ["TL", "TR", "BL", "BR"][i]
            if face_idx not in face_corners:
                face_corners[face_idx] = {}
            face_corners[face_idx][tag] = converted_corner

        face_idx_tl = face_idx_br = None

        for face_idx, corners in face_corners.items():
            for tag in corners:
                if tag == "TL":
                    face_idx_tl = face_idx
                elif tag == "BR":
                    face_idx_br = face_idx

        if face_idx_tl != face_idx_br:
            print("=======! Bounding box spans multiple faces !=======")

            for face_idx, corners in face_corners.items():
                processed_corners = {}

                for tag, corner in corners.items():
                    processed_corners = adjust_coordinates_based_on_corner(
                        tag, corner, processed_corners
                    )

                tl_star, br_star = adjust_bounding_box_corners(
                    processed_corners, face_width
                )

                converted_yolo_annotation = convert_corners_to_yolo(
                    face_idx, tl_star, br_star, face_width, face_width
                )

                converted_yolo_annotation = (
                    yolo_annotation_class,
                    *converted_yolo_annotation[1:],
                )

                face_annotation_file = os.path.join(
                    output_path,
                    img_path.split(".")[0],
                    f"{convert_face_idx_to_name(face_idx)}.txt",
                )

                if not os.path.exists(face_annotation_file):
                    os.makedirs(os.path.dirname(face_annotation_file), exist_ok=True)
                    open(face_annotation_file, "w").close()

                with open(face_annotation_file, "a") as f:
                    f.write(" ".join(map(str, converted_yolo_annotation)) + "\n")

                print(f"==== Wrote to {face_annotation_file} ====")
                print("=======================================")

        else:

            tl_star = face_corners[face_idx_tl]["TL"]
            br_star = face_corners[face_idx_tl]["BR"]

            converted_yolo_annotation = convert_corners_to_yolo(
                yolo_annotation_class,
                tl_star,
                br_star,
                face_width,
                face_width,
            )

            # Before writing to the face annotation file, check if it needs to be cleared
            face_annotation_file = os.path.join(
                output_path,
                img_path.split(".")[0],
                f"{convert_face_idx_to_name(face_idx_tl)}.txt",
            )

            # This is where we check if the file exists and clear it if necessary
            if not os.path.exists(face_annotation_file):
                os.makedirs(os.path.dirname(face_annotation_file), exist_ok=True)
                open(face_annotation_file, "w").close()  # This clears the file

            with open(face_annotation_file, "a") as f:
                f.write(" ".join(map(str, converted_yolo_annotation)) + "\n")
            print(f"== Wrote to {face_annotation_file} ==")


def visualize_annotations_on_equirectangular_image(input_path, img_path):
    """
    Visualize YOLO annotations on the original equirectangular image.

    Parameters:
    - input_path: Base input directory where the original images are stored.
    - img_path: Path to the specific original image file.
    - P_w, P_h: Width and height of the original equirectangular image.
    """
    print(f"Visualizing annotations for equirectangular image: {img_path}...")
    # Construct the path to the image and its annotation file
    image_file_path = os.path.join(input_path, img_path)
    annotation_file_path = os.path.join(input_path, img_path.split(".")[0] + ".txt")

    # Load the image
    image = cv2.imread(image_file_path)
    if image is None:
        print(f"Image file {image_file_path} not found.")
        return

    # Check if the annotation file exists
    if not os.path.exists(annotation_file_path):
        print(f"Annotation file {annotation_file_path} not found.")
        return

    P_w = image.shape[1]
    P_h = image.shape[0]

    print(f"Size of equirectangular image: {P_w} x {P_h} pixels.")

    # Read the annotations and plot each bounding box on the image
    with open(annotation_file_path, "r") as file:
        for line in file:
            _, x_center_norm, y_center_norm, w_norm, h_norm = map(float, line.split())
            print(
                f"Normalized Annotation: \n X Center: {x_center_norm}, Y Center: {y_center_norm}, Width: {w_norm}, Height: {h_norm}"
            )

            # Convert normalized coordinates to absolute pixel values
            x_center_abs = int(x_center_norm * P_w)
            y_center_abs = int(y_center_norm * P_h)
            width_abs = int(w_norm * P_w)
            height_abs = int(h_norm * P_h)

            print(
                f"Absolute coordinates: \n X Center: {x_center_abs}, Y Center: {y_center_abs}, Width: {width_abs}, Height: {height_abs}"
            )

            # Calculate corners of the bounding box
            x_min = int(x_center_abs - width_abs / 2)
            y_min = int(y_center_abs - height_abs / 2)
            x_max = int(x_center_abs + width_abs / 2)
            y_max = int(y_center_abs + height_abs / 2)

            print(f"Bounding box corners: ({x_min}, {y_min}), ({x_max}, {y_max})")

            # Draw bounding box on the image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

    # Display the image with annotations
    # cv2.imshow(f"Annotated Equirectangular Image: {img_path}", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Optionally, save the visualized image to disk
    visualized_img_path = os.path.join(
        input_path, img_path.split(".")[0] + "_annotated.png"
    )
    cv2.imwrite(visualized_img_path, image)
    print(f"Annotated equirectangular image saved to {visualized_img_path}")
    print("=======================================")


def visualize_annotations_with_corners(output_path, img_folder, face_name):
    """
    Visualize annotations on the specified face image with a red dot for each corner of the bounding boxes.

    Parameters:
    - output_path: The base output directory where the images and annotations are stored.
    - img_folder: The folder name derived from the original image name, used to locate the specific image and its annotations.
    - face_name: The name of the face (e.g., 'back') to visualize annotations for.
    """
    # Construct the paths to the image and its annotation file
    img_path = os.path.join(output_path, img_folder, f"{face_name}.png")
    annotation_path = os.path.join(output_path, img_folder, f"{face_name}.txt")

    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image file {img_path} not found.")
        return

    # Check if the annotation file exists
    if not os.path.exists(annotation_path):
        print(f"Annotation file {annotation_path} not found.")
        return

    img_copy = img.copy()

    # Read the annotations and plot each bounding box on the image
    with open(annotation_path, "r") as file:
        for line in file:
            # Parse the YOLO formatted annotation
            _, x_center_norm, y_center_norm, w_norm, h_norm = map(float, line.split())

            print(
                f"{face_name} annotation: {x_center_norm}, {y_center_norm}, {w_norm}, {h_norm}"
            )

            # Convert normalized coordinates to absolute pixel values
            x_center_abs = x_center_norm * img.shape[1]
            y_center_abs = y_center_norm * img.shape[0]
            width_abs = w_norm * img.shape[1]
            height_abs = h_norm * img.shape[0]

            print(
                f"{face_name} annotation absolute coordinates of the center: {x_center_abs}, {y_center_abs}, {width_abs}, {height_abs}"
            )

            # Calculate the corners of the bounding box
            x_min = int(x_center_abs - (width_abs / 2))
            y_min = int(y_center_abs - (height_abs / 2))
            x_max = int(x_center_abs + (width_abs / 2))
            y_max = int(y_center_abs + (height_abs / 2))

            print(
                f"{face_name} bounding box corners: ({x_min}, {y_min}), ({x_max}, {y_max})"
            )

            # Draw bounding box on the image
            cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

    # Display the image with annotations
    # cv2.imshow(f"Annotated {face_name}", img)
    # cv2.waitKey(0)  # Wait for a key press to close the image window
    # cv2.destroyAllWindows()

    # Optionally, save the visualized image to disk
    visualized_img_path = os.path.join(
        output_path, img_folder, f"{face_name}_annotated_corners.png"
    )
    cv2.imwrite(visualized_img_path, img_copy)
    print(f"Annotated image with corners saved to {visualized_img_path}")


def main():
    # Step 1: set the paths (for now, hardcode)
    # Set the path to the images
    input_path = "../../input"
    output_path = "../../output"

    # TODO: Remember to not hardcode the dimensions of the equirectangular image
    face_width = 1024  # Example dimensions of the cubemap faces

    # Initialize counters
    total_images = 0
    processed_folders = 0
    valid_extensions = {
        ".jpg",
        ".png",
    }  # Consider only these file extensions as valid images

    # Count total images in input directory
    for img_path in os.listdir(input_path):
        if img_path.endswith(tuple(valid_extensions)):
            total_images += 1

    # Count folders in output directory
    # if os.path.exists(output_path):
    #    for item in os.listdir(output_path):
    #        if os.path.isdir(os.path.join(output_path, item)):
    #            processed_folders += 1

    remaining_to_process = total_images - processed_folders

    print(f"Total images in input: {total_images}")
    print(f"Processed images in output: {processed_folders}")
    print(f"Remaining to process: {remaining_to_process}")

    # Step 2: convert the images (ignore .DS_Store files)
    for img_path in os.listdir(input_path):
        if img_path != ".DS_Store" and not img_path.endswith("_annotated.png"):
            if img_path.endswith(".jpg") or img_path.endswith(".png"):
                # Check if the image is empty or corrupted
                img = cv2.imread(os.path.join(input_path, img_path))
                if img is None:
                    print(f"Skipping {img_path} as it is empty or corrupted")
                    continue

                # Check if output folder for this image already exists
                # folder_name = img_path.split(".")[
                #    0
                # ]  # Assuming the folder name is derived from the image name
                # output_folder = os.path.join(output_path, folder_name)
                # if os.path.exists(output_folder):
                #    continue  # Skip this image and go to the next one

                convert_equirectangular_image_to_cubic(
                    input_path, img_path, output_path, face_width
                )

                # Step 2.1: visualize annotations on the equirectangular image
                # visualize_annotations_on_equirectangular_image(input_path, img_path)

                # Step 3: convert the annotations from yolo to xy coordinates
                process_annotations(input_path, output_path, img_path, face_width)

                # Step 4: visualize processed annotations
                # Example call to visualize annotations for the 'back' face
                # Assuming 'img_path' is available and corresponds to the processed image
                # img_folder = img_path.split(".")[
                #    0
                # ]  # Extract the folder name from the image path
                # visualize_annotations_with_corners(output_path, img_folder, "right")
                # visualize_annotations_with_corners(output_path, img_folder, "left")
                # visualize_annotations_with_corners(output_path, img_folder, "front")
                # visualize_annotations_with_corners(output_path, img_folder, "back")
                # visualize_annotations_with_corners(output_path, img_folder, "bottom")
                # visualize_annotations_with_corners(output_path, img_folder, "top")

                remaining_to_process -= 1
                print(f"Remaining to process: {remaining_to_process}")


if __name__ == "__main__":
    main()

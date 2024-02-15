import os

import cv2
import numpy as np
import py360convert as p3c


def convert_face_idx_to_name(face_idx):
    """
    0F 1R 2B 3L 4U 5D
    """
    face_names = ["front", "right", "back", "left", "top", "bottom"]
    return face_names[face_idx]


def convert_yolo_to_corners(yolo_annotation, P_w, P_h):
    # Parse YOLO annotation
    _, x_center_norm, y_center_norm, w_norm, h_norm = map(
        float, yolo_annotation.split()
    )

    # Convert normalized values to absolute coordinates
    x_center_abs = x_center_norm * P_w
    y_center_abs = y_center_norm * P_h
    w_abs = w_norm * P_w
    h_abs = h_norm * P_h

    # Calculate top-left and bottom-right corners
    x_min = x_center_abs - (w_abs / 2)
    y_min = y_center_abs - (h_abs / 2)
    x_max = x_center_abs + (w_abs / 2)
    y_max = y_center_abs + (h_abs / 2)

    # Calculate top-right and bottom-left corners
    top_left = (x_min, y_min)
    top_right = (x_max, y_min)
    bottom_left = (x_min, y_max)
    bottom_right = (x_max, y_max)

    return top_left, top_right, bottom_left, bottom_right


def corners_to_yolo(face_name, top_left, bottom_right, face_w, face_h):
    # Calculate the absolute coordinates and size of the bounding box
    x_min, y_min = top_left
    x_max, y_max = bottom_right

    # Calculate center coordinates and size in absolute terms
    x_center_abs = (x_min + x_max) / 2
    y_center_abs = (y_min + y_max) / 2
    width_abs = x_max - x_min
    height_abs = y_max - y_min

    # Normalize these values with respect to the face dimensions
    x_center_norm = x_center_abs / face_w
    y_center_norm = y_center_abs / face_h
    width_norm = width_abs / face_w
    height_norm = height_abs / face_h

    # Return the YOLO format annotation
    return face_name, x_center_norm, y_center_norm, width_norm, height_norm


def reproject_point(x, y, pano_width, pano_height, face_w=256):
    # Step 1: Convert (y, x) to UV coordinates
    coor = np.array([x, y])
    # uv = p3c.utils.coor2uv(coor, pano_height, pano_width)

    # Step 2: Convert UV coordinates to XYZ coordinates
    # xyz = p3c.utils.uv2unitxyz(uv)

    # Step 3: Determine the face and coordinates within the face
    face_type = p3c.utils.equirect_facetype(pano_height, pano_width)

    # Convert x and y to integers before using them as indices
    x_int = int(x)
    y_int = int(y)

    # Ensure the indices are within the valid range
    x_int = max(0, min(x_int, pano_width - 1))
    y_int = max(0, min(y_int, pano_height - 1))

    face_idx = face_type[y_int, x_int]

    cube_faces_xyz = p3c.utils.xyzcube(face_w)
    face_xyz = cube_faces_xyz[:, face_idx * face_w : (face_idx + 1) * face_w, :]

    uv_face = p3c.utils.xyz2uv(face_xyz)
    coor_xy_face = p3c.utils.uv2coor(uv_face, pano_height, pano_width)

    y_proj, x_proj = np.unravel_index(
        np.argmin(np.linalg.norm(coor_xy_face - coor, axis=-1)), (face_w, face_w)
    )

    # Step 4: Return the face index and the new coordinates
    return face_idx, (x_proj, y_proj)


def convert_xy_to_cubemap_coordinates(point, pano_width, pano_height, face_w=256):
    # Unpack the point
    x, y = point

    # Reproject the chosen point (e.g., center or top-left corner)
    # You might need to adjust the reproject_point function or ensure it can accept these values directly
    face_idx, (x_proj, y_proj) = reproject_point(x, y, pano_width, pano_height, face_w)

    return face_idx, (x_proj, y_proj)


def convert_image_to_cubic(input_path, img_path, output_path, face_width):

    print(f"Processing image {img_path}.")
    img = os.path.join(input_path, img_path)

    # Open and transform to a numpy array with shape [H, W, 3] using cv2 (convert to RGB)
    try:
        img = cv2.imread(img)
    except FileNotFoundError:
        print(f"File {img} not found")
        return

    # Project from equirectangular to cubic
    front, right, back, left, top, bottom = p3c.e2c(
        img, face_w=face_width, mode="bilinear", cube_format="list"
    )

    # make directory, with panoid as name, to save them in
    folder = img_path.split(".")[0]
    directory = os.path.join(output_path, folder)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save cubemap faces
    cv2.imwrite(f"{directory}/front.png", front)
    cv2.imwrite(f"{directory}/right.png", right)
    cv2.imwrite(f"{directory}/back.png", back)
    cv2.imwrite(f"{directory}/left.png", left)
    cv2.imwrite(f"{directory}/top.png", top)
    cv2.imwrite(f"{directory}/bottom.png", bottom)

    print("=======================================")


def adjust_coordinates_based_on_corner(
    tag, corner, processed_corners, face_width, face_height
):
    # Check if other corners have been processed and adjust coordinates accordingly
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

    # After considering other corners, update the current corner's coordinates
    processed_corners[tag] = corner

    return processed_corners


def compute_new_coordinates_for_single_corner(
    corner, corner_type, face_width, face_height
):
    """
    Computes new bounding box coordinates based on the position of a single corner within a face.
    """
    x, y = corner

    if corner_type == "TL":
        return corner, (face_width, face_height)
    elif corner_type == "TR":
        return (0, y), (x, face_height)
    elif corner_type == "BL":
        return (x, 0), (0, y)
    elif corner_type == "BR":
        return (0, 0), corner
    else:
        raise ValueError("Invalid corner type")


def process_annotations(input_path, output_path, img_path, P_w, P_h, face_width):
    annotation_file = img_path.split(".")[0] + ".txt"
    print("=======================================")
    print(f"===== Processing annotations for {img_path} =====")
    print("=======================================")
    # print(f'Input path: {input_path}')
    annotations_path = os.path.join(input_path, annotation_file)
    # print(f'Annotations path: {annotations_path}')

    try:
        with open(annotations_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File {annotations_path} not found")
        return

    for line in lines:
        print(f"===== Processing annotation: {line} =====")

        # Aggregate corners by face
        face_corners = {}

        yolo_annotation = line.strip()  # Remove any extra whitespace
        top_left, top_right, bottom_left, bottom_right = convert_yolo_to_corners(
            yolo_annotation, P_w, P_h
        )
        corners = [top_left, top_right, bottom_left, bottom_right]
        for i, corner in enumerate(corners):
            face_idx, converted_corner = convert_xy_to_cubemap_coordinates(
                corner, P_w, P_h, face_width
            )
            tag = ["TL", "TR", "BL", "BR"][i]
            if face_idx not in face_corners:
                face_corners[face_idx] = {}
            face_corners[face_idx][tag] = converted_corner
            # print(f'Face {face_idx} corner {tag}: {converted_corner}')

        print(f"Original annotation: {yolo_annotation}")
        print(
            f"Original corners: {top_left}, {top_right}, {bottom_left}, {bottom_right}"
        )
        for face_idx, corners in face_corners.items():
            for tag, corner in corners.items():
                print(f"Converted corner {tag}: {corner}")

        # Case 1: the bounding box spans multiple faces

        # Find the index of the TL corner using face_corners dictionary
        face_idx_tl, _ = convert_xy_to_cubemap_coordinates(
            top_left, P_w, P_h, face_width
        )
        face_idx_br, _ = convert_xy_to_cubemap_coordinates(
            bottom_right, P_w, P_h, face_width
        )
        face_idx_tr, _ = convert_xy_to_cubemap_coordinates(
            top_right, P_w, P_h, face_width
        )
        face_idx_bl, _ = convert_xy_to_cubemap_coordinates(
            bottom_left, P_w, P_h, face_width
        )

        if face_idx_tl != face_idx_br:
            print("=======! Bounding box spans multiple faces !=======")
            face_name_tl = convert_face_idx_to_name(face_idx_tl)
            face_name_tr = convert_face_idx_to_name(face_idx_tr)
            face_name_bl = convert_face_idx_to_name(face_idx_bl)
            face_name_br = convert_face_idx_to_name(face_idx_br)
            print(
                f"Top-left corner: {face_name_tl}. Top-right corner: {face_name_tr}. \n"
                f"Bottom-left corner: {face_name_bl}. Bottom-right corner: {face_name_br}."
            )

            # Process each face
            for face_idx, corners in face_corners.items():
                print(
                    f"====== Processing face {convert_face_idx_to_name(face_idx)} ======"
                )
                processed_corners = {}  # Initialize dictionary for adjusted coordinates

                for tag, corner in corners.items():
                    # Adjust coordinates based on already processed corners for the face
                    processed_corners = adjust_coordinates_based_on_corner(
                        tag, corner, processed_corners, face_width, face_width
                    )

                print(f"Processed corners: {processed_corners}")

                # Finalize and convert adjusted bounding box to YOLO format
                tl_x = 0  # Default to 0, assuming no TL or BL gives a better x position
                tl_y = 0  # Default to 0, assuming no TL or TR gives a better y position
                br_x = face_width  # Default value in case BR or TR is not specifically adjusted
                br_y = face_width  # Default value in case BR or BL is not specifically adjusted

                # Apply adjustments from processed_corners for top-left coordinates
                if "TL" in processed_corners:
                    tl_x = processed_corners["TL"][0]
                    tl_y = processed_corners["TL"][1]
                elif "BL" in processed_corners:
                    # BL present but not TL, implies adjustment for x position of TL*
                    tl_x = processed_corners["BL"][0]

                if "TR" in processed_corners:
                    # TR's y position might adjust TL*'s y if TL is not explicitly processed
                    tl_y = max(tl_y, processed_corners["TR"][1])

                # Apply adjustments from processed_corners for bottom-right coordinates
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

                # Now we have tl_star and br_star considering adjustments
                tl_star = (tl_x, tl_y)
                br_star = (br_x, br_y)

                print(f"Adjusted bounding box corners: {tl_star}, {br_star}")

                converted_yolo_annotation = corners_to_yolo(
                    face_idx, tl_star, br_star, face_width, face_width
                )

                print(f"Converted YOLO annotation: {converted_yolo_annotation}")

                # Before writing to the face annotation file, check if it needs to be cleared
                face_annotation_file = os.path.join(
                    output_path,
                    img_path.split(".")[0],
                    f"{convert_face_idx_to_name(face_idx)}.txt",
                )

                # This is where we check if the file exists and clear it if necessary
                if not os.path.exists(face_annotation_file):
                    os.makedirs(os.path.dirname(face_annotation_file), exist_ok=True)
                    open(face_annotation_file, "w").close()  # This clears the file

                with open(face_annotation_file, "a") as f:
                    f.write(" ".join(map(str, converted_yolo_annotation)) + "\n")
                print(f"==== Wrote to {face_annotation_file} ====")
                print("=======================================")

        else:
            print(
                f"Bounding box is contained within a single face: {convert_face_idx_to_name(face_idx_tl)}"
            )
            tl_star = face_corners[face_idx_tl]["TL"]
            br_star = face_corners[face_idx_tl]["BR"]

            # Convert to YOLO format
            converted_yolo_annotation = corners_to_yolo(
                face_idx_tl,  # Change this
                tl_star,
                br_star,
                face_width,
                face_width,
            )

            # Swap the face name in converted_yolo_annotation with the first element of the original yolo_annotation (the class)
            converted_yolo_annotation = (
                yolo_annotation.split()[0],
                *converted_yolo_annotation[1:],
            )

            # Before writing to the face annotation file, check if it needs to be cleared
            face_annotation_file = os.path.join(
                output_path,
                img_path.split(".")[0],
                f"{convert_face_idx_to_name(face_idx_tl)}.txt",  # CHANGE THIS
            )

            # This is where we check if the file exists and clear it if necessary
            if not os.path.exists(face_annotation_file):
                os.makedirs(os.path.dirname(face_annotation_file), exist_ok=True)
                open(face_annotation_file, "w").close()  # This clears the file

            with open(face_annotation_file, "a") as f:
                f.write(" ".join(map(str, converted_yolo_annotation)) + "\n")
            print(f"== Wrote to {face_annotation_file} ==")


def visualize_annotations_on_equirectangular_image(input_path, img_path, P_w, P_h):
    """
    Visualize YOLO annotations on the original equirectangular image.

    Parameters:
    - input_path: Base input directory where the original images are stored.
    - img_path: Path to the specific original image file.
    - P_w, P_h: Width and height of the original equirectangular image.
    """
    print(f"Visualizing annotations for equirectangular image: {img_path}...")
    print(f"Size of equirectangular image: {P_w} x {P_h} pixels.")
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
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 8)

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
            cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 8)

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
    P_w, P_h = 7040, 3520  # Example dimensions of the equirectangular image
    face_width = 1024  # Example dimensions of the cubemap faces

    # Step 2: convert the images (ignore .DS_Store files)
    for img_path in os.listdir(input_path):
        if img_path != ".DS_Store" and not img_path.endswith("_annotated.png"):
            if img_path.endswith(".jpg") or img_path.endswith(".png"):
                convert_image_to_cubic(input_path, img_path, output_path, face_width)

                # Step 2.1: visualize annotations on the equirectangular image
                visualize_annotations_on_equirectangular_image(
                    input_path, img_path, P_w, P_h
                )

                # Step 3: convert the annotations from yolo to xy coordinates
                process_annotations(
                    input_path, output_path, img_path, P_w, P_h, face_width
                )

                # Step 4: visualize processed annotations
                # Example call to visualize annotations for the 'back' face
                # Assuming 'img_path' is available and corresponds to the processed image
                img_folder = img_path.split(".")[
                    0
                ]  # Extract the folder name from the image path
                visualize_annotations_with_corners(output_path, img_folder, "right")
                visualize_annotations_with_corners(output_path, img_folder, "back")
                visualize_annotations_with_corners(output_path, img_folder, "bottom")
                visualize_annotations_with_corners(output_path, img_folder, "top")


if __name__ == "__main__":
    main()

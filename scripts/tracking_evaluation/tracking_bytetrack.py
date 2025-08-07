import argparse
import json
import os

import cv2
import pandas as pd
import supervision as sv

# Define colors for annotation
CLASS_COLORS = {
    3: (0, 255, 0),  # container (green)
    4: (0, 165, 255),  # mobile toilet (orange)
    5: (255, 0, 0),  # scaffolding (blue)
}


def load_annotations(ann_path, allowed_categories):
    coco = json.load(open(ann_path, "r"))
    img_meta = {img["id"]: img for img in coco["images"]}
    records = []
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid not in allowed_categories:
            continue
        img = img_meta[ann["image_id"]]
        w, h = img["width"], img["height"]
        x, y, bw, bh = ann["bbox"]
        px, py = int(x * w), int(y * h)
        pw, ph = int(bw * w), int(bh * h)
        records.append(
            {
                "image_id": ann["image_id"],
                "file_name": img["file_name"],
                "x1": px,
                "y1": py,
                "x2": px + pw,
                "y2": py + ph,
                "score": ann.get("score", 1.0),
                "category_id": cid,
                "annotation_id": ann["id"],
            }
        )
    return pd.DataFrame(records)


def run_tracking(
    data_dir,
    ann_path,
    output_csv,
    max_age,
    annotate=False,
    annotate_dir="annotated_frames",
):
    TRACK_CATS = {3: "container", 4: "mobile_toilet", 5: "scaffolding"}

    df = load_annotations(ann_path, TRACK_CATS)
    df = df.sort_values("file_name").reset_index(drop=True)

    # Initialize ByteTrack tracker
    tracker = sv.ByteTrack(lost_track_buffer=max_age)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    outputs = []

    if annotate:
        annotate_path = os.path.join(annotate_dir, f"MA{max_age}")
        os.makedirs(annotate_path, exist_ok=True)

    for fn, group in df.groupby("file_name"):
        frame_path = os.path.join(data_dir, os.path.basename(fn))
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: couldn't load {frame_path}")
            continue

        # Prepare detections for supervision
        xyxy = group[["x1", "y1", "x2", "y2"]].to_numpy()
        confidences = group["score"].to_numpy()
        class_ids = group["category_id"].to_numpy()
        annotation_ids = group["annotation_id"].to_numpy()
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids,
            data={"annotation_id": annotation_ids},
        )

        # Update tracker with detections
        tracked = tracker.update_with_detections(detections)

        ann_ids = []
        if hasattr(tracked, "data") and "annotation_id" in tracked.data:
            ann_ids = tracked.data["annotation_id"]

        if annotate:
            vis = frame.copy()
            # Annotate boxes and labels
            labels = [
                f"#{tid} {cid}"
                for tid, cid in zip(tracked.tracker_id, tracked.class_id)
            ]
            vis = box_annotator.annotate(vis, detections=tracked)
            vis = label_annotator.annotate(vis, detections=tracked, labels=labels)

        # Collect outputs
        for bbox, tid, cid, aid in zip(
            tracked.xyxy, tracked.tracker_id, tracked.class_id, ann_ids
        ):
            x1, y1, x2, y2 = map(int, bbox)
            outputs.append(
                {
                    "image_name": fn,
                    "ID": int(tid),
                    "object_category": int(cid),
                    "annotation_id": int(aid),
                }
            )

        if annotate:
            cv2.imwrite(os.path.join(annotate_path, os.path.basename(fn)), vis)

    pd.DataFrame(outputs).to_csv(output_csv, index=False)
    print(f"Wrote {len(outputs)} rows to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ByteTrack tracking on frame-by-frame detections."
    )
    parser.add_argument("--data_dir", required=True, help="Folder with frames")
    parser.add_argument("--annotations", required=True, help="COCO JSON file")
    parser.add_argument(
        "--out_dir", required=True, help="Directory or file path for CSV output"
    )
    parser.add_argument("--annotate", action="store_true", help="Save annotated frames")
    parser.add_argument(
        "--annotate_dir",
        default="annotated_frames",
        help="Directory to save annotated frames",
    )
    parser.add_argument("--max_age", type=int, default=30, help="Max track age")

    args = parser.parse_args()

    dataset_name = os.path.basename(os.path.normpath(args.data_dir))

    if os.path.isdir(args.out_dir):
        fname = f"{dataset_name}_tracks_ma{args.max_age}_bytetrack.csv"
        output_path = os.path.join(args.out_dir, fname)
    else:
        output_path = args.out_dir

    run_tracking(
        data_dir=args.data_dir,
        ann_path=args.annotations,
        output_csv=output_path,
        max_age=args.max_age,
        annotate=args.annotate,
        annotate_dir=args.annotate_dir,
    )

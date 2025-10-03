import argparse
import json
import os

import cv2
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm

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
    max_cosine_distance,
    n_init,
    annotate=False,
    annotate_dir="annotated_frames",
):
    TRACK_CATS = {3: "container", 4: "mobile_toilet", 5: "scaffolding"}

    df = load_annotations(ann_path, TRACK_CATS)
    df = df.sort_values("file_name").reset_index(drop=True)

    tracker = DeepSort(
        max_age=max_age,
        max_cosine_distance=max_cosine_distance,
        n_init=n_init,
        nn_budget=100,
        embedder="mobilenet",
    )

    outputs = []

    # Prepare annotation directory if requested
    if annotate:
        annotate_path = os.path.join(
            annotate_dir, f"MA{max_age}_CD{int(max_cosine_distance * 100)}_NI{n_init}"
        )
        os.makedirs(annotate_path, exist_ok=True)

    loop_desc = (
        f"MA{max_age}_CD{max_cosine_distance}_NI{n_init}"
        if not annotate
        else os.path.basename(annotate_path)
    )
    for fn, group in tqdm(df.groupby("file_name"), desc=loop_desc):
        frame_path = os.path.join(data_dir, os.path.basename(fn))
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: couldn't load {frame_path}")
            continue

        # Prepare detections
        raw_detections = []
        for _, row in group.iterrows():
            x1, y1, x2, y2 = row[["x1", "y1", "x2", "y2"]]
            w, h = x2 - x1, y2 - y1
            raw_detections.append(([x1, y1, w, h], row["score"], row["category_id"]))

        tracks = tracker.update_tracks(
            raw_detections, frame=frame, others=group["annotation_id"].tolist()
        )

        if annotate:
            vis = frame.copy()

        for trk in tracks:
            if (
                not trk.is_confirmed()
                or trk.time_since_update != 0
                or trk.others is None
            ):
                continue

            aid = -1
            if trk.others is not None:
                aid = int(trk.others)

            x1, y1, w, h = map(int, trk.to_tlwh())
            x2, y2 = x1 + w, y1 + h
            cls = trk.det_class
            tid = trk.track_id

            if annotate:
                color = CLASS_COLORS.get(cls, (255, 255, 255))
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    vis,
                    f"{cls}: {tid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            outputs.append(
                {
                    "image_name": fn,
                    "ID": tid,
                    "object_category": int(cls),
                    "annotation_id": aid,
                }
            )

        if annotate:
            cv2.imwrite(os.path.join(annotate_path, os.path.basename(fn)), vis)

    pd.DataFrame(outputs).to_csv(output_csv, index=False)
    print(f"Wrote {len(outputs)} rows to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Deep SORT tracking with optional grid search and dataset naming."
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

    # Single-run parameters
    parser.add_argument(
        "--max_age", type=int, default=30, help="Max track age (single run)"
    )
    parser.add_argument(
        "--max_cosine_distance",
        type=float,
        default=0.4,
        help="Max cosine distance (single run)",
    )
    parser.add_argument(
        "--n_init", type=int, default=3, help="Number of initial frames (single run)"
    )

    # Grid-search flag
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Enable grid search over default parameter sets",
    )

    args = parser.parse_args()

    dataset_name = os.path.basename(os.path.normpath(args.data_dir))

    if args.grid:
        max_ages = [10, 30, 50]
        max_cosine_distances = [0.2, 0.4, 0.6]
        n_inits = [1, 3, 5]

        for ma in max_ages:
            for mcd in max_cosine_distances:
                for ni in n_inits:
                    fname = (
                        f"{dataset_name}_tracks_ma{ma}_cd{int(mcd * 100)}_ni{ni}.csv"
                    )
                    out_csv = os.path.join(args.out_dir, fname)
                    run_tracking(
                        data_dir=args.data_dir,
                        ann_path=args.annotations,
                        output_csv=out_csv,
                        max_age=ma,
                        max_cosine_distance=mcd,
                        n_init=ni,
                        annotate=args.annotate,
                        annotate_dir=args.annotate_dir,
                    )
    else:
        # Single run
        if os.path.isdir(args.out_dir):
            fname = f"{dataset_name}_tracks_ma{args.max_age}_cd{int(args.max_cosine_distance * 100)}_ni{args.n_init}_deepsort.csv"
            output_path = os.path.join(args.out_dir, fname)
        else:
            output_path = args.out_dir

        run_tracking(
            data_dir=args.data_dir,
            ann_path=args.annotations,
            output_csv=output_path,
            max_age=args.max_age,
            max_cosine_distance=args.max_cosine_distance,
            n_init=args.n_init,
            annotate=args.annotate,
            annotate_dir=args.annotate_dir,
        )

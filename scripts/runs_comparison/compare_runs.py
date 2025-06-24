import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

import pandas as pd


def setup_logging() -> None:
    """Configure logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth (in meters).
    """
    R = 6_371_000  # Earth radius in meters
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def load_manual_frame_metadata(frame_csv_path: Path) -> pd.DataFrame:
    """
    Load manual frame metadata CSV and index by image_name.
    """
    df = pd.read_csv(frame_csv_path, dtype={"image_name": str})
    required = {"gps_lat", "gps_lon"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required} in {frame_csv_path}")
    return df.set_index("image_name")


def load_manual_detections(detection_csv_path: Path) -> pd.DataFrame:
    """
    Load manual detections CSV for image_name/object_class pairs.
    """
    df = pd.read_csv(detection_csv_path, dtype={"image_name": str})
    required = {"image_name", "object_class"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required} in {detection_csv_path}")
    return df


def load_all_auto_jsons(auto_detection_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all JSON files from automatic detection directory.
    """
    records = []
    for jpath in auto_detection_dir.glob("*.json"):
        rec = json.loads(jpath.read_text())
        required = {"image_file_name", "gps_data", "detections"}
        if not required.issubset(rec.keys()):
            logging.warning("Skipping JSON %s: missing keys %r", jpath, rec.keys())
            continue
        records.append(rec)
    return records


def summarize_auto_detections(auto_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize auto detections: count and set of classes.
    """
    dets = auto_record.get("detections", [])
    classes = {d.get("object_class") for d in dets if "object_class" in d}
    return {"count": len(dets), "classes": classes}


def find_best_match(
    frames: pd.DataFrame,
    manual_detections: pd.DataFrame,
    auto_records: List[Dict[str, Any]],
    image_files: Set[str],
    filter_fn: Callable[[Dict[str, Any], Set[Any], Dict[str, Any]], bool],
) -> pd.DataFrame:
    """
    For each manual frame, find the best-matching auto record according to filter_fn,
    returning a DataFrame of comparison results.
    """
    rows = []
    for img_name, frame in frames.iterrows():
        lat, lon = float(frame["gps_lat"]), float(frame["gps_lon"])
        man_sub = manual_detections[manual_detections["image_name"] == img_name]
        man_classes = set(man_sub["object_class"]) if not man_sub.empty else set()

        candidates = []
        for rec in auto_records:
            auto_sum = summarize_auto_detections(rec)
            if filter_fn(rec, man_classes, auto_sum):
                d = haversine(
                    lat, lon, rec["gps_data"]["latitude"], rec["gps_data"]["longitude"]
                )
                candidates.append((rec, auto_sum, d))
        if not candidates:
            continue

        best_rec, best_sum, best_dist = min(candidates, key=lambda x: x[2])
        rows.append(
            {
                "manual_image": img_name,
                "manual_lat": lat,
                "manual_lon": lon,
                "matched_auto_image": best_rec["image_file_name"],
                "auto_lat": best_rec["gps_data"]["latitude"],
                "auto_lon": best_rec["gps_data"]["longitude"],
                "distance_m": round(best_dist, 1),
                "manual_obj_count": len(man_sub),
                "auto_obj_count": best_sum["count"],
                # Ensure classes are strings before joining
                "manual_classes": ";".join(sorted(map(str, man_classes))),
                "auto_classes": ";".join(sorted(map(str, best_sum["classes"]))),
                "image_available": img_name in image_files,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Compare manual vs. automatic detections."
    )
    parser.add_argument(
        "--manual-dir",
        type=Path,
        required=True,
        help="Path to the manual_run_YYYY-MM-DD directory",
    )
    parser.add_argument(
        "--auto-dir",
        type=Path,
        required=True,
        help="Path to the automatic_run_YYYY-MM-DD directory",
    )
    args = parser.parse_args()

    manual_dir = args.manual_dir
    auto_dir = args.auto_dir
    base_dir = Path(__file__).parent

    logging.info(
        "Loading manual frame metadata from %s",
        manual_dir / "frame_metadata/frame_metadata.csv",
    )
    frames = load_manual_frame_metadata(
        manual_dir / "frame_metadata/frame_metadata.csv"
    )
    logging.info(
        "Loading manual detections from %s",
        manual_dir / "detection_metadata/detection_metadata.csv",
    )
    manual_det = load_manual_detections(
        manual_dir / "detection_metadata/detection_metadata.csv"
    )
    logging.info("Loading auto JSONs from %s", auto_dir / "detection_metadata")
    auto_recs = load_all_auto_jsons(auto_dir / "detection_metadata")

    image_files = {p.name for p in (manual_dir / "images").glob("*") if p.is_file()}

    # Distance-based report (no class filtering)
    logging.info("Building distance-based report...")
    df_dist = find_best_match(
        frames=frames,
        manual_detections=manual_det,
        auto_records=auto_recs,
        image_files=image_files,
        filter_fn=lambda rec, mc, auto_sum: True,
    )
    out_dist = base_dir / "comparison_report_distance.csv"
    df_dist.to_csv(out_dist, index=False)
    logging.info("Wrote distance-based report to %s", out_dist)

    # Class-based report (filter auto by shared classes)
    logging.info("Building class-based report...")
    df_class = find_best_match(
        frames=frames,
        manual_detections=manual_det,
        auto_records=auto_recs,
        image_files=image_files,
        filter_fn=lambda rec, mc, auto_sum: bool(mc & auto_sum["classes"]),
    )
    out_class = base_dir / "comparison_report_classes.csv"
    df_class.to_csv(out_class, index=False)
    logging.info("Wrote class-based report to %s", out_class)


if __name__ == "__main__":
    main()

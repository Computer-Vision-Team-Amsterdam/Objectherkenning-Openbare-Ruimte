import argparse
import glob
import json
import math
import os

import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    """
    Return distance (in meters) between two lat/lon pairs, using Haversine formula.
    """
    R = 6371000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def load_manual_frame_metadata_csv(frame_csv_path):
    df = pd.read_csv(frame_csv_path, dtype={"image_name": str})
    if not {"gps_lat", "gps_lon"}.issubset(df.columns):
        raise ValueError(f"Expected 'gps_lat'/'gps_lon' in {frame_csv_path}")
    return df.set_index("image_name")


def load_manual_detections_csv(detection_csv_path):
    df = pd.read_csv(detection_csv_path, dtype={"image_name": str})
    if not {"image_name", "object_class"}.issubset(df.columns):
        raise ValueError(
            f"Expected 'image_name' and 'object_class' in {detection_csv_path}"
        )
    return df


def load_all_jsons(detection_dir):
    records = []
    for jpath in glob.glob(os.path.join(detection_dir, "*.json")):
        with open(jpath) as f:
            rec = json.load(f)
        if not {"image_file_name", "gps_data", "detections"}.issubset(rec.keys()):
            raise ValueError(f"JSON {jpath} missing required keys.")
        records.append(rec)
    return records


def build_manual_from_jsons(manual_records):
    # Build frame metadata DataFrame and detection DataFrame from manual JSONs
    frame_rows = []
    det_rows = []
    for rec in manual_records:
        name = rec["image_file_name"]
        lat = rec["gps_data"]["latitude"]
        lon = rec["gps_data"]["longitude"]
        frame_rows.append({"image_name": name, "gps_lat": lat, "gps_lon": lon})
        for det in rec.get("detections", []):
            det_rows.append({"image_name": name, "object_class": det["object_class"]})
    df_frames = pd.DataFrame(frame_rows).set_index("image_name")
    df_dets = pd.DataFrame(det_rows)
    return df_frames, df_dets


def summarize_auto_detections(auto_record):
    dets = auto_record.get("detections", [])
    return {"count": len(dets), "classes": set(d["object_class"] for d in dets)}


def main():
    parser = argparse.ArgumentParser(
        description="Compare manual and automatic detections by date"
    )
    parser.add_argument("date", help="Date directory name, e.g. 2025-05-19")
    args = parser.parse_args()

    date = args.date
    base_dir = os.path.dirname(__file__) or "."
    manual_dir = os.path.join(base_dir, "manual_runs", date)
    auto_dir = os.path.join(base_dir, "automatic_runs", date)

    # Determine manual input type: CSV vs JSON
    frame_meta_dir = os.path.join(manual_dir, "frame_metadata")
    det_meta_dir = os.path.join(manual_dir, "detection_metadata")
    if os.path.isdir(frame_meta_dir):
        # CSV-based manual run
        frames = load_manual_frame_metadata_csv(
            os.path.join(frame_meta_dir, "frame_metadata.csv")
        )
        manual = load_manual_detections_csv(
            os.path.join(det_meta_dir, "detection_metadata.csv")
        )
    else:
        # JSON-based manual run
        manual_records = load_all_jsons(det_meta_dir)
        frames, manual = build_manual_from_jsons(manual_records)

    # Available manual images folder (optional)
    image_files = {
        os.path.basename(p)
        for p in glob.glob(os.path.join(manual_dir, "images", "*"))
        if os.path.isfile(p)
    }

    # Load automatic records
    auto_records = load_all_jsons(os.path.join(auto_dir, "detection_metadata"))

    # Distance-based report
    dist_rows = []
    for img, row in frames.iterrows():
        lat, lon = float(row["gps_lat"]), float(row["gps_lon"])
        dists = [
            (
                rec,
                haversine(
                    lat, lon, rec["gps_data"]["latitude"], rec["gps_data"]["longitude"]
                ),
            )
            for rec in auto_records
        ]
        if not dists:
            continue
        best_rec, best_d = min(dists, key=lambda x: x[1])
        sub = (
            manual[manual["image_name"] == img] if not manual.empty else pd.DataFrame()
        )
        man_classes = set(sub["object_class"]) if not sub.empty else set()
        man_count = len(sub)
        auto_sum = summarize_auto_detections(best_rec)
        dist_rows.append(
            {
                "manual_image": img,
                "manual_lat": lat,
                "manual_lon": lon,
                "matched_auto_image": best_rec["image_file_name"],
                "auto_lat": best_rec["gps_data"]["latitude"],
                "auto_lon": best_rec["gps_data"]["longitude"],
                "distance_m": round(best_d, 1),
                "manual_obj_count": man_count,
                "auto_obj_count": auto_sum["count"],
                "manual_classes": ";".join(map(str, sorted(man_classes))),
                "auto_classes": ";".join(map(str, sorted(auto_sum["classes"]))),
                "image_available": img in image_files,
            }
        )
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(base_dir, f"comparison_report_distance_{date}.csv"), index=False
    )
    print(f"Distance-based report written to comparison_report_distance_{date}.csv.")

    # Class-based report
    class_rows = []
    for img, row in frames.iterrows():
        lat, lon = float(row["gps_lat"]), float(row["gps_lon"])
        sub = (
            manual[manual["image_name"] == img] if not manual.empty else pd.DataFrame()
        )
        man_classes = set(sub["object_class"]) if not sub.empty else set()
        man_count = len(sub)
        candidates = []
        for rec in auto_records:
            auto_sum = summarize_auto_detections(rec)
            if man_classes & auto_sum["classes"]:
                d = haversine(
                    lat, lon, rec["gps_data"]["latitude"], rec["gps_data"]["longitude"]
                )
                candidates.append((rec, auto_sum, d))
        if not candidates:
            continue
        best_rec, auto_sum, best_d = min(candidates, key=lambda x: x[2])
        class_rows.append(
            {
                "manual_image": img,
                "manual_lat": lat,
                "manual_lon": lon,
                "matched_auto_image": best_rec["image_file_name"],
                "auto_lat": best_rec["gps_data"]["latitude"],
                "auto_lon": best_rec["gps_data"]["longitude"],
                "distance_m": round(best_d, 1),
                "manual_obj_count": man_count,
                "auto_obj_count": auto_sum["count"],
                "manual_classes": ";".join(map(str, sorted(man_classes))),
                "auto_classes": ";".join(map(str, sorted(auto_sum["classes"]))),
                "image_available": img in image_files,
            }
        )
    pd.DataFrame(class_rows).to_csv(
        os.path.join(base_dir, f"comparison_report_classes_{date}.csv"), index=False
    )
    print(f"Class-based report written to comparison_report_classes_{date}.csv.")


if __name__ == "__main__":
    main()

"""
STEP 2: Add a railway-track Region of Interest (ROI) and flag detections
as DANGER (overlaps the track) or SAFE (doesn't).

This assumes a roughly fixed/stationary camera angle, so the track region
is a hardcoded trapezoid (tracks converge toward the horizon due to
perspective). You WILL need to tune the 4 points below for your own
footage -- see the calibration helper at the bottom of this file.

Setup:
    pip install ultralytics opencv-python numpy

Usage:
    # 1. First find your trapezoid points:
    python step2_add_roi.py --source path/to/video.mp4 --calibrate

    # 2. Then plug those points into TRACK_POLYGON below and run normally:
    python step2_add_roi.py --source path/to/video.mp4
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# TUNE THIS: four (x, y) pixel points outlining the track region, ordered
# top-left, top-right, bottom-right, bottom-left. Use --calibrate to find
# good values for your specific video/camera angle.
# ---------------------------------------------------------------------------
TRACK_POLYGON = np.array(
    [
        [857, 289],   # top-left    (near horizon, left rail)
        [1141, 279],  # top-right   (near horizon, right rail)
        [1259, 764],  # bottom-right (near camera, right rail)
        [595, 716],   # bottom-left  (near camera, left rail)
    ],
    dtype=np.int32,
)


def build_track_mask(frame_shape, polygon):
    """Binary mask: 255 inside the track polygon, 0 outside."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def box_overlaps_track(box_xyxy, track_mask):
    """True if the detection's bounding box overlaps any track-mask pixel."""
    x1, y1, x2, y2 = map(int, box_xyxy)
    h, w = track_mask.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False
    region = track_mask[y1:y2, x1:x2]
    return bool(np.any(region))


def run(source, model_name="yolov8n.pt", conf=0.4, save_path=None, slow_factor=1):
    model = YOLO(model_name)
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    track_mask = None
    writer = None
    if save_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"Saving output to: {save_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if track_mask is None:
            track_mask = build_track_mask(frame.shape, TRACK_POLYGON)

        results = model(frame, conf=conf, verbose=False)[0]

        danger_triggered = False

        for box in results.boxes:
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = map(int, xyxy)
            label = model.names[int(box.cls[0])]

            is_danger = box_overlaps_track(xyxy, track_mask)
            color = (0, 0, 255) if is_danger else (0, 255, 0)
            tag = "DANGER" if is_danger else "SAFE"
            if is_danger:
                danger_triggered = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{label}: {tag}", (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        # Draw the track ROI outline for visual reference
        overlay = frame.copy()
        cv2.fillPoly(overlay, [TRACK_POLYGON], (255, 200, 0))
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        cv2.polylines(frame, [TRACK_POLYGON], True, (255, 200, 0), 2)

        if danger_triggered:
            cv2.putText(
                frame, "!!! OBSTACLE ON TRACK !!!", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
            )

        if writer:
            writer.write(frame)

        cv2.imshow("Step 2: Track ROI + Danger Detection", frame)
        # slow_factor > 1 slows down the live preview window so you can actually see it
        if cv2.waitKey(max(1, slow_factor)) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Done. Saved video to: {save_path}")
    cv2.destroyAllWindows()


def calibrate(source):
    """
    Click 4 points on the first frame (top-left, top-right, bottom-right,
    bottom-left of the track) to get pixel coordinates for TRACK_POLYGON.
    """
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read a frame for calibration.")

    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")

    cv2.namedWindow("Calibrate: click TL, TR, BR, BL then press q")
    cv2.setMouseCallback("Calibrate: click TL, TR, BR, BL then press q", on_click)

    while True:
        display = frame.copy()
        for p in points:
            cv2.circle(display, p, 5, (0, 0, 255), -1)
        if len(points) == 4:
            cv2.polylines(display, [np.array(points)], True, (0, 255, 0), 2)
        cv2.imshow("Calibrate: click TL, TR, BR, BL then press q", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("\nCopy this into TRACK_POLYGON in step2_add_roi.py:")
    print(np.array(points, dtype=np.int32).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Click 4 points on the first frame to find TRACK_POLYGON coords",
    )
    parser.add_argument(
        "--save", default=None,
        help="Path to save output video (e.g. output.mp4) so you can review it slowly later",
    )
    parser.add_argument(
        "--slow", type=int, default=1,
        help="Slow down live preview window; try 50-100 for a watchable speed",
    )
    args = parser.parse_args()

    if args.calibrate:
        calibrate(args.source)
    else:
        run(args.source, args.model, args.conf, save_path=args.save, slow_factor=args.slow)

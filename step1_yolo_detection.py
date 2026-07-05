"""
STEP 1: Basic object detection on a video using a pretrained YOLO model.

Goal: get comfortable running YOLO frame-by-frame on video and drawing boxes.
No track/ROI logic yet -- that comes in step 2.

Setup:
    pip install ultralytics opencv-python

Usage:
    python step1_yolo_detection.py --source path/to/video.mp4
    python step1_yolo_detection.py --source 0        # webcam
"""

import argparse
import cv2
from ultralytics import YOLO


def main(source: str, model_name: str = "yolov8n.pt", conf: float = 0.4):
    # yolov8n.pt / yolo11n.pt etc. auto-downloads pretrained COCO weights
    # on first run (needs internet once; cached locally after that).
    model = YOLO(model_name)

    # source can be a video file path or an integer webcam index (as string "0")
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Starting processing. Total frames in video: {total_frames}")
    print("A window should open showing the video. If you don't see it, check your taskbar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video / stream.")
            break

        frame_num += 1
        if frame_num % 10 == 0 or frame_num == 1:
            print(f"Processing frame {frame_num}/{total_frames}...", flush=True)

        # Run detection on this frame
        results = model(frame, conf=conf, verbose=False)[0]

        # Draw boxes manually (gives us full control for step 2+)
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Step 1: YOLO Detection", frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", required=True, help="Path to video file, or '0' for webcam"
    )
    parser.add_argument(
        "--model", default="yolov8n.pt", help="YOLO weights (default: yolov8n.pt)"
    )
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    args = parser.parse_args()

    main(args.source, args.model, args.conf)


import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    video_path = input("Enter video path: ").strip().strip('"')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video. Check the file path.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = OUTPUT_DIR / "pose_output.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    frame_index = 0

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = int(frame_index * 1000 / fps)
                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                if result.pose_landmarks:
                    points = result.pose_landmarks[0]

                    for a, b in POSE_CONNECTIONS:
                        if points[a].visibility > 0.5 and points[b].visibility > 0.5:
                            p1 = (
                                int(points[a].x * width),
                                int(points[a].y * height),
                            )
                            p2 = (
                                int(points[b].x * width),
                                int(points[b].y * height),
                            )
                            cv2.line(frame, p1, p2, (0, 255, 0), 2)

                    for point in points:
                        if point.visibility > 0.5:
                            x = int(point.x * width)
                            y = int(point.y * height)
                            cv2.circle(frame, (x, y), 4, (0, 180, 255), -1)

                    cv2.putText(
                        frame, "MotionIQ | Pose Detected",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2,
                    )

                writer.write(frame)
                frame_index += 1

                if frame_index % 30 == 0:
                    print(f"Processed {frame_index} frames")

    finally:
        cap.release()
        writer.release()

    print(f"\nDone! Processed {frame_index} frames.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()

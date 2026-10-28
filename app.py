
import time
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
SAMPLE_VIDEO = BASE_DIR / "test_video.mp4"

PREVIEW_HEIGHT = 420

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

st.set_page_config(
    page_title="MotionIQ | Movement Studio",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GRAPHITE BLACK + NEON GREEN THEME
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #090D12;
        --panel: #141B1C;
        --border: #293A30;
        --neon: #B7FF39;
        --white: #F4F7F2;
        --muted: #A6B2AA;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 15% 0%,
                rgba(183, 255, 57, 0.055),
                transparent 42%
            ),
            #090D12;
        color: var(--white);
    }

    [data-testid="stHeader"] {
        background: rgba(9, 13, 18, 0.96);
    }

    [data-testid="stSidebar"] {
        background: #101519;
        border-right: 1px solid #29343A;
    }

    [data-testid="stSidebar"] * {
        color: var(--white);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .brand {
        font-size: 29px;
        font-weight: 850;
        color: var(--white);
        letter-spacing: -1px;
    }

    .brand span {
        color: var(--neon);
    }

    .muted {
        color: var(--muted);
        font-size: 13px;
    }

    .eyebrow {
        color: var(--neon);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    [data-testid="stMetric"] {
        background: #141B1C;
        border: 1px solid #293A30;
        border-radius: 14px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--white) !important;
    }

    [data-testid="stMetricDelta"] {
        color: var(--neon) !important;
    }

    .stButton > button {
        min-height: 44px;
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid #38443D;
        transition: 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: var(--neon);
        color: #10160A;
        border: none;
    }

    .stButton > button[kind="primary"]:hover {
        background: #CAFF70;
        color: #10160A;
        border: none;
    }

    [data-testid="stProgressBar"] > div > div {
        background: var(--neon);
    }

    [data-testid="stImage"] img {
        display: block;
        width: 100%;
        height: 420px;
        object-fit: contain;
        border-radius: 12px;
        background: #080B0E;
        border: 1px solid #29343A;
    }

    div[data-testid="stFileUploader"] {
        background: #141B1C;
        border: 1px solid #293A30;
        border-radius: 12px;
        padding: 10px;
    }

    div[data-baseweb="select"] > div {
        background: #141B1C;
        border-color: #293A30;
    }

    div[role="radiogroup"] label {
        color: var(--white);
    }

    hr {
        border-color: #29343A;
    }

    h1, h2, h3 {
        color: var(--white);
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        [data-testid="stImage"] img {
            height: 320px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CHECK MODEL
# =========================================================

if not MODEL_PATH.exists():
    st.error(f"Pose model not found: {MODEL_PATH}")
    st.info(
        "Confirm that pose_landmarker_lite.task exists "
        "inside the models folder."
    )
    st.stop()


# =========================================================
# LANDMARK HELPERS
# =========================================================

def is_visible(landmarks, index):
    point = landmarks[index]
    visibility = getattr(point, "visibility", 1.0)
    presence = getattr(point, "presence", 1.0)

    return visibility > 0.45 and presence > 0.3


def get_point(landmarks, index, width, height):
    point = landmarks[index]
    return int(point.x * width), int(point.y * height)


def calculate_angle(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-6:
        return None

    cosine = np.clip(
        np.dot(ba, bc) / denominator,
        -1.0,
        1.0,
    )

    return float(np.degrees(np.arccos(cosine)))


def get_knee_angle(landmarks, width, height):
    candidates = []

    sides = [
        (23, 25, 27, "Left"),
        (24, 26, 28, "Right"),
    ]

    for hip, knee, ankle, side in sides:
        indices = (hip, knee, ankle)

        if not all(is_visible(landmarks, i) for i in indices):
            continue

        points = [
            get_point(landmarks, i, width, height)
            for i in indices
        ]

        angle = calculate_angle(*points)

        if angle is None:
            continue

        confidence = min(
            getattr(landmarks[i], "visibility", 1.0)
            for i in indices
        )

        candidates.append((confidence, angle, side))

    if not candidates:
        return None, None

    _, angle, side = max(candidates, key=lambda item: item[0])
    return angle, side


# =========================================================
# SQUAT COUNTER
# =========================================================

class SquatCounter:
    def __init__(self, down_threshold, up_threshold):
        self.reps = 0
        self.phase = "Standing"
        self.feedback = "Ready to begin"
        self.minimum_angle = None
        self.depths = []
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold

    def update(self, angle):
        if angle is None:
            self.feedback = "Knee not visible"
            return

        if angle < self.down_threshold:
            if self.phase in ("Standing", "Up"):
                self.phase = "Down"
                self.minimum_angle = angle
                self.feedback = "Squat down phase"
            else:
                if self.minimum_angle is None:
                    self.minimum_angle = angle
                else:
                    self.minimum_angle = min(
                        self.minimum_angle,
                        angle,
                    )

                self.feedback = "Continue squat"

        elif angle > self.up_threshold:
            if self.phase == "Down":
                self.phase = "Up"
                self.reps += 1

                if self.minimum_angle is not None:
                    self.depths.append(self.minimum_angle)

                self.feedback = "Rep completed!"
            else:
                self.phase = "Standing"
                self.feedback = "Ready for next rep"

        elif self.phase == "Down":
            self.feedback = "Continue movement"
        else:
            self.feedback = "Bend your knees to start"


# =========================================================
# DRAW POSE OVERLAY
# =========================================================

def draw_overlay(frame, landmarks, counter):
    height, width = frame.shape[:2]

    for point_a, point_b in POSE_CONNECTIONS:
        if (
            is_visible(landmarks, point_a)
            and is_visible(landmarks, point_b)
        ):
            p1 = get_point(
                landmarks, point_a, width, height
            )
            p2 = get_point(
                landmarks, point_b, width, height
            )

            cv2.line(
                frame,
                p1,
                p2,
                (57, 255, 183),
                2,
                cv2.LINE_AA,
            )

    for i in range(len(landmarks)):
        if is_visible(landmarks, i):
            point = get_point(
                landmarks, i, width, height
            )

            cv2.circle(
                frame,
                point,
                3,
                (183, 255, 57),
                -1,
                cv2.LINE_AA,
            )

    angle, side = get_knee_angle(
        landmarks, width, height
    )

    counter.update(angle)

    # Dark information panel
    cv2.rectangle(
        frame,
        (10, 10),
        (305, 145),
        (15, 22, 22),
        -1,
    )

    cv2.putText(
        frame,
        f"SQUATS: {counter.reps}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (183, 255, 57),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"PHASE: {counter.phase}",
        (20, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 247, 242),
        2,
        cv2.LINE_AA,
    )

    angle_text = (
        f"{side} knee: {angle:.0f} deg"
        if angle is not None
        else "Knee: unavailable"
    )

    cv2.putText(
        frame,
        angle_text,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (183, 255, 57),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        counter.feedback[:30],
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 230, 220),
        1,
        cv2.LINE_AA,
    )

    return angle


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">Motion<span>IQ</span></div>'
        '<p class="muted">AI MOVEMENT INTELLIGENCE</p>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Video source")

    source_mode = st.radio(
        "Choose video",
        ["Sample video", "Upload video"],
    )

    uploaded_file = None

    if source_mode == "Upload video":
        uploaded_file = st.file_uploader(
            "Upload exercise video",
            type=["mp4", "mov", "avi", "mkv"],
        )

    st.divider()
    st.markdown("### Performance")

    resolution = st.select_slider(
        "Processing width",
        options=[320, 400, 480, 640, 800],
        value=480,
        help="Lower resolution can improve CPU processing speed.",
    )

    inference_every = st.select_slider(
        "Pose inference interval",
        options=[1, 2, 3],
        value=2,
        format_func=lambda n: {
            1: "Every frame",
            2: "Every 2nd frame",
            3: "Every 3rd frame",
        }[n],
        help=(
            "Higher intervals reduce model workload, "
            "but can miss fast movements."
        ),
    )

    preview_every = st.select_slider(
        "Preview update interval",
        options=[1, 2, 3, 5],
        value=3,
        format_func=lambda n: f"Every {n} frame(s)",
    )

    st.divider()
    st.markdown("### Squat thresholds")

    down_threshold = st.slider(
        "Bottom knee angle",
        80,
        130,
        110,
        step=5,
    )

    up_threshold = st.slider(
        "Standing knee angle",
        140,
        175,
        155,
        step=5,
    )

    st.divider()
    st.caption("LOCAL AI · MEDIAPIPE POSE LANDMARKER LITE")


# =========================================================
# VIDEO SOURCE
# =========================================================

video_source = None

if source_mode == "Sample video":
    if SAMPLE_VIDEO.exists():
        video_source = str(SAMPLE_VIDEO)
    else:
        st.error(
            "test_video.mp4 was not found in the MotionIQ folder."
        )
else:
    video_source = uploaded_file


# =========================================================
# DASHBOARD
# =========================================================

st.markdown(
    '<div class="eyebrow">COMPUTER VISION / MOTION ANALYTICS</div>'
    '<h1 style="margin-top:4px">Movement Studio</h1>'
    '<p class="muted">Pose estimation and squat analysis.</p>',
    unsafe_allow_html=True,
)

video_col, metrics_col = st.columns(
    [2.1, 1],
    gap="large",
)

with video_col:
    st.markdown("#### Video analysis")
    preview = st.empty()
    progress_bar = st.progress(0)
    progress_text = st.empty()

with metrics_col:
    st.markdown("#### Session metrics")
    rep_metric = st.empty()
    phase_metric = st.empty()
    angle_metric = st.empty()
    fps_metric = st.empty()
    detection_metric = st.empty()

start_button = st.button(
    "▶ Start analysis",
    type="primary",
    use_container_width=True,
    disabled=(video_source is None),
)

if video_source is None:
    st.info(
        "Choose a sample video or upload a video from the sidebar."
    )


# =========================================================
# VIDEO PROCESSING
# =========================================================

if start_button:
    cap = None
    temp_path = None

    try:
        if isinstance(video_source, str):
            cap = cv2.VideoCapture(video_source)
        else:
            suffix = Path(video_source.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:
                temp_file.write(video_source.getbuffer())
                temp_path = temp_file.name

            cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            st.error("Could not open the selected video.")
            st.stop()

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        source_fps = cap.get(cv2.CAP_PROP_FPS)

        if not source_fps or source_fps <= 0:
            source_fps = 25.0

        counter = SquatCounter(
            down_threshold,
            up_threshold,
        )

        frame_index = 0
        inference_count = 0
        detected_frames = 0
        angle_sum = 0.0
        angle_count = 0

        last_timestamp_ms = -1
        last_angle = None
        last_display_frame = None

        start_time = time.perf_counter()

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                current_index = frame_index
                frame_index += 1

                height, width = frame.shape[:2]

                # Resize only when the frame exceeds selected width.
                if width > resolution:
                    scale = resolution / width
                    new_size = (
                        resolution,
                        max(1, int(height * scale)),
                    )

                    frame = cv2.resize(
                        frame,
                        new_size,
                        interpolation=cv2.INTER_AREA,
                    )

                should_infer = (
                    current_index % inference_every == 0
                )

                if should_infer:
                    rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb,
                    )

                    timestamp_ms = max(
                        int(current_index * 1000 / source_fps),
                        last_timestamp_ms + 1,
                    )

                    last_timestamp_ms = timestamp_ms

                    result = landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms,
                    )

                    inference_count += 1

                    if result.pose_landmarks:
                        detected_frames += 1

                        last_angle = draw_overlay(
                            frame,
                            result.pose_landmarks[0],
                            counter,
                        )

                        if last_angle is not None:
                            angle_sum += last_angle
                            angle_count += 1

                    else:
                        last_angle = None
                        counter.update(None)

                        cv2.putText(
                            frame,
                            "NO PERSON DETECTED",
                            (18, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (80, 80, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    last_display_frame = frame.copy()

                should_update_ui = (
                    current_index % preview_every == 0
                    or (
                        total_frames > 0
                        and current_index >= total_frames - 1
                    )
                )

                if (
                    should_update_ui
                    and last_display_frame is not None
                ):
                    elapsed = (
                        time.perf_counter() - start_time
                    )

                    processing_fps = (
                        frame_index / elapsed
                        if elapsed > 0
                        else 0
                    )

                    preview.image(
                        cv2.cvtColor(
                            last_display_frame,
                            cv2.COLOR_BGR2RGB,
                        ),
                        channels="RGB",
                        use_container_width=True,
                        output_format="JPEG",
                    )

                    rep_metric.metric(
                        "Squat repetitions",
                        counter.reps,
                    )

                    phase_metric.metric(
                        "Movement phase",
                        counter.phase,
                    )

                    angle_metric.metric(
                        "Knee angle",
                        (
                            f"{last_angle:.1f}°"
                            if last_angle is not None
                            else "Unavailable"
                        ),
                    )

                    fps_metric.metric(
                        "Processing FPS",
                        f"{processing_fps:.1f}",
                    )

                    detection_rate = (
                        100 * detected_frames / inference_count
                        if inference_count
                        else 0
                    )

                    detection_metric.metric(
                        "Pose detection rate",
                        f"{detection_rate:.1f}%",
                    )

                    if total_frames > 0:
                        progress = min(
                            frame_index / total_frames,
                            1.0,
                        )

                        progress_bar.progress(progress)

                        progress_text.caption(
                            f"Frame {frame_index:,} / "
                            f"{total_frames:,} "
                            f"· {progress * 100:.1f}%"
                        )
                    else:
                        progress_text.caption(
                            f"Frames read: {frame_index:,}"
                        )

        elapsed = time.perf_counter() - start_time

        average_fps = (
            frame_index / elapsed
            if elapsed > 0
            else 0
        )

        mean_angle = (
            angle_sum / angle_count
            if angle_count
            else None
        )

        detection_rate = (
            100 * detected_frames / inference_count
            if inference_count
            else 0
        )

        progress_bar.progress(1.0)
        progress_text.success("Analysis complete.")

        st.divider()
        st.markdown("## Session summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total squats",
            counter.reps,
        )

        col2.metric(
            "Frames read",
            f"{frame_index:,}",
        )

        col3.metric(
            "Average processing FPS",
            f"{average_fps:.1f}",
        )

        col4.metric(
            "Mean knee angle",
            (
                f"{mean_angle:.1f}°"
                if mean_angle is not None
                else "Unavailable"
            ),
        )

        st.write(
            f"**Pose detection rate:** {detection_rate:.1f}%"
        )

        st.write(
            f"**Pose inference calls:** {inference_count:,}"
        )

        if counter.depths:
            st.write(
                "**Average minimum knee angle:** "
                f"{np.mean(counter.depths):.1f}°"
            )

        st.info(
            "For the most accurate squat counting, set pose "
            "inference to Every frame. Higher intervals can "
            "miss quick movements."
        )

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)

    finally:
        if cap is not None:
            cap.release()

        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except OSError:
                pass


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<hr><p class="muted">'
    'MOTIONIQ · COMPUTER VISION PROTOTYPE'
    '</p>',
    unsafe_allow_html=True,
)

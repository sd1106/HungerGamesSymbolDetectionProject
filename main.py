import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import drawing_utils

MODEL_DIR = Path(__file__).parent / "models"
MODELS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}


def ensure_model(name: str) -> str:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / name
    if not path.exists():
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(MODELS[name], path)
    return str(path)


red_dot = drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=3)
green_lines = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

# --- Landmark indices -----------------------------------------------------------
NOSE = vision.PoseLandmark.NOSE
LEFT_WRIST = vision.PoseLandmark.LEFT_WRIST
RIGHT_WRIST = vision.PoseLandmark.RIGHT_WRIST
HAND_WRIST = 0         
RING_FINGER_TIP = 16   
PINKY_TIP = 20         

pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ensure_model("pose_landmarker_lite.task")),
    running_mode=vision.RunningMode.VIDEO,
)
hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ensure_model("hand_landmarker.task")),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)

cap = cv2.VideoCapture(0)
start = time.monotonic()

with vision.PoseLandmarker.create_from_options(pose_options) as pose, \
        vision.HandLandmarker.create_from_options(hand_options) as hands:

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - start) * 1000)
        results = pose.detect_for_video(mp_image, timestamp_ms)
        hand_results = hands.detect_for_video(mp_image, timestamp_ms)

        height = img.shape[0]
        wrist_y = nose_y = None 

        if results.pose_landmarks:
            pose_lms = results.pose_landmarks[0]  # first detected person
            drawing_utils.draw_landmarks(
                img,
                pose_lms,
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=red_dot,
                connection_drawing_spec=green_lines,
            )
            wrist_y = min(pose_lms[LEFT_WRIST].y, pose_lms[RIGHT_WRIST].y) * height
            nose_y = pose_lms[NOSE].y * height

        tribute = False

        if hand_results.hand_landmarks:
            for hand_lms in hand_results.hand_landmarks:
                drawing_utils.draw_landmarks(
                    img,
                    hand_lms,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=red_dot,
                    connection_drawing_spec=green_lines,
                )

            first_hand = hand_results.hand_landmarks[0]
            difference = (first_hand[RING_FINGER_TIP].y - first_hand[PINKY_TIP].y) * height
            print(difference)

            sign_above_nose = (
                nose_y is not None and first_hand[HAND_WRIST].y * height < nose_y
            )
            tribute = difference < -35 and sign_above_nose

        if tribute:
            cv2.putText(img, "I Volunteer as Tribute!", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif wrist_y is not None and wrist_y < nose_y:
            cv2.putText(img, "Hand is Raised", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Tracker", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
import mediapipe as mp
import numpy as np
import cv2

mp_holistic = mp.solutions.holistic

UPPER_BODY_POSE = list(range(25))
POSE_FEATURES = len(UPPER_BODY_POSE) * 3
HAND_FEATURES = 21 * 3
TOTAL_FEATURES = POSE_FEATURES + HAND_FEATURES * 2  # 201

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def normalize_keypoints(keypoints):
    try:
        pose = keypoints[:POSE_FEATURES].reshape(-1, 3)
        left_shoulder = pose[11]
        right_shoulder = pose[12]
        center = (left_shoulder + right_shoulder) / 2
        pose -= center
        keypoints[:POSE_FEATURES] = pose.flatten()
    except:
        pass
    return keypoints

def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark[:25]]).flatten()
    else:
        pose = np.zeros(POSE_FEATURES)

    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(HAND_FEATURES)

    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(HAND_FEATURES)

    keypoints = np.concatenate([pose, left_hand, right_hand])

    if keypoints.shape[0] != TOTAL_FEATURES:
        keypoints = np.zeros(TOTAL_FEATURES)

    return normalize_keypoints(keypoints)

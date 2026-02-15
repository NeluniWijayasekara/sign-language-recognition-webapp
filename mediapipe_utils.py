import mediapipe as mp
import numpy as np
import cv2

mp_holistic = mp.solutions.holistic

UPPER_BODY_POSE = list(range(25))  # 0-24
POSE_FEATURES = len(UPPER_BODY_POSE) * 3  # 75
HAND_FEATURES = 21 * 3  # 63
TOTAL_FEATURES = POSE_FEATURES + HAND_FEATURES * 2  # 201

def normalize_keypoints(keypoints):
    """Normalize keypoints relative to shoulder center"""
    try:
        pose = keypoints[:POSE_FEATURES].reshape(-1, 3)
        
        # Get shoulder landmarks (indices 11 and 12 in pose landmarks)
        if len(pose) > 12:
            left_shoulder = pose[11]
            right_shoulder = pose[12]
            center = (left_shoulder + right_shoulder) / 2
            pose -= center
            keypoints[:POSE_FEATURES] = pose.flatten()
    except Exception as e:
        print(f"Normalization error: {e}")
        pass
    return keypoints

def extract_landmarks(frame):
    """Extract pose and hand landmarks from a single frame"""
    # Initialize holistic for each frame to avoid threading issues
    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # Extract pose landmarks (first 25 landmarks for upper body)
        if results.pose_landmarks:
            pose = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.pose_landmarks.landmark[:25]
            ]).flatten()
        else:
            pose = np.zeros(POSE_FEATURES)

        # Extract left hand landmarks
        if results.left_hand_landmarks:
            left_hand = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.left_hand_landmarks.landmark
            ]).flatten()
        else:
            left_hand = np.zeros(HAND_FEATURES)

        # Extract right hand landmarks
        if results.right_hand_landmarks:
            right_hand = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.right_hand_landmarks.landmark
            ]).flatten()
        else:
            right_hand = np.zeros(HAND_FEATURES)

        # Combine all landmarks
        keypoints = np.concatenate([pose, left_hand, right_hand])

        # Ensure correct shape
        if keypoints.shape[0] != TOTAL_FEATURES:
            keypoints = np.zeros(TOTAL_FEATURES)

        return normalize_keypoints(keypoints)
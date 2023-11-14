import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = 'pose_detection/media/exercising.mp4'
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Video file ended or failed to open.")
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        results = pose.process(image)
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks with different colors for each connection
        connections = mp_pose.POSE_CONNECTIONS
        connection_colors = [(245, 200, 66), (66, 245, 100), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]

        for idx, connection in enumerate(connections):
            start_idx, end_idx = connection
            start_landmark = results.pose_landmarks.landmark[start_idx]
            end_landmark = results.pose_landmarks.landmark[end_idx]

            # Use different color for each connection
            color = connection_colors[idx % len(connection_colors)]

            # Draw each connection individually
            cv2.line(image, (int(start_landmark.x * image.shape[1]), int(start_landmark.y * image.shape[0])),
                     (int(end_landmark.x * image.shape[1]), int(end_landmark.y * image.shape[0])), color, 2)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

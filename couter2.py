import time
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    angle = round(angle, 2)
    return angle


shoulder_state = None
elbow_state = None
hand_state = None
counter = 0
temp_state = []
a=0
bad_counter = []

# Counter logic function
# logic:
# *shoulder logic:
# if shoulder angle are apoximate 100 degree than the shoulder_state is up_correct
# else if shoulder angle are lower than 20 degree than the shoulder_state is down_correct
# else the shoulder_state is wrong
# *elbow logic:
# if elbow angle are apoximate 170 degree than the elbow_sate is straight
# else if elbow angle are lower than 150 degree than the elbow_state is not straight
# *couter logic:
# if (shoulder_state is down_correct and elbow_state is straight) and (the state of 2 body point are being keep for 0.5 sec) then set the hand_state to down
# if (shoulder_state is up_correct and elbow_state is straight) and (the hand_state is down) and (the state of 2 body point are being keep for 0.5 sec) than the counter is increase by 1 and hand_state is change to up
# if (shoulder_state is wrong or elbow_state is not straight) and (the state of 2 body point are being keep for 0.5 sec) and (the hand_state is down) than the counter is increase by 1 and bad_counter(an array) record the index of the counter so we can know which counter is wrong


def counter_logic(
    shoulder_angle1, shoulder_angle2, hand_state, counter, shoulder_state, temp_state,a
):
    
    # Shoulder logic
    if (
        shoulder_angle1 > 90
        and shoulder_angle1 < 180
        and shoulder_angle2 > 90
        and shoulder_angle2 < 180
    ):
        shoulder_state = "up"

    elif shoulder_angle1 < 20 and shoulder_angle2 < 20:
        shoulder_state = "down"
    elif (
        shoulder_angle1 > 50
        and shoulder_angle1 < 90
        and shoulder_angle2 > 50
        and shoulder_angle2 < 90
    ):
        shoulder_state = "middle"

    # Elbow logic
    # if elbow_angle1 > 160 and elbow_angle2 > 160 :
    #     elbow_state = "straight"
    # else:
    #     elbow_state = "not straight"

    # Counter logic
    if shoulder_state == "down":  # and elbow_state == "straight":
        hand_state = "down"
        temp_state.append("down")
        a=1


    elif shoulder_state == "up" and hand_state == "middle" :  # elbow_state == "straight" and
        # counter += 1
        hand_state = "up"
        temp_state.append("up")
        a=2

    elif shoulder_state == "middle" and hand_state == "down" :
        counter += 1
        hand_state = "middle"
        temp_state.append("middle")
        a=3
    print(a)

    return hand_state, counter, shoulder_state, temp_state,a 


def check(temp_state, counter, a,bad_counter):
    
    if counter != 0 and a==1:
        if "up" in temp_state:
            print("dung")
            
            

        else:
            if counter not in bad_counter:
                bad_counter.append(counter)
                print("sai")
            if counter in bad_counter:
                pass
            
            
    if a==3 and counter != 0:
        temp_state.clear()
    return bad_counter


vid_path = "test.mp4"
cap = cv2.VideoCapture(0)
# Curl counter variables
counter = 0
stage = None

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates1

            shoulder_left = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            elbow_left = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            wrist_left = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]

            shoulder_right = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            ]
            elbow_right = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            ]
            wrist_right = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            ]

            hip_left = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            ]
            hip_right = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            ]

            # calculate angle join

            leftshoulder_angle = calculate_angle(hip_left, shoulder_left, elbow_left)
            rightshoulder_angle = calculate_angle(
                hip_right, shoulder_right, elbow_right
            )
            leftelbow_angle = calculate_angle(shoulder_left, elbow_left, wrist_left)
            rightelbow_angle = calculate_angle(shoulder_right, elbow_right, wrist_right)

            # Visualize angle
            cv2.putText(
                image,
                str(leftshoulder_angle),
                tuple(np.multiply(shoulder_left, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                str(rightshoulder_angle),
                tuple(np.multiply(shoulder_right, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(leftelbow_angle),
                tuple(np.multiply(elbow_left, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(rightelbow_angle),
                tuple(np.multiply(elbow_right, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        except:
            pass

        hand_state, counter, shoulder_state, temp_state,a = counter_logic(
            leftshoulder_angle,
            rightshoulder_angle,
            hand_state,
            counter,
            shoulder_state,
            temp_state,
            a
        )
        badct = check(temp_state, counter, a,bad_counter)
        # print(temp_state)
        print(badct)
        print(counter)
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(
            image,
            "REPS",
            (15, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(counter),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Stage data
        cv2.putText(
            image,
            "STAGE",
            (65, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            hand_state,
            (60, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        connections = mp_pose.POSE_CONNECTIONS
        connection_colors = [
            (245, 200, 66),
            (66, 245, 100),
            (0, 0, 255),
            (255, 0, 0),
            (0, 255, 255),
            (255, 255, 0),
        ]

        # Render detections
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
        #                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        #                          )
        # Render connections
        for idx, connection in enumerate(connections):
            start_idx, end_idx = connection
            start_landmark = results.pose_landmarks.landmark[start_idx]
            end_landmark = results.pose_landmarks.landmark[end_idx]

            # Use different color for each connection
            color = connection_colors[idx % len(connection_colors)]

            # Draw each connection individually
            cv2.line(
                image,
                (
                    int(start_landmark.x * image.shape[1]),
                    int(start_landmark.y * image.shape[0]),
                ),
                (
                    int(end_landmark.x * image.shape[1]),
                    int(end_landmark.y * image.shape[0]),
                ),
                color,
                2,
            )

        cv2.imshow("Mediapipe Feed", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

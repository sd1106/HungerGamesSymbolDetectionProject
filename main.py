import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

red_dot = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=3)
green_lines = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

VideoCapture = cv2.VideoCapture(0)

with (mp_pose.Pose(static_image_mode=False) as pose):
    with mp_hands.Hands(static_image_mode=False) as hands:

        while VideoCapture.isOpened():
            ret, img = VideoCapture.read()
            if not ret:
                break


            # Process the image
            results = pose.process(img)
            handResults = hands.process(img)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(img, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=red_dot, connection_drawing_spec=green_lines)

                wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

                wrist_pixel_location = wrist.y * img.shape[0]
                nose_pixel_location = nose.y * img.shape[0]


            if handResults.multi_hand_landmarks:
                for hand_lms in handResults.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img,landmark_list=hand_lms,connections=mp_hands.HAND_CONNECTIONS,landmark_drawing_spec=red_dot,connection_drawing_spec=green_lines)

                ring = handResults.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_pixel_location = ring.y * img.shape[0]

                pinky = handResults.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_TIP]
                pinky_pixel_location = pinky.y * img.shape[0]

                difference = ring_pixel_location - pinky_pixel_location
                print(difference)

                if difference < -35:
                    cv2.putText(img, "I Volunteer as Tribute!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif wrist_pixel_location < nose_pixel_location:
                    cv2.putText(img, "Hand is Raised", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)



            cv2.imshow('Tracker', img)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break

VideoCapture.release()
cv2.destroyAllWindows()
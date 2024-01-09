import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarkerResult
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker
from mediapipe.python.solutions import pose, drawing_utils, drawing_styles


gl_result = None
def result_callback(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global gl_result
    gl_result = result
    
model_file = open('src/pose_landmarker_heavy.task', "rb")
model_data = model_file.read()
model_file.close()
base_options = BaseOptions(model_asset_buffer=model_data)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    output_segmentation_masks=True,
    result_callback = result_callback
)

with PoseLandmarker.create_from_options(options) as landmarker:
    cap=cv2.VideoCapture(0)
    prev_time = 0
    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        frame_np = np.array(img)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_np)
        timestamp = int(round(time.time()*1000))
        frame = img.numpy_view()
        landmarker.detect_async(img, timestamp)
        
        if type(gl_result) is not type(None):
            with open('keypoints.txt', 'a') as f:
                f.write(str(gl_result.pose_landmarks))
            pose_landmarks_list = gl_result.pose_landmarks
            for i in range(len(pose_landmarks_list)):
                pose_landmarks = pose_landmarks_list[i]

                body_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                body_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])

                # draw pose landmarks
                drawing_utils.draw_landmarks(
                    frame,
                    body_landmarks_proto,
                    pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                )

        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time
        cv2.putText(frame, str(int(fps)), (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 3)
        cv2.putText(frame, "Press ESC to close window", (450,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)
        cv2.imshow('MediaPipe Pose', frame)
        #if esc key is pressed then break out of the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

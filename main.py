import cv2
import mediapipe as mp
import numpy as np
import time
from keypoint_classifier import KeyPointClassifier
import screeninfo

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
keypoint_classifier = KeyPointClassifier()

playing = False
mode = 0
number = 4 # number of poses
default = False
hand_sign_id = 0
t_prev = time.localtime(time.time())
t_now = time.localtime(time.time())

code = ["6","9"]

counting = True

countdown = 180

start_time = 0

def pre_process_landmark(results):
    land_mark_list = []
    
    for data_points in results.pose_world_landmarks.landmark:
      land_mark_list.append(data_points.x)
      land_mark_list.append(data_points.y)
      land_mark_list.append(data_points.z)
    return land_mark_list
       
def convertToTime(countdown):
  minutes = int(countdown / 60)
  seconds = (countdown - (minutes * 60))

  print(countdown, minutes, seconds)
  if seconds <10:
    return "0{min} : 0{sec}".format(min = minutes, sec = seconds)
  else :
    return "0{min} : {sec}".format(min = minutes, sec = seconds)

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.8) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as pose:
  
  while cap.isOpened():
    t_now = time.localtime(time.time())
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    
    key=cv2.waitKey(1)
    if key == 110: #n
      mode = 0
    if key == 104: #h
      mode = 2
    if key == 27:
      break

    if mode == 0:
      try:  
        hand_sign_id = keypoint_classifier(pre_process_landmark(results))
        if hand_sign_id == 4:
          print("Correct pose!")
          if counting == True:
            if countdown >0:
              if t_prev.tm_sec != t_now.tm_sec:
                countdown = countdown -1
            elif countdown == 0:
              counting = False
              # print ("finish")
        else:
          # print("Wrong pose!")
          if counting == True:
            if countdown < 180:
              if t_prev.tm_sec != t_now.tm_sec:
                countdown = countdown +1
      except:
        if counting == True:
          if countdown < 180:
              if t_prev.tm_sec != t_now.tm_sec:
                countdown = countdown +1
          # print("No Pose Detected")

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    if counting == True:
      cv2.rectangle(image,(10,300),(260,400),(0,255,0),3)
      cv2.putText(image, convertToTime(countdown) ,(10,375), font, 2,(0,255,0),2)
    else:
      cv2.rectangle(image,(260,280),(325,390),(0,255,0),2)
      cv2.putText(image, (code[0]) ,(250,375), font, 4,(0,255,0),2)
      cv2.rectangle(image,(330,280),(400,390),(0,255,0),2)
      cv2.putText(image, (code[1]) ,(330,375), font, 4,(0,255,0),2)

    # Flip the image horizontally for a selfie-view display.
    window_name = 'pose'
    screen = screeninfo.get_monitors()[0]
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, image)
    t_prev = t_now
cap.release()


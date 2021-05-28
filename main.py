import cv2
import mediapipe as mp
import time
import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ProcessFrame import ProcessFrame

# This program uses the movement classifier and the emotion classifier
# to continuously  predict the emotional state, from a live camera-input,
# that captures one dancing person. 

# Elements from MediaPipe's solution to predict landmarks are used in this file 
# copied from: https://google.github.io/mediapipe/solutions/pose

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

processFrame = ProcessFrame()
getAttributes = ProcessFrame.predictAtributesInFrame

img = np.zeros((500,800,3), np.uint8)

# exit frame is a list of 25 frames classified with 7 
# as no pose was detected in the frame
exit_frame = [7]*25

# Load data and target csv-files for the movement classifier and create the classifier
movement_data_filename = 'movement_data.csv'
movement_attributes = ['speed','aceleration','x_dir','y_dir','feet_hip_dist',
                        'hands_shoulder_dist','rHand_lHand_dist','rFoot_lFoot_dist',
                        'hand_speed','feet_speed','hand_acc','feet_acc']
movement_data = pandas.read_csv(movement_data_filename, names=movement_attributes)

movement_target_filename = 'movement_target.csv'
movement_labels = ['class']
movement_target = pandas.read_csv(movement_target_filename, names=movement_labels)

movement_classifier = RandomForestClassifier(random_state=0)
movement_classifier.fit(movement_data,movement_target)

# Load data and target csv-files for the emotion classifier and create the classifier
emotion_data_filename = 'emotion_data.csv'
emotion_attributes = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
emotion_data = pandas.read_csv(emotion_data_filename, names=emotion_attributes)

emotion_target_filename = 'emotion_target.csv'
emotion_labels = ['class']
emotion_target = pandas.read_csv(emotion_target_filename, names=emotion_labels)

emotion_classifier = RandomForestClassifier(random_state=0)
emotion_classifier.fit(emotion_data,emotion_target)

def predict_pose(test_data):
  # Predict pose unless test_data is empty
  # and return the pose's class-label
  # 7 indicates that no pose was found
  if not test_data:
    pose_name = 7
  else:
    all_poses = [0,1,2,3,4,5,6]
    predicted_pose = movement_classifier.predict([test_data])
    pose_name = all_poses[(predicted_pose[0])]
  return pose_name

def classify_frame():
  print("Classifying 25 frames:")
  # class_list is used to store the predicted movements
  # runs counts every frame 
  class_list =[]
  runs = 0
  cap = cv2.VideoCapture(0)

  with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:

    while cap.isOpened() and runs < 75:
      success, image = cap.read()

      if not success:
        print("Ignoring empty camera frame.")
        continue
      # Process only every third frame
      if (runs % 3 == 0):
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        
        test_data, results = getAttributes(processFrame,image,pose)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # predict pose and write result on video
        final_result = predict_pose(test_data) 
        print("Frame no. " + str(runs) + " was classified as: " + str(final_result))
        #font = cv2.FONT_HERSHEY_SIMPLEX  
        #cv2.putText(image, final_result,(20,70),font,2,(0,0,255),2,cv2.LINE_AA)
        class_list.append(final_result)
        
        # Finally show resulting frame
        #cv2.imshow('MediaPipe Pose', image)
        img[:]=[255,255,255]
        cv2.imshow('Scene', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      runs = runs + 1
  
  cap.release()
  return class_list

def classify_list(class_list):
  predicted_mood = emotion_classifier.predict([class_list])
  print("predicted mood: " + str(predicted_mood))
  return predicted_mood

def wait_for_dancer():
  # This function shows a white screen while waiting for a pose to get detected 
  print("Waiting for dance")
  img[:]=(255,255,255)
  cap = cv2.VideoCapture(0)

  with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = pose.process(image)
      
      # Check if frame has pose-landmarks 
      if hasattr(results.pose_landmarks, 'landmark'):
        break
      
      else:
        cv2.imshow('Scene',img) 
      if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    print("Dancer has antered the stage")
    # Happening when pose has been detected 
    # screen changes from white->pink->white
    for x in range (255,0,-5):
      img[:]=(255,x,255)
      cv2.imshow('Scene',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    for x in range (0,255,5):
      img[:]=(255,x,255)
      cv2.imshow('Scene',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def sunset():
  print("Showing sunset happening before exiting program")
  # this function creates a screen 
  # with changing colors and shapes, simulating a sunset
  o = 0
  b = 255
  c = -100
  circleCenter_x = 400
  def cool2(e,b,c):
      img[0:260,0:800]=(b,0,e)
      for x in range(260,500):
          img[x,0:800]=(255-x,0,e)
      cv2.circle(img, (circleCenter_x,c),100,(0,0,255),-1)
  def cool3(e,c):
      u = 250
      img[0:e,0:800]=(250,0,255)
      for x in range(e,e+255):
          img[x,0:800]=(u,0,255)
          u = u-1
      img[e+250:500,0:800]=(0,0,255)
      cv2.circle(img, (circleCenter_x,c),100,(0,0,255),-1)
  
  while(o<255):
      cool2(o,b,c)
      cv2.imshow('Scene', img)
      o=o+1
      c = c+1
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break  
  
  l=245
  while l !=-250:
      cool3(l,c)
      cv2.imshow('Scene', img)
      l = l-1
      c = c+1
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

def happening(mood):
  def happy():
    print("Showing happy happening")
    # shows screen fading from white->yellow->white
    for x in range (0,255):
      img[:]=(255-x,255,255)
      cv2.imshow('Scene',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    for x in range (0,255):
      img[:]=(0+x,255,255)
      cv2.imshow('Scene',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  def sad():
    print("Showing sad happening")
    # shows screen fading from white->grey->white
    for x in range (0,200):
      img[:]=(255-x)
      cv2.imshow('Scene',img)
      time.sleep(0.01)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.5)
    for x in range (0,200):
      img[:]=(55+x)
      cv2.imshow('Scene',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  def angry():
    print("Showing angry happening")
    # shows screen flashing black and white 
    for x in range (100):
      img[:]= (0,0,0)
      cv2.imshow('Scene',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      time.sleep(0.05)
      img[:]= (255,255,255)
      cv2.imshow('Scene',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
  moods = [1,2,3]
  THEmood = moods[mood[0]]
  if THEmood == 1:
    happy()
  elif THEmood == 2:
    sad()
  else:
    angry()

def startProgram():
  wait_for_dancer()
  while True:
    class_list = classify_frame()
    if (len(class_list) != 25):
      break
    # if no pose has been detected for 25 frames, the program will end
    if (class_list == exit_frame):
      print("Dancer has left the stage")
      break   
    mood = classify_list(class_list)
    happening(mood)
  # sunset is the final happening that ends the program
  sunset()
  print("Exiting program")

startProgram()
cv2.destroyAllWindows()

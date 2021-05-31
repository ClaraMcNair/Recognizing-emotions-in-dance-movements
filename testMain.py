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

# test-video
testVideo = 'test200.mov'
cap = cv2.VideoCapture(testVideo)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

img = np.zeros((500,800,3), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX 

# exit frame is a list of 25 frames classified with 7 
# as no pose was detected in the frame
exit_frame = [7]*25

# Load data and target csv-files for the movement classifier and create the classifier
movement_data_filename = 'movement_data.csv'
movement_attributes = ['speed','aceleration','openness','x','y','a','b','c','d','e','f','g']
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

emotion_classifier = RandomForestClassifier()
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

def classify_25_frames(startingPoint):
  # class_list is used to store the predicted movements
  # runs counts every frame 
  class_list =[]
  runs = 0
  cap = cv2.VideoCapture(testVideo)
  cap.set(cv2.CAP_PROP_POS_FRAMES, startingPoint)

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
        cv2.putText(image, 'classified movement: '+ str(final_result),(20,70),font,2,(0,0,255),2,cv2.LINE_AA)
        class_list.append(final_result)
        
        # Finally show resulting frame
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      else:
        image = cv2.flip(image,1)
        cv2.putText(image, 'classified movement: ',(20,70),font,2,(0,0,255),2,cv2.LINE_AA)
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
              break
      runs = runs + 1
      startingPoint = startingPoint + 1
  
  cap.release()
  cv2.destroyAllWindows()
  return class_list, startingPoint

def classify_list(class_list):
  predicted_mood = emotion_classifier.predict([class_list])
  return predicted_mood

def wait_for_dancer():
  # This function shows a white screen while waiting for a pose to get detected 
  #img[:]=(255,255,255)
  frameCount = 0
  cap = cv2.VideoCapture(testVideo)

  with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        break

      # Flip the image horizontally
      image = cv2.flip(image, 1)
      cv2.putText(image, 'waiting for dancer',(20,70),font,2,(0,0,255),2,cv2.LINE_AA)

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = pose.process(image)
      
      # Check if frame has pose-landmarks 
      if hasattr(results.pose_landmarks, 'landmark'):
        break 
      else:
        cv2.imshow('test',image) 
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      frameCount = frameCount + 1 
    
    # Happening when pose has been detected 
    # screen changes from white->pink->white
    for x in range (255,0,-5):
      img[:]=(255,x,255)
      cv2.imshow('test',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    for x in range (0,255,5):
      img[:]=(255,x,255)
      cv2.imshow('test',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return frameCount

def sunset():
  # this function creates a screen 
  # with changing colors and shapes, simulating a sunset
  red = 0
  blue = 255
  sunCenter_y = -100
  sunCenter_x = 400
  
  def blueToPink(red,blue,sunCenter_y):
      img[0:260]=(blue,0,red)
      for x in range(260,500):
          img[x]=(255-x,0,red)
      # add red circle as sun
      cv2.circle(img, (sunCenter_x,sunCenter_y),100,(0,0,255),-1)
  
  def pinkToRed(horizonHeight,sunCenter_y):
      blue = 250
      img[0:horizonHeight]=(250,0,255)
      for x in range(horizonHeight,horizonHeight+255):
          img[x]=(blue,0,255)
          blue = blue-1
      img[horizonHeight+250:500]=(0,0,255)
      # add red circle as sun
      cv2.circle(img, (sunCenter_x,sunCenter_y),100,(0,0,255),-1)
  
  while(red<255):
      blueToPink(red,blue,sunCenter_y)
      cv2.imshow('Scene', img)
      red=red+1
      sunCenter_y = sunCenter_y+1
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break  
  
  horizonHeight=245
  while horizonHeight != -250:
      pinkToRed(horizonHeight,sunCenter_y)
      cv2.imshow('Scene', img)
      horizonHeight = horizonHeight-1
      sunCenter_y = sunCenter_y+1
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

def happening(mood):
  def happy():
    # shows screen fading from white->yellow->white
    for blue in range (255,0,-1):
      img[:]=(blue,255,255)
      cv2.putText(img, 'predicted mood: happy',(20,70),font,2,(0,0,255),2,cv2.LINE_AA)
      cv2.imshow('test',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    for blue in range (0,255):
      img[:]=(blue,255,255)
      cv2.putText(img, 'predicted mood: happy',(20,70),font,2,(0,0,255),2,cv2.LINE_AA)
      cv2.imshow('test',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  def sad():
    # shows screen fading from white->grey->white
    for x in range (0,200):
      img[:]=(255-x)
      cv2.putText(img, 'predicted mood: sad',(20,70),font,2,(0,0,255),2,cv2.LINE_AA)
      cv2.imshow('test',img)
      time.sleep(0.01)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.5)
    for x in range (0,200):
      img[:]=(55+x)
      cv2.putText(img, 'predicted mood: sad',(20,70),font,2,(0,0,255),2,cv2.LINE_AA)
      cv2.imshow('test',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  def angry():
     # shows screen flashing black and white 100 times
    for x in range (100):
      # black image 
      img[:]= (0,0,0)
      cv2.putText(img, 'predicted mood: angry',(20,70),font,2,(0,0,255),2,cv2.LINE_AA)
      cv2.imshow('test',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      time.sleep(0.05)
      # white image
      img[:]= (255,255,255)
      cv2.putText(img, 'predicted mood: angry',(20,70),font,2,(0,0,255),2,cv2.LINE_AA)
      cv2.imshow('test',img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
  currentMood = mood[0]
  if currentMood == 0:
    happy()
  elif currentMood == 1:
    sad()
  else:
    angry()

def startProgram():
  startingPoint = wait_for_dancer()
  while True:
    class_list, startingPoint = classify_25_frames(startingPoint)
    if (len(class_list) != 25):
      break
    # if no pose has been detected for 25 frames, the program will end
    if (class_list == exit_frame):
      break   
    mood = classify_list(class_list)
    happening(mood)
  # sunset is the final happening that ends the program
  sunset()

startProgram()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import csv
import os
from ProcessFrame import ProcessFrame

# This program creates a target and a data csv-file for classification  
# It takes video-files located in a specified folder
# Atteibutes are calculated for every third frame in each video using the class ProcessFrame

# Elements from MediaPipe's solution to predict landmarks are used in this file 
# copied from: https://google.github.io/mediapipe/solutions/pose


# Define names of data and target files in csv-format
# Define target label and path to folder with video-files 
csvDataFile = '_.csv'
csvTargetFile = '_.csv'
targetLabel = 'i'
folderName = 'hop'

processFrame = ProcessFrame()
getAttributes = ProcessFrame.predictAtributesInFrame

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
    # Read every video file in the given folder 
    for video in os.listdir(folderName):
        if video.endswith('.mov') or video.endswith('.MOV'):
            thefile = folderName+'/'+video 
            cap = cv2.VideoCapture(thefile)
            
            # Initialize varibles, 
            # frameNo is used to count every frame
            # runs is used to count every frame that gets processed 
            frameNo = 0
            runs = 0
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break

                # Only process every third frame, to have a lower fps, avg video has around 30fps
                if (frameNo % 3 == 0):
                    # Flip the image horizontally for a later selfie-view display, and convert
                    # the BGR image to RGB.
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False 
                    
                    # Get attributes for current frame 
                    data, results = getAttributes(processFrame,image,pose)
                    
                    # Ignore the first two processed frames 
                    if (runs>1):
                        # Add the list of data to the given CSV-data-file
                        with open (csvDataFile, 'a') as file_obj:
                            writer = csv.writer(file_obj)
                            writer.writerow(data)
                        # Add target label to the given CSV-target-file
                        with open (csvTargetFile, 'a') as file_obj:
                            writer = csv.writer(file_obj)
                            writer.writerow(targetLabel)
                         
                    runs = runs + 1
   
                    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    # Show final frame with landmarks                        
                    cv2.imshow('MediaPipe Pose', image)             
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                frameNo = frameNo+1
            
            cap.release()
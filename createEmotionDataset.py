import cv2
import mediapipe as mp
import pandas
import csv
from sklearn.ensemble import RandomForestClassifier
from ProcessFrame import ProcessFrame

# This program uses the movement classifier
# and predicts what movement is detected in each frame from the webcam-input 
# a series of 25 classified movement gets added to a given data-file 
# and an assigned class label gets added to a given target-file

# Elements from MediaPipe's solution to predict landmarks are used in this file 
# copied from: https://google.github.io/mediapipe/solutions/pose

# Define names of data and target files in csv-format
# Define target label  
csvDataFile = 'test1.csv'
csvTargetFile = 'test2.csv'
class_label = '2'

processFrame = ProcessFrame()
getAttributes = ProcessFrame.predictAtributesInFrame

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load data and target csv-files for the movement classifier and create the classifier
data_filename = 'movement_data.csv'
attribute_names = ['speed','aceleration','x_dir','y_dir','feet_hip_dist',
                    'hands_shoulder_dist','rHand_lHand_dist','rFoot_lFoot_dist',
                    'hand_speed','feet_speed','hand_acc','feet_acc']
data = pandas.read_csv(data_filename, names=attribute_names)

target_filename = 'movement_target.csv'
target_name = ['class']
target = pandas.read_csv(target_filename, names=target_name)

movement_classifier = RandomForestClassifier()
movement_classifier.fit(data,target)

def predict_pose(test_data):
  # Predict pose unless test_data is empty
  # and return the pose's class-label
  # 7 indicates that no pose was found
  if not test_data:
    pose_name = '7'
  else:
    all_poses = ['0','1','2','3','4','5','6']
    predicted_pose = movement_classifier.predict([test_data])
    pose_name = all_poses[(predicted_pose[0])]
  return pose_name


def predict_25_frames():
    
    # class_list is used to store the predicted movements
    # runs counts every frame 
    class_list = []
    runs = 0
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        # read 75 frames, since only every thrid frame is processed, 
        # it will add up to 25 processed frames
        while cap.isOpened() and runs <= 75:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            # Process only every third frame  
            if (success and runs % 3 == 0):
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
                font = cv2.FONT_HERSHEY_SIMPLEX  
                cv2.putText(image, final_result,(20,70),font,2,(0,0,255),2,cv2.LINE_AA)
                
                # Finally show resulting frame
                cv2.imshow('MediaPipe Pose', image)
                # Ignore first two processed frames
                if (runs>1):
                    class_list.append(final_result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            runs = runs + 1
            
        cap.release()
        cv2.destroyAllWindows()
        return class_list

def load_list_to_dataset(class_list):
    # Add the list to the given CSV-data-file 
    with open (csvDataFile, 'a') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(class_list)
    # Add class label to the given CSV-target-file
    with open (csvTargetFile, 'a') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(class_label)    

def createEmotionDataset():
    while (True):
        class_list = predict_25_frames()        
        if (len(class_list) != 25):
            break
        load_list_to_dataset(class_list)
               
createEmotionDataset()
cv2.destroyAllWindows
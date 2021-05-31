# Recognizing emotions in dance movements

This project aims to examine how a program can detect emotions in a user’s dance movements, and how this can be utilized in an interactive scenography. The project utilizes machine-learning to predict which emotion is portrayed through the user’s dance moves via a live video-feed. Each classification of the user’s emotion is set to trigger a corresponding graphical animation. MediaPipe’s Pose-model serves as the basis for the whole program by extracting 33 key points on the human body on a given image.  Rudolf Laban’s theories on movement are used as an essential part of the feature-engineering of the machine-learning models’ features, as it provides tangible ways to measure body movements. The project creates two random forest classifiers. The first one classifies one of seven different movements. A list of 25 classified movements, can be classified by the second classifier, which predicts one of three different emotions: happy, sad and angry. The program may produce a generalized prediction of the user’s movements but manages to create an interactive relation between the user and the scenography's visual appearance. 

 ###  EPILEPSY WARNING ###

The happening 'angry' will display a sequence of fast flashing lights. User discretion is advised.

### Required pip-modules ###

The user must have installed the following pip-modules in order for the program to run on their computer:

mediapipe

opencv-python 

pandas 1.2.4

numpy

scikit-learn

scipy

### User-guide ###
 
In order for this program to run, the user must have a working webcam. If you have a external webcam, please change variable  :  `webcamSource`to 1 in main.py, line 25

Step 1: Clone this repository

Step 2: Set up the webcam so that the user can stand 3-4 meters from the webcam, so that the whole body is in view in the camera frame. The user must remain within 3-4 meters from the webcam.

Step 3: Run the program på running the file :  `main.py`

Step 4: Have the user step into the frame. When the screen flashes pink, the users pose has been detected.

Step 5: Dance!

To exit the program: Press 'Q' or leave the frame.


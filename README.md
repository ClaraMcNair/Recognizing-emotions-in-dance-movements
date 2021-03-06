# Recognizing emotions in dance movements

This project aims to examine how a program can detect emotions in a user’s dance movements, and how this can be utilized in an interactive scenography. The project utilizes machine-learning to predict which emotion is portrayed through the user’s dance moves via a live video-feed. Each classification of the user’s emotion is set to trigger a corresponding graphical animation. MediaPipe’s Pose-model serves as the basis for the whole program by extracting 33 key points on the human body on a given image. The project creates two random forest classifiers. The first one classifies one of seven different movements. A list of 25 classified movements, can be classified by the second classifier, which predicts one of three different emotions: happy, sad and angry. The program may produce a generalized prediction of the user’s movements but manages to create an interactive relation between the user and the program's visual appearance. 

 ###  EPILEPSY WARNING ###

The happening 'angry' will display a sequence of fast flashing lights. User discretion is advised.

### Required pip-modules ###

The user must have installed the following pip-modules in order for the program to run on their computer:

mediapipe

opencv-python 

pandas 

numpy

scikit-learn

scipy

matplotlib

### User-guide ###
 
The program can only be used by one person at a time.
In order for this program to run, the user must have a working webcam. If you have an external webcam, please change variable  :  `webcamSource` (main.py, line 25) away from 0 so that it refers to the external camera.

Step 1: Clone this repository

Step 2: Set up the webcam so that the user can stand 3-4 meters from the webcam, so that the whole body is in view of the camera. The user must remain within 3-4 meters from the webcam.

Step 3: Run the program by running the file :  `main.py`

Step 4: Have the user step into the frame. When the screen flashes pink, the user's pose has been detected.

Step 5: Dance!

To exit the program: Press 'Q' or leave the frame.


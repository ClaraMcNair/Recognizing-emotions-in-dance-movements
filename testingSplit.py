
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Load data and target csv-files for the movement classifier 
movement_data_filename  = 'movement_data.csv'
movement_attributes = ['speed','aceleration','x_dir','y_dir','feet_hip_dist',
        'hands_shoulder_dist','rHand_lHand_dist','rFoot_lFoot_dist',
        'hand_speed','feet_speed','hand_acc','feet_acc']
movement_data = pandas.read_csv(movement_data_filename, names=movement_attributes)

movement_target_filename = 'movement_target.csv'
movement_labels= ['class']
movement_target = pandas.read_csv(movement_target_filename, names=movement_labels)


# Load data and target csv-files for the emotion classifier with 25 frames
emotion_data_filename = 'emotion_data.csv'
emotion_attributes = ['1','2','3','4','5','6','7','8','9','10','11','12',
                        '13','14','15','16','17','18','19','20','21','22','23','24','25']
emotion_25_data = pandas.read_csv(emotion_data_filename, names=emotion_attributes)

emotion_target_filename = 'emotion_target.csv'
emotion_labels = ['class']
emotion_25_target = pandas.read_csv(emotion_target_filename, names=emotion_labels)


#Load data and target csv-files for the emotion classifier with 50 frames 
emotion_50_datafile = 'emotion_data_50frames.csv'
emotion_50_attr = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15',
                    '16','17','18','19','20','21','22','23','24','25','26','27','28',
                    '29','30','31','32','33','34','35','36','37','38','39','40','41',
                    '42','43','44','45','46','47','48','49','50']
emotion_50_data = pandas.read_csv(emotion_50_datafile, names=emotion_50_attr)

emotion_50_targetfile = 'emotion_target_50frames.csv'
emotion_50_label = ['class']
emotion_50_target = pandas.read_csv(emotion_50_targetfile , names=emotion_50_label)


# defining the six different classifiers

movement_classifier = RandomForestClassifier(random_state=0)
movement_KNN= KNeighborsClassifier()

emotion_classifier = RandomForestClassifier(random_state=0)
emotion_KNN= KNeighborsClassifier()

emotion_50_classifier = RandomForestClassifier(random_state=0)
emotion_50_KNN= KNeighborsClassifier()

#Splitting the data
#Using built-in-function 'train_test_split' from scikit
#test_size 0.2 means that 20% will go to testing, the remaining 80% is for training
movement_data_train, movement_data_test, movement_target_train, movement_target_test = train_test_split(movement_data, movement_target, test_size=0.2, random_state=0)
emotion_25_data_train, emotion_25_data_test, emotion_25_target_train, emotion_25_target_test = train_test_split(emotion_25_data, emotion_25_target, test_size=0.2, random_state=0)
emotion_50_data_train, emotion_50_data_test, emotion_50_target_train, emotion_50_target_test = train_test_split(emotion_50_data, emotion_50_target, test_size=0.2, random_state=0)


#Fitting the models using the training-set
movement_classifier.fit(movement_data_train, movement_target_train)
movement_KNN.fit(movement_data_train, movement_target_train)

emotion_classifier.fit(emotion_25_data_train, emotion_25_target_train)
emotion_KNN.fit(emotion_25_data_train, emotion_25_target_train)

emotion_50_classifier.fit(emotion_50_data_train, emotion_50_target_train)
emotion_50_KNN.fit(emotion_50_data_train, emotion_50_target_train)


#Showing the models accuracy(in percent) using built-in-function 'score'
# The test-sets are given as arguments. The predicted class of the test-set is compared to the actual class from the targetfile. 
print('Accuracy for Movement classifier using Random Forest: ', movement_classifier.score(movement_data_test, movement_target_test))
print('Accuracy for Movement classifier using KNN: ', movement_KNN.score(movement_data_test, movement_target_test))

print('Accuracy for Emotion classifier with 25 frames using Random Forest: ', emotion_classifier.score(emotion_25_data_test, emotion_25_target_test))
print('Accuracy for Emotion classifier with 25 frames using KNN: ', emotion_KNN.score(emotion_25_data_test, emotion_25_target_test))

print('Accuracy for Emotion classifier with 50 frames using Random Forest: ', emotion_50_classifier.score(emotion_50_data_test,emotion_50_target_test))
print('Accuracy for Emotion classifier with 50 frames using KNN: ', emotion_50_KNN.score(emotion_50_data_test,emotion_50_target_test))

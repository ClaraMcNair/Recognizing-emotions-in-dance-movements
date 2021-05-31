import statistics
from scipy.spatial import distance

class ProcessFrame:
    # Initialize class variables 
    prev_pose = [0] * 66 
    prev_speed = 0.0
    prev_avgY = 0.5
    prev_avgX = 0.5
    prev_rHand = [0,0]
    prev_lHand = [0,0]
    prev_rFoot = [0,0]
    prev_lFoot = [0,0]
    prev_hand_speed = 0
    prev_feet_speed = 0
    
    def predictAtributesInFrame(self,image,pose):
        
        results = pose.process(image)

        # Initialize empty lists every time a new frame is proccesed 
        allpoints = []
        all_X_coords = []
        all_Y_coords = []
        data = []
        
        if hasattr(results.pose_landmarks, 'landmark'): 
            # Add all x- and y-coordinates from each pose-landmark on the image to the list allpoints
            # and add all x- and y-coordinates to the lists all_X_coords and all_Y_coords
            for data_point in results.pose_landmarks.landmark:
                allpoints.append(data_point.x)
                allpoints.append(data_point.y)
                all_Y_coords.append(data_point.y)
                all_X_coords.append(data_point.x)

            # calculate the average of x- and y-coordinates
            avgX = statistics.mean(all_X_coords)
            avgY = statistics.mean(all_Y_coords)

            # Define coordinates for both feet and hands 
            landmarks = results.pose_landmarks.landmark
            rHand = [landmarks[16].x,landmarks[16].y]
            lHand = [landmarks[15].x,landmarks[15].y]
            rFoot = [landmarks[28].x,landmarks[28].y]
            lFoot = [landmarks[27].x,landmarks[27].y]
            
            # Calculate average full body speed
            speed = distance.euclidean(allpoints,self.prev_pose)
            
            # Calculate average speed for hands
            lHand_speed = distance.euclidean(lHand, self.prev_lHand)
            rHand_speed = distance.euclidean(rHand, self.prev_rHand)
            hand_speed = (lHand_speed + rHand_speed)/2
            
            # Calculate hand acceleration
            hand_acc = hand_speed - self.prev_hand_speed
               
            # Calculate average speed for feet
            lFoot_speed = distance.euclidean(lFoot, self.prev_lFoot)
            rFoot_speed = distance.euclidean(rFoot, self.prev_rFoot)
            feet_speed = (lFoot_speed + rFoot_speed)/2
            
            # Calculate feet acceleration
            feet_acc = feet_speed - self.prev_feet_speed
            
            # Calcuate accelaration for the full body
            acceleration = speed - self.prev_speed
            
            # Calculate average distance between hands and shoulders
            lHand_to_hip = distance.euclidean(lHand,(landmarks[11].x,landmarks[11].y))
            rHand_to_hip = distance.euclidean(rHand,(landmarks[12].x,landmarks[12].y))
            hand_shoulder_distance = (lHand_to_hip + rHand_to_hip)/2
            
            # Calculate average distance between feet and hips 
            lFoot_to_hip = distance.euclidean(lFoot,(landmarks[23].x,landmarks[23].y))
            rFoot_to_hip = distance.euclidean(rFoot,(landmarks[24].x,landmarks[24].y))
            feet_hip_distance = (lFoot_to_hip + rFoot_to_hip)/2
            
            # Calculate distance between hands 
            rHand_lHand_distance = distance.euclidean(lHand,rHand)
            
            # Calculate distance between feet
            rFoot_lFoot_distance = distance.euclidean(lFoot,rFoot)
            
            # Calculate the average movement direction for the full body on both the x- and y-axis
            y_dir = avgY - self.prev_avgY
            x_dir = avgX - self.prev_avgX 
            
            # Add all the above calculated data to the data list
            data.extend([speed, acceleration, x_dir, y_dir, feet_hip_distance, 
                        hand_shoulder_distance, rHand_lHand_distance, rFoot_lFoot_distance, 
                        hand_speed, feet_speed, hand_acc, feet_acc])
        
            # Update the previous values
            self.prev_pose = allpoints 
            self.prev_speed = speed
            self.prev_avgY = avgY
            self.prev_avgX = avgX
            self.prev_feet_speed = feet_speed
            self.prev_hand_speed = hand_speed
            self.prev_rHand = rHand
            self.prev_lHand = lHand
            self.prev_rFoot = rFoot
            self.prev_lFoot = lFoot
        
        return data, results

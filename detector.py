import cv2
import mediapipe as mp
class Detector:
    def __init__(self, static_mode = False, model_complex = 1, smooth_landmark = True, segmentation = False,
    smooth_segment = False, min_detect_conf = 0.6, min_track_conf = 0.6):
        self.static_mode = static_mode
        self.model_complex = model_complex
        self.smooth_landmark = smooth_landmark
        self.segmentation = segmentation
        self.smooth_segment = smooth_segment
        self.min_detect_conf = min_detect_conf
        self.min_track_conf = min_track_conf

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(self.static_mode, self.model_complex, self.smooth_landmark,
        self.segmentation, self.smooth_segment, 
        self.min_detect_conf, self.min_track_conf
        )

        


    def get_pose(self, img, draw = True):
        from mediapipe.framework.formats import landmark_pb2

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.res = self.pose.process(img)
        self.landmarks = self.res.pose_landmarks
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.pose_connections = self.mp_pose.POSE_CONNECTIONS

        #Only display the shoulders and part of the arms
        self.landmark_subset = landmark_pb2.NormalizedLandmarkList(
            landmark = [
                self.landmarks.landmark[11],
                self.landmarks.landmark[12],
                self.landmarks.landmark[13],
                self.landmarks.landmark[14]
            ]
        )
        if self.landmarks:
            if draw:

                #Draws only the shoulders and part of the arms
                #TODO: Figure out how to draw the lines back in without error                
                self.mp_draw.draw_landmarks(img, 
                    self.landmark_subset, 
                    # self.pose_connections,
                    landmark_drawing_spec= self.mp_drawing_styles.get_default_pose_landmarks_style())
                # self.mp_draw.draw_landmarks(img, self.res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                #Backup line
                # self.mp_draw.draw_landmarks(img, 
                #     self.landmarks, 
                #     self.pose_connections,
                #     landmark_drawing_spec= self.mp_drawing_styles.get_default_pose_landmarks_style())


        return img

    def detect_sitting(self, img):
        #TODO: Work on classification model/data
        pass


if __name__ == 'main':
    def main():
        cap = cv2.VideoCapture(0)
        detector = Detector()

        while cap.isOpened():
            succ, img = cap.read()
            if not succ:
                print("Check camera, no output")
                continue
            
            cv2.imshow("Posture Detector", detector.get_pose(img))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        
    main()
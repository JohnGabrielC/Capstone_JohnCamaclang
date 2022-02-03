from detector import Detector
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_mp2

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
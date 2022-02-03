from numpy import save
from detector import Detector
import cv2
import mediapipe as mp
import os

def save_image(counter, img):
    path = "C:/Users/John/Documents/School/Capstone"
    cv2.imwrite(os.path.join(path, "frame_%d.jpg" % counter), img)
    print("frame_%d" % counter + " saved")

def main():
    cap = cv2.VideoCapture(0)
    detector = Detector()
    count = 0
    while cap.isOpened():
        succ, img = cap.read()
        if not succ:
            print("Check camera, no output")
            break

        cv2.imshow("Posture Detector", detector.get_pose(img))
        if cv2.waitKey(5) & 0xFF == 27:
            break
        #Check if key is either spacebar or esc
        a = cv2.waitKey(5)
        if a % 256 == 32:
            save_image(count, detector.get_pose(img))
            count += 1

    cap.release()
    cv2.destroyAllWindows
main()
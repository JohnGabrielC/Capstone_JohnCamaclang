from numpy import save
from detector import Detector
import cv2
import mediapipe as mp
import os
def main():
    d = Detector()
    d.get_pose()
main()
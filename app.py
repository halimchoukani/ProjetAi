import csv
import argparse
import math
import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui
import keyboard
from utils import CvFpsCalc
from model import KeyPointClassifier
import itertools

# Zoom functions
def resetZoom():
    pyautogui.hotkey('ctrl', '0')

def zoom(zoom_in=True):
    keyboard.press('ctrl')
    pyautogui.scroll(100 if zoom_in else -100)
    keyboard.release('ctrl')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()

def main():
    args = get_args()
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    
    keypoint_classifier = KeyPointClassifier()    
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    prev_distance, prev_index_y = None, None
    frame_skip_counter = 0

    while cap.isOpened():
        frame_skip_counter += 1
        if frame_skip_counter % 2 != 0:
            continue

        fps = cvFpsCalc.get()
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                h, w, _ = image.shape
                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

                landmark_list = calc_landmark_list(hand_landmarks, image)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                action = keypoint_classifier_labels[hand_sign_id]
                if action == "Open":
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if prev_distance is None:
                        prev_distance = distance

                    if abs(distance - prev_distance) > 4:
                        zoom(zoom_in=distance > prev_distance)
                    prev_distance = distance

                elif action == "Close":
                    resetZoom()

                elif action == "Pointer":
                    if prev_index_y is None:
                        prev_index_y = y2
                    else:
                        scroll_amount = int(prev_index_y - (args.height / 2))
                        pyautogui.scroll(scroll_amount)
                        prev_index_y = y2

        cv.imshow('Hand Gesture Recognition', image)
        if cv.waitKey(1) & 0xFF == 27:  # ESC key to break
            break

    cap.release()
    cv.destroyAllWindows()


def calc_landmark_list(landmarks, image):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[min(int(landmark.x * image_width), image_width - 1),
             min(int(landmark.y * image_height), image_height - 1)] for landmark in landmarks.landmark]

def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    normalized_landmarks = [(x - base_x, y - base_y) for x, y in landmark_list]
    max_value = max(map(lambda n: abs(n), itertools.chain(*normalized_landmarks)))
    return [n / max_value for n in itertools.chain(*normalized_landmarks)]

if __name__ == '__main__':
    main()

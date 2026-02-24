import cv2
import mediapipe as mp
import pyautogui
import math
import time

pyautogui.FAILSAFE = False

# Screen and Camera Resolution
w_cam, h_cam = 640, 480
w_screen, h_screen = pyautogui.size()

frame_r = 100 
smoothening = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

def map_val(val, in_min, in_max, out_min, out_max):
    return (val - in_min) * (out_max - out_min) / (in_max - in_min)

prev_scroll_y = 0
prev_scroll_x = 0
scroll_speed = 30

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Initialize MediaPipe Tasks HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=VisionRunningMode.VIDEO
)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)

last_click_time = 0
click_cooldown = 0.5 # Seconds
last_timestamp_ms = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally for a selfie-view display
    img = cv2.flip(img, 1)
    
    # Convert the BGR image to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # Calculate timestamp in milliseconds
    current_ms = int(time.time() * 1000)
    if current_ms <= last_timestamp_ms:
        current_ms = last_timestamp_ms + 1
    last_timestamp_ms = current_ms
    
    # Detect hands
    hand_landmarker_result = detector.detect_for_video(mp_image, current_ms)
    
    cv2.rectangle(img, (frame_r, frame_r), (w_cam - frame_r, h_cam - frame_r), (255, 0, 255), 2)

    if hand_landmarker_result.hand_landmarks:
        for hand_landmarks in hand_landmarker_result.hand_landmarks:
            h, w, c = img.shape
            
            lm_list = []
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                # Draw small circles on landmarks
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
                
            if len(lm_list) != 0:
                thumb_tip = (lm_list[4][1], lm_list[4][2])
                index_tip = (lm_list[8][1], lm_list[8][2])
                middle_tip = (lm_list[12][1], lm_list[12][2])
                
                # Check which fingers are up
                fingers = []
                # Thumb (approximate based on x coordinate for open hand)
                if lm_list[4][1] < lm_list[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                
                # Other fingers
                for tip_id, pip_id in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                    if lm_list[tip_id][2] < lm_list[pip_id][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Cursor Movement
                x3 = int(map_val(index_tip[0], frame_r, w_cam - frame_r, 0, w_screen))
                y3 = int(map_val(index_tip[1], frame_r, h_cam - frame_r, 0, h_screen))
                
                # Clamp coordinates to screen size
                x3 = max(0, min(x3, w_screen))
                y3 = max(0, min(y3, h_screen))

                # Smooth Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # Scrolling mode: Index and Middle are up, others down
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                    cv2.circle(img, index_tip, 15, (0, 255, 0), cv2.FILLED)
                    cv2.circle(img, middle_tip, 15, (0, 255, 0), cv2.FILLED)
                    avg_y = (index_tip[1] + middle_tip[1]) / 2
                    avg_x = (index_tip[0] + middle_tip[0]) / 2
                    
                    if prev_scroll_y != 0:
                        diff_y = prev_scroll_y - avg_y
                        diff_x = avg_x - prev_scroll_x
                        
                        # Threshold to avoid jitters
                        if abs(diff_y) > 5:
                            pyautogui.scroll(int(diff_y * scroll_speed))
                        
                        if abs(diff_x) > 5:
                            pyautogui.hscroll(int(diff_x * scroll_speed))
                            
                    prev_scroll_y = avg_y
                    prev_scroll_x = avg_x
                else:
                    prev_scroll_y = 0
                    prev_scroll_x = 0
                    
                    # Normal Cursor Movement
                    pyautogui.moveTo(clocX, clocY)
                    plocX, plocY = clocX, clocY

                # Left Click
                distance_left = get_distance(index_tip, thumb_tip)
                if distance_left < 30 and time.time() - last_click_time > click_cooldown:
                    cv2.circle(img, (int((index_tip[0] + thumb_tip[0])//2), int((index_tip[1] + thumb_tip[1])//2)), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()
                    last_click_time = time.time()
                
                # Right Click
                distance_right = get_distance(middle_tip, thumb_tip)
                if distance_right < 30 and distance_left > 40 and time.time() - last_click_time > click_cooldown:
                    cv2.circle(img, (int((middle_tip[0] + thumb_tip[0])//2), int((middle_tip[1] + thumb_tip[1])//2)), 15, (255, 0, 0), cv2.FILLED)
                    pyautogui.click(button='right')
                    last_click_time = time.time()

    cv2.imshow("Hand Mouse Controller", img)
    if cv2.waitKey(1) & 0xFF == 27: # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

CAM_W, CAM_H = 320, 240
ROI_X = 0
MIN_AREA = 800
KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

def nothing(_):
    pass

def setup_trackbars():
    cv2.namedWindow("Callibration")
    for name, val in [("H min", 75), ("H max", 105),
                      ("S min", 120), ("S max", 255),
                      ("V min", 70), ("V max", 255)]:
        cv2.createTrackbar(name, "Callibration", val, 255, nothing)
    cv2.resizeWindow("Callibration", 400, 250)

def get_range():
    g = lambda n: cv2.getTrackbarPos(n, "Callibration")
    lower = np.array([g("H min"), g("S min"), g("V min")], np.uint8)
    upper = np.array([g("H max"), g("S max"), g("V max")], np.uint8)
    return lower, upper

def track_hand(frame, lower, upper):
    roi = frame[:, ROI_X:]
    blur = cv2.GaussianBlur(roi, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + 90) % 180).astype(np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE, iterations=2)
    mask = cv2.medianBlur(mask, 5)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask, None
    
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None, mask, None
    
    c = cv2.convexHull(c)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, mask, None
    
    cy = M["m01"] / M["m00"]
    return cy, mask, c

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera cannot be opened")
        return
    
    setup_trackbars()
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame = cv2.flip(cv2.resize(frame, (CAM_W, CAM_H)), 1)
        
        lower, upper = get_range()
        cy, mask_roi, cnt = track_hand(frame, lower, upper)
        
        view = frame.copy()
        cv2.rectangle(view, (ROI_X, 0), (CAM_W - 1, CAM_H - 1), (0, 255, 0), 1)
        
        if cnt is not None:
            cnt_full = cnt + np.array([[[ROI_X, 0]]])
            cv2.drawContours(view, [cnt_full], -1, (0, 0, 255), 2)
            cx = int(cv2.moments(cnt)["m10"] / cv2.moments(cnt)["m00"])
            cv2.circle(view, (ROI_X + cx, int(cy)), 5, (255, 0, 0), -1)
        
        full_mask = np.zeros((CAM_H, CAM_W), np.uint8)
        full_mask[:, ROI_X:] = mask_roi
        
        mask_bgr = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([view, mask_bgr])
        cv2.imshow("Debug", combined)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
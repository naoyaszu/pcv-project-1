import cv2
import numpy as np

CAM_W, CAM_H = 320, 240

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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + 90) % 180).astype(np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([frame, mask_bgr])
        cv2.imshow("Debug", combined)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import numpy as np

CAM_W, CAM_H = 320, 240

def nothing(_):
    pass

def setup_trackbars():
    cv2.namedWindow("Callibration")
    # nilai awal umum untuk kulit di YCrCb
    for name, val in [("Y min", 0), ("Y max", 255),
                      ("Cr min", 133), ("Cr max", 173),
                      ("Cb min", 77), ("Cb max", 127)]:
        cv2.createTrackbar(name, "Callibration", val, 255, nothing)
    cv2.resizeWindow("Callibration", 400, 250)

def get_range():
    g = lambda n: cv2.getTrackbarPos(n, "Callibration")
    lower = np.array([g("Y min"), g("Cr min"), g("Cb min")], np.uint8)
    upper = np.array([g("Y max"), g("Cr max"), g("Cb max")], np.uint8)
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
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, lower, upper)
        
        # Tampilkan side-by-side
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([frame, mask_bgr])
        cv2.imshow("Debug", combined)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
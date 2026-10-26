import cv2
import numpy as np
import random
import time
import os

CAM_W, CAM_H = 320, 240          
ROI_X = 0                      
GAME_W, GAME_H = 640, 480
SHIP_X, SHIP_W, SHIP_H = 80, 100, 60
ALPHA = 0.4
MAX_STEP = 40                     
MIN_AREA = 800                   
Y_RANGE = (30, 210)              
MAX_LIVES = 5
HEART_SIZE = 32
BORDER = 6          
TOP_BAR = 56         
PLAY_TOP = TOP_BAR
PLAY_BOTTOM = GAME_H - BORDER
BTN_R = 40

KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from PIL import Image, ImageDraw, ImageFont
FONT_BIG = ImageFont.truetype(os.path.join(BASE_DIR, "assets/vcr_osd_mono.ttf"), 56)
FONT_MID = ImageFont.truetype(os.path.join(BASE_DIR, "assets/vcr_osd_mono.ttf"), 26)
FONT_SMALL = ImageFont.truetype(os.path.join(BASE_DIR, "assets/vcr_osd_mono.ttf"), 22)

def draw_text(img, text, xy, font, color=(255, 255, 255), anchor="la"):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=font,
                             fill=(color[2], color[1], color[0]), anchor=anchor)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def load_sprite(filename, size=None):
    path = os.path.join(BASE_DIR, filename)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED) 
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def overlay(bg, sprite, x, y):
    h, w = sprite.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, bg.shape[1]), min(y + h, bg.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = s[:, :, 3:4] / 255.0
    roi = bg[y0:y1, x0:x1]
    roi[:] = (alpha * s[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)

SHIP_IMG = load_sprite("assets/ship.png", (SHIP_W, SHIP_H))
GOOD_IMG = load_sprite("assets/obstacle1.png")   
BAD_IMG = load_sprite("assets/obstacle2.png")    
HEART_FULL = load_sprite("assets/heart.png", (HEART_SIZE, HEART_SIZE))
HEART_EMPTY = load_sprite("assets/heart_empty.png", (HEART_SIZE, HEART_SIZE))
BG_IMG = cv2.imread(os.path.join(BASE_DIR, "assets/background.png"))
BG_IMG = cv2.resize(BG_IMG, (GAME_W, GAME_H), interpolation=cv2.INTER_AREA)
BG_IMG = (BG_IMG * 0.6).astype(np.uint8)
BONUS_POINTS = 50

#hsv
LOWER_HSV = np.array([75, 120, 70], np.uint8)
UPPER_HSV = np.array([105, 255, 255], np.uint8)

def track_hand(frame, lower, upper):
    roi = frame[:, ROI_X:]
    blur = cv2.GaussianBlur(roi, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + 90) % 180).astype(np.uint8)  # merah jadi ~90
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE, iterations=2)
    mask = cv2.medianBlur(mask, 5)   # haluskan tepi bergerigi

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


class Game:
    def __init__(self):
        self.started = False
        self.mouse = (-1, -1)
        self.reset()

    def reset(self):
        self.ship_y = GAME_H / 2
        self.bullets = []                 # [x, y, w, h, speed, img, kind]
        self.clock = 0.0                  # jam game, berhenti saat pause
        self.last_tick = time.time()
        self.last_spawn = 0.0
        self.bonus = 0
        self.paused = False
        self.over = False
        self.final_score = 0
        self.lives = MAX_LIVES
        self.last_hit = -10.0              

    @property
    def score(self):
        if self.over:
            return self.final_score
        return int(self.clock * 10) + self.bonus

    def toggle_pause(self):
        if self.started and not self.over:
            self.paused = not self.paused

    def on_button(self, x, y):
        cx, cy = GAME_W // 2, (GAME_H + TOP_BAR) // 2
        return abs(x - cx) <= BTN_R and abs(y - cy) <= BTN_R

    def update(self, target_y):
        now = time.time()
        dt = now - self.last_tick
        self.last_tick = now              
        if self.over or self.paused or not self.started:
            return
        self.clock += dt
        t = self.clock

        if not np.isfinite(self.ship_y):
            self.ship_y = GAME_H / 2
        if target_y is not None and np.isfinite(target_y):
            diff = target_y - self.ship_y
            self.ship_y += float(np.clip(diff * 0.5, -MAX_STEP, MAX_STEP))
        self.ship_y = float(np.clip(self.ship_y, PLAY_TOP + SHIP_H / 2, PLAY_BOTTOM - SHIP_H / 2))
        interval = max(0.5, 1.0 - t * 0.005)
        speed_base = 6 + min(t * 0.05, 8)
        if t - self.last_spawn > interval and len(self.bullets) < 9:
            self.last_spawn = t
            kind = "good" if random.random() < 0.4 else "bad"   # 40% poin, 60% bahaya
            base = GOOD_IMG if kind == "good" else BAD_IMG
            h = random.randint(30, 60)
            w = int(h * base.shape[1] / base.shape[0])
            img = cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA)
            y = random.randint(PLAY_TOP, PLAY_BOTTOM - h)
            self.bullets.append([GAME_W + random.randint(0, 150), y, w, h,
                                 speed_base + random.uniform(0, 3), img, kind])

        ship = (SHIP_X - SHIP_W // 2 + 8, self.ship_y - SHIP_H / 2 + 5,
                SHIP_W - 16, SHIP_H - 10)
        alive = []
        for b in self.bullets:
            b[0] -= b[4]
            if self.collide(ship, (b[0], b[1], b[2], b[3])):
                if b[6] == "good":
                    self.bonus += BONUS_POINTS
                    self.lives = min(MAX_LIVES, self.lives + 1) 
                    continue          
                self.lives -= 1
                self.last_hit = self.clock
                if self.lives <= 0:
                    self.lives = 0
                    self.final_score = self.score
                    self.over = True
                continue
            if b[0] + b[2] > 0:       
                alive.append(b)
        self.bullets = alive

    @staticmethod
    def collide(a, b):
        return (a[0] < b[0] + b[2] and a[0] + a[2] > b[0] and
                a[1] < b[1] + b[3] and a[1] + a[3] > b[1])
    def draw(self):
        img = BG_IMG.copy()
        x, y = SHIP_X, int(self.ship_y)
        blink = (self.clock - self.last_hit < 0.6) and int(self.clock * 12) % 2 == 0
        if not blink:
            overlay(img, SHIP_IMG, x - SHIP_W // 2, y - SHIP_H // 2)
        # peluru
        for bx, by, bw, bh, _, spr, _kind in self.bullets:
            overlay(img, spr, int(bx), int(by))

        white = (255, 255, 255)
        cv2.rectangle(img, (0, 0), (GAME_W, TOP_BAR), white, -1)
        cv2.rectangle(img, (0, 0), (BORDER, GAME_H), white, -1)
        cv2.rectangle(img, (GAME_W - BORDER, 0), (GAME_W, GAME_H), white, -1)
        cv2.rectangle(img, (0, GAME_H - BORDER), (GAME_W, GAME_H), white, -1)

        draw_text(img, f"Score: {self.score}", (16, TOP_BAR // 2), FONT_MID, (25, 25, 25), "lm")
        for i in range(MAX_LIVES):
            spr = HEART_FULL if i >= MAX_LIVES - self.lives else HEART_EMPTY
            overlay(img, spr, GAME_W - 10 - (MAX_LIVES - i) * (HEART_SIZE + 6), 12)
        if self.over:
            draw_text(img, "GAME OVER", (180, 230), FONT_BIG, (0, 0, 255))
            draw_text(img, "Press R to start again", (180, 280), FONT_SMALL, (255, 255, 255))
        if self.paused and not self.over:
            cx, cy = GAME_W // 2, (GAME_H + TOP_BAR) // 2
            draw_text(img, "PAUSED", (cx, cy - 25), FONT_BIG, (0, 255, 255), "mm")
            draw_text(img, "Press P to resume", (cx, cy + 30), FONT_SMALL, (255, 255, 255), "mm")

        if not self.started:
            img[TOP_BAR:] = (img[TOP_BAR:] * 0.35).astype(np.uint8) 
            cx, cy = GAME_W // 2, (GAME_H + TOP_BAR) // 2
            s = 1.15 if self.on_button(*self.mouse) else 1.0      
            h, w = BTN_R * 0.8 * s, BTN_R * 1.4 * s                 
            tri = np.array([[cx - w / 3, cy - h],
                            [cx - w / 3, cy + h],
                            [cx + 2 * w / 3, cy]], np.int32)
            cv2.fillPoly(img, [tri], (255, 255, 255), cv2.LINE_AA)
        return img
    
def on_mouse(event, x, y, flags, game):
    game.mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN and not game.started and game.on_button(x, y):
        game.started = True

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera cannot be opened")
        return
    
    game = Game()
    cv2.namedWindow("404 Ufo")
    cv2.setMouseCallback("404 Ufo", on_mouse, game)
    smooth_y = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(cv2.resize(frame, (CAM_W, CAM_H)), 1)  # efek cermin

        lower, upper = LOWER_HSV, UPPER_HSV
        cy, mask_roi, cnt = track_hand(frame, lower, upper)

        target = None
        if cy is not None and np.isfinite(cy):
            if smooth_y is None:
                smooth_y = cy
            elif abs(cy - smooth_y) > 2:
                smooth_y = ALPHA * cy + (1 - ALPHA) * smooth_y
        if smooth_y is not None:
            target = float(np.interp(smooth_y, Y_RANGE, [PLAY_TOP, PLAY_BOTTOM]))

        game.update(target)
        view = frame.copy()
        cv2.rectangle(view, (ROI_X, 0), (CAM_W - 1, CAM_H - 1), (0, 255, 0), 1)
        if cnt is not None:
            cnt_full = cnt + np.array([[[ROI_X, 0]]])
            cv2.drawContours(view, [cnt_full], -1, (0, 0, 255), 2)
            cv2.circle(view, (ROI_X + int(cv2.moments(cnt)["m10"] / cv2.moments(cnt)["m00"]),
                              int(cy)), 5, (255, 0, 0), -1)
            
        full_mask = np.zeros((CAM_H, CAM_W), np.uint8)
        full_mask[:, ROI_X:] = mask_roi
        side = np.vstack([view, cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)])
        side = cv2.resize(side, (320, GAME_H))

        cv2.imshow("404 Ufo", np.hstack([game.draw(), side]))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("r"), ord("R")):
            game.reset()
        if key in (ord("p"), ord("P")):
            game.toggle_pause()
        if key in (13, 32) and not game.started: 
            game.started = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from quanser.hardware import HIL
from array import array
import time

# ==========================================================
# 0. CONFIGURATION & CALIBRATION
# ==========================================================
sys.path.append(r'C:\Program Files\Quanser\Quanser SDK\python')
MODEL_SEG_PATH = r'C:\Users\Rohit\Desktop\Robotics_Project\fruit_sorting_final\weights\best.pt'

# Strictly mapped values to prevent "Banana/Tomato" identity swaps
BANANA_VALS = {"name": "BANANA", "x": 450, "y": 320, "force": 0.85}
TOMATO_VALS = {"name": "TOMATO", "x": 500, "y": 400, "force": 0.70}

KP_BASE, KP_ELBOW = 0.0006, 0.0009
CHANNELS = array('I', [1000, 1001, 1002, 1003, 1004]) 
ZONE_START = [2.1400, 0.2200, 0.6400, 0.0, 0.0]
ZONE_WAYPOINT_2 = [-0.0053, 0.0852, -0.1035, 0.0, 0.0]
ZONE_PLACE = [-1.4510, 0.5677, 0.6713, 0.0, 0.0]

PATROL_SEQUENCE = [
    ([1.4899, 0.2200, 0.6400, 0.0, 0.0], "ZONE 2"), 
    ([1.5400, 0.2400, 0.8600, 0.0, 0.0], "ZONE 4"), 
    ([2.1640, 0.2400, 0.8600, 0.0, 0.0], "ZONE 3")
]

# ==========================================================
# 1. HELPER FUNCTIONS
# ==========================================================

def travel(card, pipeline, align_obj, target, dur, mode, current_grip=0.0):
    start_t = time.time()
    state = array('d', [0.0]*5)
    card.read_other(CHANNELS, 5, state)
    s_j = list(state)
    while (time.time() - start_t) < dur:
        t = (time.time() - start_t) / dur
        interp = [s_j[i] + (target[i]-s_j[i])*(t**2*(3-2*t)) for i in range(5)]
        interp[4] = current_grip
        card.write_other(CHANNELS, 5, array('d', interp))
        cv2.waitKey(1)

def run_precision_alignment(card, pipeline, align_obj, model, target_idx):
    step, lost_cnt = 0, 0
    f_cx, f_cy, f_q3 = 0, 0, 0.0
    p = BANANA_VALS if target_idx == 0 else TOMATO_VALS

    while step < 33:
        f = pipeline.poll_for_frames()
        if not f: continue
        img = np.asanyarray(align_obj.process(f).get_color_frame().get_data())
        state = array('d', [0.0]*5); card.read_other(CHANNELS, 5, state); q = list(state)

        # PHASE 1: ACTIVE VISION (CENTERING)
        if step < 8:
            res = model.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf=0.50, verbose=False)
            found = False
            if res[0].masks is not None:
                for i, box in enumerate(res[0].boxes):
                    # CRITICAL FIX: Only look for the target_idx we started with
                    if int(box.cls[0]) == target_idx:
                        mask = (res[0].masks.data[i].cpu().numpy()*255).astype(np.uint8)
                        y, x = np.where(mask > 0)
                        pts = np.column_stack((x, y)).astype(np.float32)
                        
                        if target_idx == 0: # BANANA
                            mean_pca, eigenvectors, _ = cv2.PCACompute2(pts, np.mean(pts, axis=0).reshape(1, -1))
                            f_cx, f_cy = int(mean_pca[0][0]), int(mean_pca[0][1])
                            maj_angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])
                            brd_angle = maj_angle + (np.pi / 2.0)
                            vx, vy = np.cos(brd_angle), np.sin(brd_angle)
                            f_q3 = -np.arctan2(abs(vy), abs(vx)) if (vx * vy) < 0 else np.arctan2(abs(vy), abs(vx))
                        else: # TOMATO
                            f_cx, f_cy = int(np.mean(x)), int(np.mean(y))
                            f_q3 = 0.0
                        
                        ex, ey = p["x"] - f_cx, p["y"] - f_cy
                        q[0] += ex * KP_BASE; q[2] -= ey * KP_ELBOW
                        found = True; break
            if not found:
                lost_cnt += 1
                if lost_cnt > 5: return False
        
        # PHASE 3: BLIND PUSH
        elif step >= 11:
            q[2] -= 30.0 * KP_ELBOW 
            q[3] = f_q3

        card.write_other(CHANNELS, 5, array('d', q))
        
        # HUD: Forced to the original target identity
        cv2.drawMarker(img, (p["x"], p["y"]), (255, 0, 255), cv2.MARKER_CROSS, 25, 2)
        cv2.putText(img, f"PICKING: {p['name']}", (20, 60), 1, 2.5, (0, 255, 0), 3)
        cv2.putText(img, f"STEP: {step}/33", (20, 110), 1, 1.8, (255, 255, 255), 2)
        cv2.imshow("Q-Arm Control", img)
        step += 1
        cv2.waitKey(1)
        time.sleep(1.0)
    return True

# ==========================================================
# 2. MAIN MISSION CONTROL
# ==========================================================
model = YOLO(MODEL_SEG_PATH)
pipeline = rs.pipeline(); align_obj = rs.align(rs.stream.color); pipeline.start()
card = HIL(); card.open("qarm_usb", "0")

try:
    while True:
        travel(card, pipeline, align_obj, ZONE_START, 5.0, "HOME", 0.0)
        
        target_found = False
        for pose, zone_name in PATROL_SEQUENCE:
            if target_found: break
            travel(card, pipeline, align_obj, pose, 6.0, f"SCAN {zone_name}", 0.0)
            
            time.sleep(1.0)
            f = pipeline.wait_for_frames()
            img = np.asanyarray(align_obj.process(f).get_color_frame().get_data())
            res = model.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf=0.55, verbose=False)
            
            selected_idx = -1
            if res[0].boxes is not None:
                cls_list = res[0].boxes.cls.cpu().numpy().astype(int)
                # BANANA PRIORITY
                if 0 in cls_list: selected_idx = 0     
                elif 1 in cls_list: selected_idx = 1   
            
            if selected_idx != -1:
                if run_precision_alignment(card, pipeline, align_obj, model, selected_idx):
                    p = BANANA_VALS if selected_idx == 0 else TOMATO_VALS
                    state = array('d', [0.0]*5); card.read_other(CHANNELS, 5, state); q_curr = list(state)
                    for s in range(11):
                        q_curr[4] = (p["force"] / 10) * s
                        card.write_other(CHANNELS, 5, array('d', q_curr))
                        time.sleep(0.1)
                    
                    travel(card, pipeline, align_obj, ZONE_WAYPOINT_2, 12.0, "TRANSIT", p["force"])
                    travel(card, pipeline, align_obj, ZONE_PLACE, 15.0, "DROP", p["force"])
                    
                    release = list(ZONE_PLACE); release[4] = 0.0
                    card.write_other(CHANNELS, 5, array('d', release)); time.sleep(2.0)
                    target_found = True
finally:
    card.close(); pipeline.stop(); cv2.destroyAllWindows()
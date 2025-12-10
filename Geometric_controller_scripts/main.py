from splat_render import SplatRenderer
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion
import torch.nn as nn
import torch.nn.functional as F
from control import QuadrotorController
from quad_dynamics import model_derivative
import tello
from collisionChecker import doesItCollide
import threading, queue, time, torch
import os
import argparse
import window_detector as wd
import sys
import csv

sys.path.append(os.path.abspath(os.path.join(
    "/home/alien/YourDirectoryID_p5/Code",
    "/home/alien/YourDirectoryID_p5/external/core"
)))

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

# ==============================
# Camera intrinsics / extrinsics
# ==============================
global K, C2W, P

# Intrinsic matrix
K = np.array([
    (891.6754191807679, 0.0, 959.5651770640923),
    (0.0, 892.0086815400638, 537.534495239334),
    (0.0, 0.0, 1.0)
])

# Extrinsic matrix (camera to world)
C2W = np.array([
    (0.9900756877459461, 0.010927776933738212, 0.1401096578601137, 0.06838445617022369),
    (0.14053516476096534, -0.07698661243687784, -0.9870779751220785, -0.7929120172024942),
    (4.163336342344337e-17, 0.9969722389298413, -0.07775831018752641, -0.11880440318664898)
])

# Full projection matrix (not strictly needed for ray, but kept)
P = K @ C2W

K_INV = np.linalg.inv(K)

# Camera->body rotation (camera frame to NED-like body frame)
R_B_C = np.array([
    [0.0, 0.0, 1.0],   # x_b = z_c  (forward)
    [1.0, 0.0, 0.0],   # y_b = x_c  (right)
    [0.0, 1.0, 0.0],   # z_b = y_c  (down)
])

SCALE = 0.5

waypoint_log = []

MODEL_PATH = "/home/alien/YourDirectoryID_p3/UNet_background_attention_1.pth"

# Use torch.device directly
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================
# UNet (same as your attention U-Net)
# ====================================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # encoder
        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = double_conv(512, 1024)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # attention blocks
        self.att4 = AttentionBlock(F_g=1024, F_l=512, F_int=256)
        self.att3 = AttentionBlock(F_g=512, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=256, F_l=128, F_int=64)
        self.att1 = AttentionBlock(F_g=128, F_l=64, F_int=32)

        # decoder convs
        self.dconv_up4 = double_conv(1024 + 512, 512)
        self.dconv_up3 = double_conv(512 + 256, 256)
        self.dconv_up2 = double_conv(256 + 128, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.bottleneck(x)

        g = self.upsample(x)
        att4 = self.att4(g=g, x=conv4)
        x = torch.cat([g, att4], dim=1)
        x = self.dconv_up4(x)

        g = self.upsample(x)
        att3 = self.att3(g=g, x=conv3)
        x = torch.cat([g, att3], dim=1)
        x = self.dconv_up3(x)

        g = self.upsample(x)
        att2 = self.att2(g=g, x=conv2)
        x = torch.cat([g, att2], dim=1)
        x = self.dconv_up2(x)

        g = self.upsample(x)
        att1 = self.att1(g=g, x=conv1)
        x = torch.cat([g, att1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out


# =======================
# Small helper functions
# =======================
def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rpy_to_R_c2w(rpy):
    """
    Convert drone roll, pitch, yaw (world frame) into
    a camera->world rotation, assuming camera frame == body frame.
    rpy: [roll, pitch, yaw]
    """
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # ZYX (yaw-pitch-roll) body-to-world
    R = np.array([
        [cp * cy,                 cp * sy,                -sp],
        [sr * sp * cy - cr * sy,  sr * sp * sy + cr * cy, sr * cp],
        [cr * sp * cy + sr * sy,  cr * sp * sy - sr * cy, cr * cp]
    ])
    return R


def log_waypoint(t, pos, vel=None, rpy=None, label="state"):
    """
    Append a waypoint entry to the global log.
    """
    global waypoint_log
    entry = {
        "t": float(t),
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
        "label": label,
    }

    if vel is not None:
        vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
        speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        entry["vx"] = vx
        entry["vy"] = vy
        entry["vz"] = vz
        entry["speed"] = speed

    if rpy is not None:
        entry["roll"] = float(rpy[0])
        entry["pitch"] = float(rpy[1])
        entry["yaw"] = float(rpy[2])

    waypoint_log.append(entry)


def move_upwards_kinematic(currentPose,
                           distance=0.15,
                           steps=60,
                           renderer=None,
                           video_writer=None,
                           sleep_dt=0.02):

    start_pos = currentPose['position'].copy()
    rpy = currentPose['rpy'].copy()
    forward_world = np.array([0.0, 0.0, 1.0])  # Direction +/- Z

    print(f"[FWD-KIN] Moving +Z by {distance} m from {start_pos}")

    for k in range(steps):
        alpha = float(k + 1) / float(steps)
        pos_k = start_pos + alpha * distance * forward_world
        currentPose['position'] = pos_k

        if renderer is not None and video_writer is not None:
            color_frame, _, _ = renderer.render(pos_k, rpy)
            video_writer.write(color_frame)

        time.sleep(sleep_dt)

    print(f"[FWD-KIN] Completed upward move. Final pos = {currentPose['position']}")
    return currentPose


def move_to_point_kinematic(currentPose,
                            target_pos,
                            steps=120,
                            renderer=None,
                            video_writer=None,
                            sleep_dt=0.02):
    """
    Pure kinematic interpolation from currentPose['position']
    to target_pos in WORLD frame. No dynamics, no controller.
    """
    start_pos = currentPose['position'].copy()
    rpy = currentPose['rpy'].copy()   # keep orientation fixed

    target_pos = np.array(target_pos, dtype=float)

    print(f"[KIN-MOVE] From {start_pos} -> {target_pos} in {steps} steps")

    for k in range(steps):
        alpha = float(k + 1) / float(steps)
        pos_k = (1.0 - alpha) * start_pos + alpha * target_pos
        currentPose['position'] = pos_k

        if renderer is not None and video_writer is not None:
            color_frame, _, _ = renderer.render(pos_k, rpy)
            video_writer.write(color_frame)

        time.sleep(sleep_dt)

    print(f"[KIN-MOVE] Completed. Final pos = {currentPose['position']}")
    return currentPose


def move_forward_kinematic(currentPose,
                           distance=0.15,
                           steps=60,
                           renderer=None,
                           video_writer=None,
                           sleep_dt=0.02):

    start_pos = currentPose['position'].copy()
    rpy = currentPose['rpy'].copy()
    forward_world = np.array([1.0, 0.0, 0.0])  # Direction +/- X

    print(f"[FWD-KIN] Moving +X by {distance} m from {start_pos}")

    for k in range(steps):
        alpha = float(k + 1) / float(steps)
        pos_k = start_pos + alpha * distance * forward_world
        pos_k[2] = start_pos[2]  # Keeping altitude fixed
        currentPose['position'] = pos_k

        if renderer is not None and video_writer is not None:
            color_frame, _, _ = renderer.render(pos_k, rpy)
            video_writer.write(color_frame)

        time.sleep(sleep_dt)

    print(f"[FWD-KIN] Completed forward move. Final pos = {currentPose['position']}")
    return currentPose


def yaw_in_place(currentPose,
                 delta_yaw=np.pi,
                 steps=90,
                 renderer=None,
                 video_writer=None,
                 sleep_dt=0.02):
    """
    Slowly yaw the drone in place (no dynamics, just for visualization + pose update).
    - currentPose['position'] is kept fixed.
    - Yaw is interpolated from current yaw to current yaw + delta_yaw.
    - For rendering we keep the camera LEVEL: roll = pitch = 0.
    """

    roll0, pitch0, yaw0 = currentPose['rpy']
    yaw1 = wrap_angle(yaw0 + delta_yaw)

    print(f"[YAW] Starting in-place yaw: yaw0={yaw0:.3f}, yaw1={yaw1:.3f}, steps={steps}")

    for k in range(steps):
        alpha = float(k + 1) / float(steps)
        yaw_k = wrap_angle(yaw0 + alpha * delta_yaw)

        currentPose['rpy'] = np.array([roll0, pitch0, yaw_k])

        if renderer is not None and video_writer is not None:
            cam_rpy = np.array([0.0, 0.0, yaw_k])   # only yaw to renderer
            color_frame, _, _ = renderer.render(currentPose['position'], cam_rpy)
            video_writer.write(color_frame)

        time.sleep(sleep_dt)

    currentPose['rpy'] = np.array([0.0, 0.0, yaw1])

    print(f"[YAW] Completed in-place yaw; final yaw={currentPose['rpy'][2]:.3f}")


# Algorithm
# 1. Get optical data from RAFT only when the drone gets closer to the window
# 2. Once optical flow data is obtained, Use canny edge detector to detect edge and draw a contour around it
# 3. Get the Largest area contour and obtain the centre of the detected contour.
# 4. Perform visual servoing (horizontal and vertical alignment).
# 5. Once aligned, Move forward 0.5 meters
# 6. End the video stream

def make_panel_rgb(color_rgb):
    h, w, _ = color_rgb.shape
    panel = np.hstack([color_rgb])
    return np.ascontiguousarray(panel, dtype=np.uint8)


def frame_to_tensor_rgb(frame_rgb, device=DEVICE, scale=SCALE):
    rgb = frame_rgb
    if scale != 1.0:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float()  # HWC -> CHW
    t = t[None]  # add batch dim
    return t.to(device)


def compute_flow(model, f1, f2):
    padder = InputPadder(f1.shape)
    f1, f2 = padder.pad(f1, f2)
    flow_low, flow_up = model(f1, f2, iters=20, test_mode=True)
    return flow_up


def flow_to_vis(flow):
    flo = flow[0].permute(1, 2, 0).cpu().numpy()
    vis = flow_viz.flow_to_image(flo)
    return vis  # RGB uint8


def detect_hole_from_flow(flow_vis):
    gray = cv2.cvtColor(flow_vis, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = flow_vis.copy()
    hole_center = None

    if len(contours) == 0:
        return overlay, hole_center

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < 500:  # Thresholding the larger area
        return overlay, hole_center

    cv2.drawContours(overlay, [largest], -1, (0, 255, 0), 2)
    M = cv2.moments(largest)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        hole_center = (cX, cY)
        cv2.circle(overlay, hole_center, 5, (255, 0, 0), -1)
        cv2.putText(overlay, "HOLE", (cX - 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return overlay, hole_center


class FrameSaverThread(threading.Thread):
    def __init__(self, frame_dir, model_viz=None, device=None, hz=20.0, frame_count=200):
        """
        frame_dir   : directory to save frames
        model_viz   : RAFT model to run visualization (can be None)
        device      : torch device
        hz          : save rate
        """
        super().__init__(daemon=True)
        self.frame_dir = frame_dir
        os.makedirs(self.frame_dir, exist_ok=True)
        self.model_viz = model_viz
        self.device = device if device is not None else DEVICE
        self.interval = 1.0 / float(hz)
        self.queue = queue.Queue(maxsize=32)
        self.last_save = time.monotonic() - self.interval
        self.count = 0
        self.frame_count = frame_count
        self._stop_evt = threading.Event()

        # RAFT-related state
        self.prev_tensor = None
        self.hole_center = None
        self.last_flow_vis = None
        self.last_hole_overlay_big = None
        self.last_color_with_hole = None

    def enqueue(self, color_rgb):
        """Non-blocking: drop oldest if full so control loop never blocks"""
        try:
            if self.queue.full():
                _ = self.queue.get_nowait()
            self.queue.put_nowait(color_rgb)
        except queue.Full:
            pass

    def stop(self):
        self._stop_evt.set()

    def run(self):
        with torch.no_grad():
            latest_panel = None
            while not self._stop_evt.is_set():
                # Get newest frame if available
                try:
                    item = self.queue.get(timeout=0.05)
                    while True:
                        try:
                            item = self.queue.get_nowait()
                        except queue.Empty:
                            break
                except queue.Empty:
                    item = None

                if item is not None:
                    color_rgb = item  # renderer output assumed RGB
                    latest_panel = make_panel_rgb(color_rgb)

                    # ---------- RAFT + HOLE DETECTION ----------
                    if self.model_viz is not None:
                        cur_tensor = frame_to_tensor_rgb(color_rgb, device=self.device, scale=SCALE)

                        if self.prev_tensor is not None:
                            flow = compute_flow(self.model_viz, self.prev_tensor, cur_tensor)
                            flow_vis = flow_to_vis(flow)
                            self.last_flow_vis = flow_vis.copy()

                            hole_overlay, center_small = detect_hole_from_flow(flow_vis)

                            if center_small is not None:
                                cxs, cys = center_small
                                # up-scale center from RAFT scale back to full image
                                cx = int(cxs / SCALE)
                                cy = int(cys / SCALE)
                                self.hole_center = (cx, cy)

                                hole_overlay_big = cv2.resize(
                                    hole_overlay,
                                    (color_rgb.shape[1], color_rgb.shape[0]),
                                    interpolation=cv2.INTER_NEAREST
                                )
                                self.last_hole_overlay_big = hole_overlay_big.copy()
                                self.last_color_with_hole = color_rgb.copy()

                                # debug snapshot
                                debug_path = os.path.join(self.frame_dir, "hole_debug.png")
                                cv2.imwrite(debug_path, hole_overlay_big)

                        self.prev_tensor = cur_tensor

                # Save at ~Hz
                now = time.monotonic()
                if latest_panel is not None and (now - self.last_save) >= self.interval:
                    if self.count < self.frame_count:
                        fname = os.path.join(self.frame_dir, f"frame_{self.count:06d}.png")
                        # latest_panel is RGB; OpenCV expects BGR -> convert
                        cv2.imwrite(fname, latest_panel)
                        self.count += 1
                        self.last_save = now


# =========================================
# Geometric controller-based goToWaypoint
# =========================================
def goToWaypoint(currentPose, targetPose,
                 velocity=0.1,
                 renderer=None, video_writer=None,
                 log_path=True,
                 log_dir=None,
                 frame_saver_thread=None):
    """
    Navigate quadrotor to a target waypoint using the geometric controller
    state convention (quaternion stored as [w, x, y, z]).

    If log_path=False, no 'state' entries are added to waypoint_log.
    Frames are written to video_writer if provided.
    If frame_saver_thread is provided, frames are also pushed to RAFT.
    """

    dt = 0.01
    tolerance = 0.1
    max_time = 30.0

    controller = QuadrotorController(tello)
    param = tello

    # --- initial state from currentPose ---
    pos = np.array(currentPose['position'], dtype=float)
    rpy = np.array(currentPose['rpy'], dtype=float)

    vel = np.zeros(3)
    pqr = np.zeros(3)

    roll, pitch, yaw = rpy
    quat = (Quaternion(axis=[0, 0, 1], radians=yaw) *
            Quaternion(axis=[0, 1, 0], radians=pitch) *
            Quaternion(axis=[1, 0, 0], radians=roll))

    # geometric controller expects [w, x, y, z]
    current_state = np.concatenate([
        pos, vel,
        [quat.w, quat.x, quat.y, quat.z],
        pqr
    ])

    target_position = np.array(targetPose, dtype=float)

    distance = np.linalg.norm(target_position - pos)
    estimated_time = min(distance / max(velocity, 1e-6) * 2.0, max_time)

    print(f"  Navigating: {pos} → {target_position}")
    print(f"  Distance: {distance:.2f} m, Est. time: {estimated_time:.1f}s")

    if distance < tolerance:
        print("  Already at target (distance < tolerance).")
        return {'position': pos, 'rpy': rpy}

    num_points = max(2, int(estimated_time / dt))
    time_points = np.linspace(0, estimated_time, num_points)

    direction = target_position - pos
    dist_dir = np.linalg.norm(direction)
    unit_direction = direction / dist_dir if dist_dir > 1e-6 else np.zeros(3)

    accel_time = min(1.0, estimated_time * 0.25)
    decel_time = accel_time
    cruise_time = max(0.0, estimated_time - accel_time - decel_time)
    denom = (0.5 * accel_time + cruise_time + 0.5 * decel_time)
    cruise_vel = min(velocity, distance / max(denom, 1e-6))

    trajectory_points, velocities, accelerations = [], [], []

    for tt in range(num_points):
        t = time_points[tt]
        if t <= accel_time:
            vel_mag = (cruise_vel / accel_time) * t
            acc_mag = cruise_vel / accel_time
            prog = 0.5 * (cruise_vel / accel_time) * t * t / max(distance, 1e-6)
        elif t <= accel_time + cruise_time:
            vel_mag = cruise_vel
            acc_mag = 0.0
            prog = (0.5 * cruise_vel * accel_time +
                    cruise_vel * (t - accel_time)) / max(distance, 1e-6)
        else:
            t_d = t - accel_time - cruise_time
            vel_mag = cruise_vel - (cruise_vel / max(decel_time, 1e-6)) * t_d
            vel_mag = max(0.0, vel_mag)
            acc_mag = -cruise_vel / max(decel_time, 1e-6)
            prog = (0.5 * cruise_vel * accel_time +
                    cruise_vel * cruise_time +
                    cruise_vel * t_d -
                    0.5 * (cruise_vel / max(decel_time, 1e-6)) * (t_d * t_d)) / max(distance, 1e-6)

        prog = np.clip(prog, 0.0, 1.0)

        trajectory_points.append(pos + prog * direction)
        velocities.append(vel_mag * unit_direction)
        accelerations.append(acc_mag * unit_direction)

    trajectory_points = np.array(trajectory_points)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)

    controller.set_trajectory(trajectory_points, time_points, velocities, accelerations)

    # Simulate dynamics
    state = current_state.copy()
    state_final = state.copy()

    for i, tt in enumerate(time_points):
        current_pos = state[0:3]
        current_vel = state[3:6]

        if log_path:
            log_waypoint(tt, current_pos, vel=current_vel, rpy=None, label="state")

        err = np.linalg.norm(current_pos - target_position)
        print(f"t={tt:6.3f}s | current={current_pos} | target={target_position} | err={err:.4f} m")

        if renderer is not None and video_writer is not None:
            qw, qx, qy, qz = state[6], state[7], state[8], state[9]
            quat_tmp = Quaternion(w=qw, x=qx, y=qy, z=qz)
            yaw_tmp, pitch_tmp, roll_tmp = quat_tmp.yaw_pitch_roll

            rpy_tmp = np.array([0.0, 0.0, yaw_tmp])
            color_frame, _, _ = renderer.render(current_pos, rpy_tmp)
            video_writer.write(color_frame)

            # Feed frame to RAFT saver thread if provided
            if frame_saver_thread is not None:
                frame_saver_thread.enqueue(color_frame)

        if err < tolerance and tt > 1.0:
            print(f"  ✓ Reached at t={tt:.2f}s, err={err:.3f} m")
            state_final = state.copy()
            break

        if i < len(time_points) - 1:
            # recompute control at each sub-step using current state
            def dyn(tau, X):
                u = controller.compute_control(X, tau)
                return model_derivative(tau, X, u, param)

            sol = solve_ivp(
                dyn,
                [tt, tt + dt],
                state,
                method='RK45',
                max_step=dt
            )
            state = sol.y[:, -1]
            state_final = state.copy()
        else:
            state_final = state.copy()

    final_pos = state_final[0:3]
    qw, qx, qy, qz = state_final[6], state_final[7], state_final[8], state_final[9]
    final_quat = Quaternion(w=qw, x=qx, y=qy, z=qz)
    yaw_f, pitch_f, roll_f = final_quat.yaw_pitch_roll
    final_rpy = np.array([roll_f, pitch_f, yaw_f])

    return {
        'position': final_pos,
        'rpy': final_rpy
    }


forward_offsets = {
    1: 0.3,
    2: 0.3,
    3: 0.25
}
inpass_y_offsets = {
    1: -0.05,
    2: 0.17,
    3: -0.2
}

side_offsets = {
    1: -0.30,
    2: 0.40,
    3: 0.0
}

forward_x_offsets = {
    1: 0.2,
    2: 0.15,
    3: 0.1
}

forward_z_offsets = {
    1: 0.08,
    2: -0.01,
    3: -0.1
}

# Window Flag
all_windows_passed = False


def move_sidewards_kinematic(currentPose,
                             distance=0.15,
                             steps=60,
                             renderer=None,
                             video_writer=None,
                             sleep_dt=0.02):

    start_pos = currentPose['position'].copy()
    rpy = currentPose['rpy'].copy()
    forward_world = np.array([0.0, 1.0, 0.0])  # Direction +/- Y

    print(f"[SIDE-KIN] Moving +Y by {distance} m from {start_pos}")

    for k in range(steps):
        alpha = float(k + 1) / float(steps)
        pos_k = start_pos + alpha * distance * forward_world
        pos_k[2] = start_pos[2]  # Keeping altitude fixed
        currentPose['position'] = pos_k

        if renderer is not None and video_writer is not None:
            color_frame, _, _ = renderer.render(pos_k, rpy)
            video_writer.write(color_frame)

        time.sleep(sleep_dt)

    print(f"[SIDE-KIN] Completed side move. Final pos = {currentPose['position']}")
    return currentPose


def detect_and_forward(win_idx, currentPose, model, renderer, video_writer):
    """
    Detect window (win_idx) using UNet, compute 3D ray through its center,
    and move the drone forward along the horizontal projection of that ray.
    Returns updated currentPose, or -1 on failure.
    """
    global waypoint_log

    timeCounter = 0
    ray_world_last = None
    last_center = None

    # Try a few detections and keep the last valid center
    for it in range(5):
        timeCounter += 1

        color_image, depth_image, metric_depth = renderer.render(
            currentPose['position'],
            currentPose['rpy']
        )
        video_writer.write(color_image)

        mask, center, area, corners = wd.extract_window_featues(model, color_image)

        if mask is not None:
            cv2.imwrite(f'./log/postprocess/w{win_idx}_mask_iter_{it:03d}.png', mask)
            annot = color_image.copy()
            if center is not None:
                cv2.circle(annot, center, 5, (0, 0, 255), -1)
            if corners is not None:
                cv2.polylines(annot, [np.int32(corners)], True, (0, 255, 0), 2)
            cv2.imwrite(f'./log/postprocess/w{win_idx}_annot_iter_{it:03d}.png', annot)

        if center is None:
            print(f"[W{win_idx}] iter {it}: no window detected.")
            continue

        last_center = center
        cx, cy = center
        img_h, img_w, _ = color_image.shape
        img_cx = img_w / 2.0
        img_cy = img_h / 2.0

        err_x_px = cx - img_cx
        err_y_px = cy - img_cy
        print(f"[W{win_idx}] iter {it}: center=({cx:.1f}, {cy:.1f}), "
              f"err_px=({err_x_px:.1f}, {err_y_px:.1f})")

    # Fallback if UNet fails: use image center
    if last_center is None:
        print(f"[W{win_idx}] No window center detected after attempts. Using image center fallback.")
        img_h, img_w, _ = color_image.shape
        last_center = (int(img_w / 2), int(img_h / 2))

    # =============================
    # Build 3D RAY from last_center
    # =============================
    cx, cy = last_center
    print(f"[W{win_idx}] Using final center = ({cx}, {cy}) for ray computation.")

    u = float(cx)
    v = float(cy)
    pixel_h = np.array([u, v, 1.0], dtype=float)

    # 1) Direction in camera coordinates
    ray_cam = K_INV @ pixel_h
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    # 2) Camera → BODY frame (NED body: +x forward, +y right, +z down)
    ray_body = R_B_C @ ray_cam

    # 3) BODY → WORLD using current rpy
    R_c2w_dynamic = rpy_to_R_c2w(currentPose['rpy'])
    cam_origin_world = currentPose['position'].copy()

    ray_world = R_c2w_dynamic @ ray_body
    ray_world = ray_world / np.linalg.norm(ray_world)
    ray_world_last = ray_world.copy()

    print(f"[RAY W{win_idx}] origin_w = {cam_origin_world}, dir_w = {ray_world}")

    # ==========================================
    # Move FORWARD ALONG RAY (HORIZONTAL ONLY)
    # ==========================================
    cur_pos = currentPose['position'].copy()

    if ray_world_last is None:
        print(f"[WARN W{win_idx}] No valid ray_world_last, using +X as forward")
        ray_world_last = np.array([1.0, 0.0, 0.0])

    dist_forward = forward_offsets.get(win_idx, 0.5)

    # Project ray onto horizontal plane
    ray_flat = ray_world_last.copy()
    ray_flat[2] = 0.0

    norm_flat = np.linalg.norm(ray_flat)
    if norm_flat < 1e-6:
        forward_vec = np.array([1.0, 0.0, 0.0])
        print(f"[TARGET W{win_idx}] ray nearly vertical, using +X as forward")
    else:
        forward_vec = ray_flat / norm_flat

    # base forward move
    targetPose = cur_pos + dist_forward * forward_vec

    # per-window lateral bump / offset
    targetPose[0] += forward_x_offsets[win_idx]
    y_bump = inpass_y_offsets.get(win_idx, 0.0)
    targetPose[1] += y_bump
    targetPose[2] -= forward_z_offsets[win_idx]  # small altitude tweak

    print(f"[W{win_idx}-TARGET] cur_pos={cam_origin_world}")
    print(f"[W{win_idx}-TARGET] forward_vec={forward_vec}")
    print(f"[W{win_idx}-TARGET] targetPose={targetPose}")

    newPose = goToWaypoint(
        currentPose,
        targetPose,
        velocity=0.3,
        renderer=renderer,
        video_writer=video_writer,
        log_path=True
    )

    if isinstance(newPose, int):
        print(f"[W{win_idx}] goToWaypoint reported collision or failure.")
        return -1

    print(f"[W{win_idx}] Final pose after forward ray move: {newPose['position']}")
    return newPose


def main(renderer):
    global waypoint_log, all_windows_passed

    os.makedirs('./log', exist_ok=True)
    os.makedirs('./log/postprocess', exist_ok=True)
    os.makedirs('./log/frames', exist_ok=True)

    saver_thread = None  # for RAFT FrameSaverThread

    # Initial pose
    currentPose = {
        'position': np.array([0.0, 0.0, 0.0]),   # world origin
        'rpy': np.radians([0.0, 0.0, 0.0])       # level, yaw=0
    }

    # Initial render (for video size, sanity)
    color_image, depth_image, metric_depth = renderer.render(
        currentPose['position'],
        currentPose['rpy']
    )
    cv2.imwrite("./log/start_frame.png", color_image)
    print(f"[MAIN] Start pose: {currentPose['position']}")

    # Video writer
    h, w, _ = color_image.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = "./log/windows_unet_ray_forward.mp4"
    fps = 30
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    video_writer.write(color_image)

    # -------------------------------
    # Load UNet model for window seg
    # -------------------------------
    print("[MAIN] Loading UNet model...")
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    wd.DEVICE = DEVICE
    print("[MAIN] UNet model loaded.")

    # -------------------------------
    # WINDOW 1: detect + ray + move
    # -------------------------------
    win_idx = 1
    currentPose = detect_and_forward(win_idx, currentPose, model, renderer, video_writer)
    if isinstance(currentPose, int):
        # failure
        video_writer.release()
        return -1

    # -------------------------------
    # SIDE SHIFT: W1 → W2
    # -------------------------------
    print(f"[INFO] Finished window {win_idx}, shifting in y to window 2")

    cur_pos = currentPose['position'].copy()
    target_side = np.array([
        cur_pos[0],
        cur_pos[1] + side_offsets[win_idx],  # shift in y
        cur_pos[2]
    ])

    sidePose = goToWaypoint(
        currentPose,
        target_side,
        velocity=0.2,
        renderer=renderer,
        video_writer=video_writer,
        log_path=True
    )

    if isinstance(sidePose, int):
        print("[WARN] Side-shift to window 2 collided, aborting.")
        video_writer.release()
        return -1

    currentPose = sidePose
    print(f"[INFO] Reached side-shift pose for window 2: {currentPose['position']}")

    # -------------------------------
    # WINDOW 2: detect + ray + move
    # -------------------------------
    win_idx = 2
    currentPose = detect_and_forward(win_idx, currentPose, model, renderer, video_writer)
    if isinstance(currentPose, int):
        video_writer.release()
        return -1

    currentPose = move_sidewards_kinematic(
        currentPose,
        distance=0.28,
        steps=60,
        renderer=renderer,
        video_writer=video_writer,
        sleep_dt=0.02
    )

    # -------------------------------
    # WINDOW 3: detect + ray + move
    # -------------------------------
    win_idx = 3
    currentPose = detect_and_forward(win_idx, currentPose, model, renderer, video_writer)
    if isinstance(currentPose, int):
        video_writer.release()
        return -1

    cur_pos = currentPose['position'].copy()
    target_side = np.array([
        cur_pos[0] + 0.3,
        cur_pos[1] - 0.25,  # shift in y
        cur_pos[2] + 0.05
    ])

    sidePose = goToWaypoint(
        currentPose,
        target_side,
        velocity=0.2,
        renderer=renderer,
        video_writer=video_writer,
        log_path=True
    )

    if isinstance(sidePose, int):
        print("[WARN] Side-shift after window 3 collided, aborting.")
        video_writer.release()
        return -1

    currentPose = sidePose
    print(f"[INFO] Reached side-shift pose after window 3: {currentPose['position']}")
    all_windows_passed = True

    # ======================================
    # RAFT BLOCK AFTER ALL WINDOWS PASSED
    # ======================================
    if all_windows_passed:
        ####################################################
        # 1) INIT RAFT MODEL (once)
        ####################################################
        raft_ckpt = "/home/alien/YourDirectoryID_p4/external/models/raft-sintel.pth"
        print("[RAFT] Loading RAFT model for hole detection...")

        raft_args = argparse.Namespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False
        )
        raft_model = RAFT(raft_args)

        checkpoint = torch.load(raft_ckpt, map_location=DEVICE)
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {}
        for k_ckpt, v in state_dict.items():
            new_key = k_ckpt[7:] if k_ckpt.startswith("module.") else k_ckpt
            new_state_dict[new_key] = v
        raft_model.load_state_dict(new_state_dict)
        raft_model = raft_model.to(DEVICE)
        raft_model.eval()
        print("[RAFT] Model loaded and set to eval().")

        ####################################################
        # 2) RESTART saver_thread TO USE RAFT
        ####################################################
        if saver_thread is not None:
            saver_thread.stop()
            saver_thread.join(timeout=2.0)

        # start RAFT-aware FrameSaverThread
        saver_thread = FrameSaverThread(
            frame_dir="./log/frames_raft",
            model_viz=raft_model,    # RAFT model
            device=DEVICE,
            hz=5.0                   # RAFT frequency
        )
        saver_thread.start()
        print("[RAFT] FrameSaverThread for RAFT started.")

        ####################################################
        # 3) LEFT–RIGHT SWEEP WHILE RAFT RUNS
        ####################################################
        sweep_cycles = 3
        sweep_offset = 0.15   # meters left/right
        sweep_vel = 0.3

        for k_sweep in range(sweep_cycles):
            print(f"[RAFT-SWEEP] cycle {k_sweep+1}/{sweep_cycles}")

            # ---- move LEFT (−y) ----
            cur_pos = currentPose['position'].copy()
            target_left = np.array([
                cur_pos[0],
                cur_pos[1] - sweep_offset,
                cur_pos[2] 
            ])
            newPose = goToWaypoint(
                currentPose,
                target_left,
                velocity=sweep_vel,
                renderer=renderer,
                video_writer=video_writer,
                log_path=True,
                log_dir="./log",
                frame_saver_thread=saver_thread,
            )
            if isinstance(newPose, int):
                print("[RAFT-SWEEP] collision on LEFT move, aborting sweep.")
                break
            currentPose = newPose
            print(f"[RAFT-SWEEP] after LEFT, pos = {currentPose['position']}")

            # ---- move RIGHT (+y) ----
            cur_pos = currentPose['position'].copy()
            target_right = np.array([
                cur_pos[0],
                cur_pos[1] + sweep_offset,
                cur_pos[2] 
            ])
            newPose = goToWaypoint(
                currentPose,
                target_right,
                velocity=sweep_vel,
                renderer=renderer,
                video_writer=video_writer,
                log_path=True,
                log_dir="./log",
                frame_saver_thread=saver_thread,
            )
            if isinstance(newPose, int):
                print("[RAFT-SWEEP] collision on RIGHT move, aborting sweep.")
                break
            currentPose = newPose
            print(f"[RAFT-SWEEP] after RIGHT, pos = {currentPose['position']}")

        ####################################################
        # 4) USE DETECTED HOLE CENTER → 3D RAY
        ####################################################
        hole_center = getattr(saver_thread, "hole_center", None)
        print(f"[RAFT] Final detected hole center (px): {hole_center}")

        if hole_center is not None:
            cx, cy = hole_center

            # get a fresh frame for image geometry
            color_image, _, _ = renderer.render(
                currentPose['position'],
                currentPose['rpy']
            )

            img_h, img_w, _ = color_image.shape
            img_cx = img_w / 2.0
            img_cy = img_h / 2.0
            print(f"[RAFT] image center = ({img_cx:.1f}, {img_cy:.1f})")

            # ---- 4.1 Form pixel homogeneous coords ----
            pixel_h = np.array([float(cx), float(cy), 1.0], dtype=float)

            # ---- 4.2 Ray in camera coords ----
            ray_cam = K_INV @ pixel_h
            ray_cam = ray_cam / np.linalg.norm(ray_cam)

            # ---- 4.3 Camera → BODY → WORLD ----
            ray_body = R_B_C @ ray_cam
            R_c2w_dyn = rpy_to_R_c2w(currentPose['rpy'])
            cam_origin_world = currentPose['position'].copy()

            ray_world = R_c2w_dyn @ ray_body
            ray_world = ray_world / np.linalg.norm(ray_world)

            print(f"[RAFT-RAY] origin_w = {cam_origin_world}, dir_w = {ray_world}")

            ################################################
            # 5) MOVE FORWARD ALONG RAY (HORIZONTAL)
            ################################################
            ray_flat = ray_world.copy()
            ray_flat[2] = 0.0

            norm_flat = np.linalg.norm(ray_flat)
            if norm_flat < 1e-6:
                forward_vec = np.array([1.0, 0.0, 0.0])
                print("[RAFT-RAY] ray nearly vertical, using +X as forward.")
            else:
                forward_vec = ray_flat / norm_flat

            dist_forward = 0.32
            target_hole = cam_origin_world + dist_forward * forward_vec
            target_hole[1] -= 0.1
            target_hole[2] = cam_origin_world[2] - 0.1

            print(f"[RAFT-RAY] target_hole = {target_hole}")

            ################################################
            # 6) SAVE HOLE OVERLAY / FRAME
            ################################################
            if getattr(saver_thread, "last_hole_overlay_big", None) is not None:
                cv2.imwrite(
                    "./log/frames/hole_final_overlay.png",
                    saver_thread.last_hole_overlay_big)
                print("[RAFT] Saved hole_final_overlay.png")

            if getattr(saver_thread, "last_color_with_hole", None) is not None:
                cv2.imwrite(
                    "./log/frames/hole_final_frame.png",
                    saver_thread.last_color_with_hole
                )
                print("[RAFT] Saved hole_final_frame.png")

            ################################################
            # 7) GO THROUGH THE HOLE (first)
            ################################################
            newPose = goToWaypoint(
                currentPose,
                target_hole,
                velocity=0.4,
                renderer=renderer,
                video_writer=video_writer,
                log_path=True,
                log_dir="./log",
                frame_saver_thread=saver_thread,
            )
            if isinstance(newPose, int):
                print("[RAFT-RAY] collision/abort while going through hole.")
            else:
                currentPose = newPose
                print(f"[RAFT-RAY] final pose after passing hole: {currentPose['position']}")

        # Stop RAFT thread before exit
        if saver_thread is not None:
            saver_thread.stop()
            saver_thread.join(timeout=2.0)
        
        # ============================
        #   7.5) SLOW YAW 180° IN PLACE
        # ============================
        # This animates the yaw over many frames so you SEE it in real time.
        yaw_in_place(
            currentPose,
            delta_yaw=np.pi,        # 180 degrees
            steps=90,               # more steps => slower, smoother
            renderer=renderer,
            video_writer=video_writer,
            sleep_dt=0.02           # ~1.8 s total if 90 * 0.02
        )


        # ================================
        # KINEMATIC -X move after yaw
        # ================================
        currentPose = move_forward_kinematic(
            currentPose,
            distance= -0.10,   
            steps=60,       
            renderer=renderer, 
            video_writer=video_writer,
            sleep_dt=0.02
        )
        ######################################
        #Yaw-Leftwards=======#################\
        ######################################
        currentPose = move_sidewards_kinematic(
            currentPose,
            distance= 0.22,   
            steps=60,       
            renderer=renderer, 
            video_writer=video_writer,
            sleep_dt=0.02
        )
        # ==========================================
        # AFTER YAW + LEFTWARD: MOVE TO FIXED POINT
        # ==========================================
        target_after_yaw = np.array([
            0.979672,
            0.358291,
            0.0541516
        ], dtype=float)

        print(f"[POST-YAW] Kinematic move to fixed target {target_after_yaw}")

        currentPose = move_to_point_kinematic(
            currentPose,
            target_pos=target_after_yaw,
            steps=120,          # tune smoothness
            renderer=renderer,
            video_writer=video_writer,
            sleep_dt=0.02
        )

        print(f"[POST-YAW] Reached fixed point (kinematic): {currentPose['position']}")


        #########################
        #Yaw-right###############
        #########################
        currentPose = move_sidewards_kinematic(
            currentPose,
            distance= -0.45,   
            steps=60,       
            renderer=renderer, 
            video_writer=video_writer,
            sleep_dt=0.02
        )
        # ==========================================
        # AFTER YAW + RIGHTWARD: MOVE TO FIXED POINT
        # ==========================================
        target_after_yaw = np.array([
            0.40109104,
            -0.09106485,
            0.0545095
        ], dtype=float)

        print(f"[POST-YAW] Kinematic move to fixed target {target_after_yaw}")

        currentPose = move_to_point_kinematic(
            currentPose,
            target_pos=target_after_yaw,
            steps=120,          # tune smoothness
            renderer=renderer,
            video_writer=video_writer,
            sleep_dt=0.02
        )

        print(f"[POST-YAW] Reached fixed point (kinematic): {currentPose['position']}")
        ######################################
        #Yaw-Leftwards=======#################
        ######################################
        currentPose = move_sidewards_kinematic(
            currentPose,
            distance= 0.38,   
            steps=60,       
            renderer=renderer, 
            video_writer=video_writer,
            sleep_dt=0.02
        )
        # ==========================================
        # AFTER YAW + LEFTWARD: MOVE TO FIXED POINT
        # ==========================================
        target_after_yaw = np.array([
            0.12,
            0.0,
            0.0
        ], dtype=float)

        print(f"[POST-YAW] Kinematic move to fixed target {target_after_yaw}")

        currentPose = move_to_point_kinematic(
        currentPose,
        target_pos=target_after_yaw,
        steps=120,          # tune smoothness
        renderer=renderer,
        video_writer=video_writer,
        sleep_dt=0.02
        )

        print(f"[POST-YAW] Reached fixed point (kinematic): {currentPose['position']}")

    # Final frame
    color_image_final, _, _ = renderer.render(
        currentPose['position'],
        currentPose['rpy']
    )
    cv2.imwrite("./log/final_frame_w2.png", color_image_final)

    # Close video
    video_writer.release()
    print(f"[MAIN] Video saved to {video_path}")

    # ----------------------------------------
    # Save waypoint log to CSV
    # ----------------------------------------
    csv_path = os.path.join("./log", "waypoints.csv")
    if len(waypoint_log) > 0:
        fieldnames = sorted({k for entry in waypoint_log for k in entry.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in waypoint_log:
                writer.writerow(entry)
        print(f"[INFO] Saved {len(waypoint_log)} waypoints to {csv_path}")
    else:
        print("[INFO] No waypoints logged; not writing CSV.")

    print("[MAIN] Finished WINDOW 1–3 detection + 3D-ray forward moves + RAFT hole pass.")
    return 0


if __name__ == "__main__":
    config_path = "../data/P5_colmap_splat/P5_colmap/splatfacto/2025-11-17_130359/config.yml"
    json_path = "../data/render_settings/render_settings.json"

    renderer = SplatRenderer(config_path, json_path)
    main(renderer)

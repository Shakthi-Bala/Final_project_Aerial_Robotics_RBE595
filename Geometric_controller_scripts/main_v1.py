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


# =========================================
# Geometric controller-based goToWaypoint
# =========================================
def goToWaypoint(currentPose, targetPose,
                 velocity=0.1,
                 renderer=None, video_writer=None,
                 log_path=True):
    """
    Navigate quadrotor to a target waypoint using the geometric controller
    state convention (quaternion stored as [w, x, y, z]).

    If log_path=False, no 'state' entries are added to waypoint_log.
    Frames are written to video_writer if provided.
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

    print(f"  Navigating (noisy start): {pos} → {target_position}")
    print(f"  Distance: {distance:.2f} m, Est. time: {estimated_time:.1f}s")

    if distance < tolerance:
        print("  Already at target.")
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

    for tt in time_points:
        if tt <= accel_time:
            vel_mag = (cruise_vel / accel_time) * tt
            acc_mag = cruise_vel / accel_time
            prog = 0.5 * (cruise_vel / accel_time) * tt * tt / max(distance, 1e-6)
        elif tt <= accel_time + cruise_time:
            vel_mag = cruise_vel
            acc_mag = 0.0
            prog = (0.5 * cruise_vel * accel_time +
                    cruise_vel * (tt - accel_time)) / max(distance, 1e-6)
        else:
            t_d = tt - accel_time - cruise_time
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

    for i, tt in enumerate(time_points):

        control_input = controller.compute_control(state, tt)

        current_pos = state[0:3]
        current_vel = state[3:6]   # vx, vy, vz from state vector

        if log_path:
            log_waypoint(tt, current_pos, vel=current_vel, rpy=None, label="state")

        err = np.linalg.norm(current_pos - target_position)
        print(f"t={tt:6.3f}s | current={current_pos} | target={target_position} | err={err:.4f} m")

        # Render & save frame at this state
        if renderer is not None and video_writer is not None:
            qw, qx, qy, qz = state[6], state[7], state[8], state[9]
            quat_tmp = Quaternion(w=qw, x=qx, y=qy, z=qz)
            yaw_tmp, pitch_tmp, roll_tmp = quat_tmp.yaw_pitch_roll

            # camera rendered with roll=pitch=0, yaw from state
            rpy_tmp = np.array([0.0, 0.0, yaw_tmp])

            color_frame, _, _ = renderer.render(current_pos, rpy_tmp)
            video_writer.write(color_frame)

        if err < tolerance and tt > 1.0:
            print(f"  ✓ Reached at t={tt:.2f}s, err={err:.3f} m")
            state_final = state
            break

        if i < len(time_points) - 1:
            sol = solve_ivp(
                lambda tau, X: model_derivative(tau, X, control_input, param),
                [tt, tt + dt],
                state,
                method='RK45',
                max_step=dt
            )
            state = sol.y[:, -1]
            state_final = state
    else:
        state_final = state
        print(f"  Final error: {err:.3f}m")
        print(f"CurrentPose at end: {state_final[0:3]}")
        print(f"TargetPosition: {targetPose}")

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
        2: 0.65,
        3: 0.65
    }
inpass_y_offsets = {
        1: -0.18,   
        2: 0.35,   
        3: 0.00
    }
# ===================================
# Main: WINDOW 1 ONLY (UNet-based)
# ===================================
def main(renderer):
    global waypoint_log

    os.makedirs('./log', exist_ok=True)
    os.makedirs('./log/postprocess', exist_ok=True)

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
    video_path = "./log/window1_unet_ray_forward.mp4"
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
    wd.DEVICE = DEVICE  # if your window_detector uses this
    print("[MAIN] UNet model loaded.")

    # -------------------------------
    # WINDOW 1: detect + ray + move
    # -------------------------------
    win_idx = 1
    num_windows = 3
    timeCounter = 0
    ray_world_last = None

    # For robustness, do a few detection iterations and use the last valid center
    last_center = None

    for it in range(5):
        timeCounter += 1

        color_image, depth_image, metric_depth = renderer.render(
            currentPose['position'],
            currentPose['rpy']
        )
        video_writer.write(color_image)

        # Run your helper to get mask & center from UNet
        mask, center, area, corners = wd.extract_window_featues(model, color_image)

        if mask is not None:
            cv2.imwrite(f'./log/postprocess/w1_mask_iter_{it:03d}.png', mask)
            annot = color_image.copy()
            if center is not None:
                cv2.circle(annot, center, 5, (0, 0, 255), -1)
            if corners is not None:
                cv2.polylines(annot, [np.int32(corners)], True, (0, 255, 0), 2)
            cv2.imwrite(f'./log/postprocess/w1_annot_iter_{it:03d}.png', annot)

        if center is None:
            print(f"[W1] iter {it}: no window detected.")
            continue

        last_center = center
        cx, cy = center
        img_h, img_w, _ = color_image.shape
        img_cx = img_w / 2.0
        img_cy = img_h / 2.0

        err_x_px = cx - img_cx
        err_y_px = cy - img_cy
        print(f"[W1] iter {it}: center=({cx:.1f}, {cy:.1f}), "
              f"err_px=({err_x_px:.1f}, {err_y_px:.1f})")

    if last_center is None:
        print("[W1] No window center detected after attempts. Aborting.")
        video_writer.release()
        return -1

    # =============================
    # Build 3D RAY from last_center
    # =============================
    cx, cy = last_center
    print(f"[W1] Using final center = ({cx}, {cy}) for ray computation.")

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

    print(f"[RAY] origin_w = {cam_origin_world}, dir_w = {ray_world}")

    # ==========================================
    # Move FORWARD ALONG RAY (HORIZONTAL ONLY)
    # ==========================================
    cur_pos = currentPose['position'].copy()

    if ray_world_last is not None:
        # dist_forward = forward_offsets.get(win_idx, 0.5)
        dist_forward = 0.5

                    # Project ray onto horizontal plane (no vertical motion)
        ray_flat = ray_world_last.copy()
        ray_flat[2] = 0.0

        norm_flat = np.linalg.norm(ray_flat)
        if norm_flat < 1e-6:
            forward_vec = np.array([1.0, 0.0, 0.0])
            print("[TARGET] ray nearly vertical, using +X as forward")
        else:
            forward_vec = ray_flat / norm_flat

        # base forward move
        targetPose1 = cur_pos + dist_forward * forward_vec

        # ✅ per-window lateral bump during pass
        targetPose1[0] += 0.07
        y_bump = inpass_y_offsets.get(win_idx, 0.0)
        targetPose1[1] += y_bump

        # small altitude tweak
        targetPose1[2] = cur_pos[2] - 0.08


    print(f"[W1-TARGET] cur_pos={cam_origin_world}")
    print(f"[W1-TARGET] forward_vec={forward_vec}")
    print(f"[W1-TARGET] targetPose1={targetPose1}")

    # ---- navigate through window 1 ----
    newPose = goToWaypoint(
        currentPose,
        targetPose1,
        velocity=0.3,
        renderer=renderer,
        video_writer=video_writer,
        log_path=True
    )

    if isinstance(newPose, int):
        print("[W1] goToWaypoint reported collision or failure.")
        video_writer.release()
        return -1

    currentPose = newPose
    print(f"[W1] Final pose after forward ray move: {currentPose['position']}")

    # Final frame
    color_image_final, _, _ = renderer.render(
        currentPose['position'],
        currentPose['rpy']
    )
    cv2.imwrite("./log/final_frame_w1.png", color_image_final)

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

    print("[MAIN] Finished WINDOW 1 detection + 3D-ray forward move (UNet only).")
    return 0


if __name__ == "__main__":
    config_path = "../data/P5_colmap_splat/P5_colmap/splatfacto/2025-11-17_130359/config.yml"
    json_path = "../data/render_settings/render_settings.json"

    renderer = SplatRenderer(config_path, json_path)
    main(renderer)

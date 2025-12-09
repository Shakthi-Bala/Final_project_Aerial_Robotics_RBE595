import numpy as np
import math
from pyquaternion import Quaternion
from numpy.linalg import norm
import scipy.io


def vee_of_skew(M: np.ndarray) -> np.ndarray:
    """Map a 3x3 skew-symmetric matrix to R^3."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


class gc:
    """
    Geometric position & attitude controller on SE(3).
    Replaces the old PID-based quad_control.
    """

    def __init__(self, mass: float = 0.15, dt: float = 0.01):
        self.dt = dt
        self.mass = mass
        self.g = 9.81

        # Same mapping convention as your original PID controller:
        # linearThrustToU chosen so hover throttle ≈ 0.5
        self.linearThrustToU = mass * self.g * 2.0 / 4.0

        #With velocity 0.3
        # self.kp_pos = np.array([0.01255, 0.01255, 0.001])
        # self.kd_vel = np.array([0.01255, 0.01255, 0.001])
        # self.kp_att = np.array([0.01255, 0.01255, 0.001])

        #With velocity 0.5
        # self.kp_pos = np.array([0.03555, 0.03555, 0.0008])
        # self.kd_vel = np.array([0.03555, 0.03555, 0.0008])
        # self.kp_att = np.array([0.03555, 0.03555, 0.0008])

        self.kp_pos = np.array([0.15555, 0.15555, 0.0008])
        self.kd_vel = np.array([0.15555, 0.15555, 0.0008])
        self.kp_att = np.array([0.15555, 0.15555, 0.0008])
        # Saturations
        self.max_acc = 10.0   # [m/s^2]
        self.max_rate = 5.0   # [rad/s]

        # Logging
        self.current_time = 0.0
        self.timeArray = 0.0
        self.controlArray = np.array([0.0, 0.0, 0.0, 0.0])

    def step(self, X, WP, VEL_SP, ACC_SP):
        """
        Geometric controller step.

        X      : state [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
        WP     : [x_ref, y_ref, z_ref, yaw_ref]
        VEL_SP : desired velocity (3,)
        ACC_SP : desired feedforward acceleration (3,)
        """

        pos = X[0:3]          
        vel = X[3:6]         
        quat_list = X[6:10]   
        omega = X[10:13]      

        quat = Quaternion(quat_list)
        yaw_des = WP[3]

        pos_error = WP[0:3] - pos
        vel_error = VEL_SP - vel

        # PD in translation + feedforward ACC_SP
        acc_cmd = ACC_SP + self.kp_pos * pos_error + self.kd_vel * vel_error

        # Clamp commanded acceleration
        acc_norm = norm(acc_cmd)
        if acc_norm > self.max_acc:
            acc_cmd = acc_cmd / acc_norm * self.max_acc

        f_des = acc_cmd + np.array([0.0, 0.0, self.g])

        f_norm = norm(f_des)
        if f_norm < 1e-6:
            # Degenerate case: just point up
            b3_des = np.array([0.0, 0.0, 1.0])
        else:
            b3_des = f_des / f_norm

        c_y, s_y = math.cos(yaw_des), math.sin(yaw_des)
        b1_yaw = np.array([c_y, s_y, 0.0])

        # b2_des = b3_des x b1_yaw
        b2_des = np.cross(b3_des, b1_yaw)
        if norm(b2_des) < 1e-6:
            # handle singular if b3_des aligned with yaw axis
            b1_yaw = np.array([math.cos(yaw_des + 0.1),
                               math.sin(yaw_des + 0.1),
                               0.0])
            b2_des = np.cross(b3_des, b1_yaw)

        b2_des = b2_des / norm(b2_des)
        b1_des = np.cross(b2_des, b3_des)
        b1_des = b1_des / norm(b1_des)

        R_des = np.column_stack((b1_des, b2_des, b3_des))

        R = quat.rotation_matrix.T

        R_err = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_rot = vee_of_skew(R_err)

        omega_des = self.kp_att * e_rot
        omega_des = np.clip(omega_des, -self.max_rate, self.max_rate)

        # Use these as "torques" going into the mixer (like your PID tau_x,y,z)
        tau_x, tau_y, tau_z = omega_des

        thrust = f_norm * self.mass
        # Convert to a throttle [0..1] given your linearThrustToU convention
        throttle = thrust / (4.0 * self.linearThrustToU)

        u1 = throttle - tau_x + tau_y + tau_z
        u2 = throttle + tau_x - tau_y + tau_z
        u3 = throttle + tau_x + tau_y - tau_z
        u4 = throttle - tau_x - tau_y - tau_z

        U = np.array([u1, u2, u3, u4])
        U = np.clip(U, 0.0, 1.0)

    
        self.controlArray = np.vstack(
            (self.controlArray, np.array([throttle, tau_x, tau_y, tau_z]))
        )
        self.timeArray = np.append(self.timeArray, self.current_time)
        self.current_time += self.dt

        try:
            scipy.io.savemat(
                "./log/control.mat",
                {"control_time": self.timeArray, "control_premix": self.controlArray},
            )
        except Exception:
            pass

        return U


class QuadrotorController:
    """
    Fully geometric controller interface (drop-in replacement).
    Keeps the same external API as your previous QuadrotorController.
    """

    def __init__(self, drone_params):
        """
        drone_params must at least have:
          - mass
          - linearThrustToU (used only in the emergency hover fallback)
        """
        self.params = drone_params
        self.controller = gc(mass=drone_params.mass, dt=0.01)

        # Trajectory tracking data
        self.trajectory_points = None
        self.trajectory_velocities = None
        self.trajectory_accelerations = None
        self.time_points = None

        # Performance metrics
        self.position_errors = []
        self.velocity_errors = []

    # ------------------------------------------------------------------
    # Trajectory interface
    # ------------------------------------------------------------------
    def set_trajectory(self, trajectory_points, time_points, velocities, accelerations):
        """Set the reference trajectory (same as before)."""
        self.trajectory_points = trajectory_points
        self.time_points = time_points
        self.trajectory_velocities = velocities
        self.trajectory_accelerations = accelerations

    def get_desired_state(self, t):
        """Get desired (pos, vel, acc) at time t with simple interpolation."""
        if (
            self.trajectory_points is None
            or len(self.trajectory_points) == 0
            or self.time_points is None
        ):
            return np.zeros(3), np.zeros(3), np.zeros(3)

        if t <= self.time_points[0]:
            idx = 0
        elif t >= self.time_points[-1]:
            idx = len(self.trajectory_points) - 1
        else:
            idx = np.searchsorted(self.time_points, t)
            if idx > 0:
                t1, t2 = self.time_points[idx - 1], self.time_points[idx]
                alpha = (t - t1) / (t2 - t1) if t2 != t1 else 0.0

                pos = (1.0 - alpha) * self.trajectory_points[idx - 1] + alpha * self.trajectory_points[idx]
                vel = (1.0 - alpha) * self.trajectory_velocities[idx - 1] + alpha * self.trajectory_velocities[idx]
                acc = (1.0 - alpha) * self.trajectory_accelerations[idx - 1] + alpha * self.trajectory_accelerations[idx]
                return pos, vel, acc
            else:
                idx = 0

        return (
            self.trajectory_points[idx],
            self.trajectory_velocities[idx],
            self.trajectory_accelerations[idx],
        )

    def compute_control(self, current_state, t):
        """
        Main control call.
        current_state: same state vector as before.
        t: current simulation time.
        """
        # Get desired trajectory state
        pos_des, vel_des, acc_des = self.get_desired_state(t)

        # Build waypoint [x, y, z, yaw]
        waypoint = np.append(pos_des, 0.0)  # zero yaw for now

        try:
            U = self.controller.step(current_state, waypoint, vel_des, acc_des)
        except Exception as e:
            print(f"Controller error: {e}")
            # Emergency hover using original mapping
            hover_thrust = self.params.mass * 9.81
            hover_throttle = hover_thrust / (4.0 * self.params.linearThrustToU)
            U = np.array([hover_throttle] * 4)

        # Track performance using the true state
        pos_error = np.linalg.norm(current_state[0:3] - pos_des)
        vel_error = np.linalg.norm(current_state[3:6] - vel_des)

        self.position_errors.append(pos_error)
        self.velocity_errors.append(vel_error)

        return U

    def reset_metrics(self):
        """Reset stored performance metrics."""
        self.position_errors = []
        self.velocity_errors = []

    def get_performance_summary(self):
        """Return a formatted string with basic performance statistics."""
        if not self.position_errors:
            return "No performance data"

        return f"""
Performance Summary:
  Mean Position Error: {np.mean(self.position_errors):.3f} m
  Max Position Error:  {np.max(self.position_errors):.3f} m
  Mean Velocity Error: {np.mean(self.velocity_errors):.3f} m/s
  Max Velocity Error:  {np.max(self.velocity_errors):.3f} m/s
"""

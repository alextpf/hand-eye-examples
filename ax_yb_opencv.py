"""
t: target board
b: base
g: gripper
c: camera

1.
calibrateHandEye() solves for:
T_bg1 * T_gc * T_ct1  = T_bt = T_bg2 * T_gc * T_ct2

i.e.
( T_bg2.inv() * T_bg1 ) * T_gc = T_gc * ( T_ct2 * T_ct1.inv() )
           A            *  X   =   X  *          B

knowns are A & B, or
- A: T_bg2.inv() * T_bg1
- B: T_ct2 * T_ct1.inv()

unknowns are:
- X: T_gc

R_gc, t_gc  = calibrateHandEye(R_A, t_A, R_B, t_B)

2.
calibrateRobotWorldHandEye() solves for:
T_cw * T_wb = T_cb = T_cg * T_gb
i.e. T_cw * T_wb = T_cg * T_gb
or: AX = YB

knowns are A & B, or
- A: T_cw
- B: T_gb

unknowns are:
- X: T_wb
- Y: T_cg

R_wb, t_wb, R_cg, t_cg = calibrateRobotWorldHandEye(R_cw, t_cw, R_gb, t_gb)
i.e.
R_X, t_X, R_Y, t_Y = calibrateRobotWorldHandEye(R_A, t_A, R_B, t_B)
"""
import numpy as np
import cv2
from gtsam import Pose3, Rot3
import pandas as pd
from typing import Tuple, List, Dict

# --- Global Ground Truth (GT) ---
# X: T_wb_gt (World to Base - Robot-World transformation)
T_wb_gt = Pose3(Rot3.RzRyRx(0.1, -0.1, 0.2), np.array([0.1, 0.2, 0.3]))
# Y: T_cg_gt (Camera to Gripper - Hand-Eye transformation)
T_cg_gt = Pose3(Rot3.RzRyRx(0.3, 0.2, -0.1), np.array([-0.3, 0.4, -0.5]))
# Noise standard deviation for camera measurements (A)
NOISE_STD = 0.005 # Using a smaller noise for the measurement A

# --- Helper Functions ---
def calculate_pose_error(T_gt: Pose3, T_est: Pose3, name: str) -> Tuple[float, float, float]:
    """Calculates Rotation, Translation, and Combined Logmap error norms."""
    error_pose = T_gt.between(T_est)
    log_error = Pose3.Logmap(error_pose)
    R_error_norm = np.linalg.norm(log_error[0:3]) # Error in Radians
    T_error_norm = np.linalg.norm(log_error[3:6]) # Error in Meters
    Combined_error_norm = np.linalg.norm(log_error)
    return R_error_norm, T_error_norm, Combined_error_norm

# --- 1. Data Generation ---
def generate_calibration_data(N: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], Pose3, Pose3]:
    """
    Generates N pairs of measurement data (T_cw, T_gb) for the T_cw * T_wb = T_cg * T_gb problem.
    T_cw (A) is the measured camera pose (with noise).
    T_gb (B) is the measured robot pose (assumed perfect).
    """
    R_cw_meas = [] # Rotation of T_cw (Matrix A)
    T_cw_meas = [] # Translation of T_cw (Matrix A)
    R_gb_robot = [] # Rotation of T_gb (Matrix B)
    T_gb_robot = [] # Translation of T_gb (Matrix B)

    np.random.seed(42)

    for _ in range(N):
        # 1. Generate Robot Pose T_gb (Matrix B) with high excitation (perfect measurement)
        rx = (np.random.rand() - 0.5) * np.pi
        ry = (np.random.rand() - 0.5) * np.pi
        rz = (np.random.rand() - 0.5) * np.pi
        tx = (np.random.rand() - 0.5) * 1.5
        ty = (np.random.rand() - 0.5) * 1.5
        tz = (np.random.rand() - 0.5) * 1.5
        T_gb_i = Pose3(Rot3.RzRyRx(rx, ry, rz), np.array([tx, ty, tz]))

        # 2. Calculate the corresponding Camera Measurement T_cw (Matrix A)
        # Loop: T_cw * T_wb_gt = T_cg_gt * T_gb
        # Solving for T_cw: T_cw = (T_cg_gt * T_gb) * T_wb_gt^-1
        T_cw_ideal = T_cg_gt.compose(T_gb_i).compose(T_wb_gt.inverse())

        # 3. Add noise to the camera measurement (T_cw)
        T_cw_noisy = T_cw_ideal.compose(
             Pose3.Expmap(np.random.normal(0, NOISE_STD, 6))
        )

        # 4. Convert GTSAM Pose3 to OpenCV R/T format (Numpy Arrays)
        R_cw_meas.append(T_cw_noisy.rotation().matrix())
        T_cw_meas.append(T_cw_noisy.translation().reshape(3, 1))

        R_gb_robot.append(T_gb_i.rotation().matrix())
        T_gb_robot.append(T_gb_i.translation().reshape(3, 1))

    return (R_cw_meas, T_cw_meas, R_gb_robot, T_gb_robot, T_wb_gt, T_cg_gt)


# --- 2. Main Calibration Function ---
def solve_axby_opencv():
    N = 500
    print(f"Generating {N} data points for AX=YB (T_cw * T_wb = T_cg * T_gb) with noise_std={NOISE_STD}...")

    # R_A, T_A is T_cw (Camera Measurement)
    # R_B, T_B is T_gb (Robot Pose)
    R_A, T_A, R_B, T_B, T_wb_gt, T_cg_gt = generate_calibration_data(N)

    # Dictionary of available OpenCV methods for AX=YB (RobotWorldHandEye)
    methods: Dict[str, int] = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    results = []

    # ----------------------------------------------------------------------
    # Iteration through methods
    # ----------------------------------------------------------------------
    for name, method_id in methods.items():
        try:
            # OpenCV call for AX=YB (Robot-World Hand-Eye)
            # The function signature for RWH:
            # R_wb, t_wb, R_cg, t_cg = cv2.calibrateRobotWorldHandEye(R_cw, t_cw, R_gb, t_gb, method)
            R_wb_raw, T_wb_raw, R_cg_raw, T_cg_raw = cv2.calibrateRobotWorldHandEye(
                R_A, # Input A (T_cw)
                T_A,
                R_B, # Input B (T_gb)
                T_B,
                method=method_id
            )

            # 1. T_wb (World to Base, X) is returned first
            T_wb_est_pose = Pose3(Rot3(R_wb_raw), T_wb_raw.flatten())
            # 2. T_cg (Camera to Gripper, Y) is returned second
            T_cg_est_pose = Pose3(Rot3(R_cg_raw), T_cg_raw.flatten())

            # Calculate errors
            R_wb_err, T_wb_err, C_wb_err = calculate_pose_error(T_wb_gt, T_wb_est_pose, "T_wb")
            R_cg_err, T_cg_err, C_cg_err = calculate_pose_error(T_cg_gt, T_cg_est_pose, "T_cg")

            total_error = C_wb_err + C_cg_err

            results.append({
                "Method": name,
                "R_wb_err": R_wb_err,
                "T_wb_err": T_wb_err,
                "R_cg_err": R_cg_err,
                "T_cg_err": T_cg_err,
                "Total_Error": total_error,
                "T_wb_est": T_wb_est_pose.translation(),
                "T_cg_est": T_cg_est_pose.translation(),
            })

        except cv2.error as e:
            print(f"Warning: Method {name} failed. Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with method {name}: {e}")

    # ----------------------------------------------------------------------
    # 3. Output and Selection
    # ----------------------------------------------------------------------

    if not results:
        print("\nNo calibration results were successfully calculated.")
        return

    # Filter out failed methods (where error is inf)
    df = pd.DataFrame(results)
    df = df[df['Total_Error'] != np.inf].sort_values(by="Total_Error").reset_index(drop=True)

    if df.empty:
        print("\nAll calibration methods failed.")
        return

    # Print detailed comparison table
    print("\n" + "="*80)
    print("🤖 OpenCV Robot-World Hand-Eye Calibration (AX=YB) Comparison")
    print("="*80)
    print(df[["Method", "R_wb_err", "T_wb_err", "R_cg_err", "T_cg_err", "Total_Error"]].to_string(index=False, float_format="%.6f"))
    print("="*80)

    best_method = df.iloc[0]

    # Print Ground Truth
    print(f"\n--- Ground Truth (GT) ---")
    print(f"T_wb_gt (World to Base) Translation: {T_wb_gt.translation()}")
    print(f"T_cg_gt (Camera to Gripper) Translation: {T_cg_gt.translation()}")

    # Print Best Estimate
    print(f"\n--- Best Estimated Pose (Method: {best_method['Method']}) ---")
    print(f"T_wb_est Translation: {best_method['T_wb_est']}")
    print(f"T_cg_est Translation: {best_method['T_cg_est']}")

    # Final summary
    print("\n--- Summary of Best Method ---")
    print(f"Best Method: {best_method['Method']}")
    print(f"Total Combined Error: {best_method['Total_Error']:.6f}")
    print(f"T_wb (World-Base) R/T Error: {best_method['R_wb_err']:.6f} rad / {best_method['T_wb_err']:.6f} m")
    print(f"T_cg (Hand-Eye) R/T Error: {best_method['R_cg_err']:.6f} rad / {best_method['T_cg_err']:.6f} m")

if __name__ == "__main__":
    solve_axby_opencv()
"""
1.
calibrateHandEye() solves for:
T_cg * T_gb = T_cb = T_ct * T_tb
i.e.
T_cg * T_gb = T_ct * T_tb
  X  *  A   =   B  *  Y

knowns are A & B, or
- A: analogous to T_gb
- B: analogous to T_ct

unknowns are:
- X: analogous to T_cg
- Y: analogous to T_tb

R_cg, t_cg, R_wb, t_wb = calibrateHandEye(R_gb, t_gb, R_ct, t_ct)
i.e.
R_X, t_X, R_Y, t_Y = calibrateHandEye(R_A, t_A, R_B, t_B)

2.
calibrateRobotWorldHandEye() solves for:
T_cw * T_wb = T_cb = T_cg * T_gb
i.e. T_cw * T_wb = T_cg * T_gb
or: AX = YB

knowns are A & B, or
- A: analogous to T_cw
- B: analogous to T_gb

unknowns are:
- X: analogous to T_wb
- Y: analogous to T_cg

R_wb, t_wb, R_cg, t_cg = calibrateRobotWorldHandEye(R_cw, t_cw, R_gb, t_gb)
i.e.
R_X, t_X, R_Y, t_Y = calibrateRobotWorldHandEye(R_A, t_A, R_B, t_B)
"""
import numpy as np
import cv2
from gtsam import Pose3, Rot3
import pandas as pd
from typing import Tuple, List, Dict

# --- 1. Data Generation ---
def generate_calibration_data(N: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], Pose3, Pose3]:
    """
    Generates N pairs of measurement data (T_da_meas, T_cb_i) for the T_da = T_dc * T_cb * T_ba problem.
    """
    # Ground Truth (GT)
    T_dc_gt = Pose3(Rot3.RzRyRx(0.1, -0.1, 0.2), np.array([0.1, 0.2, 0.3]))
    T_ba_gt = Pose3(Rot3.RzRyRx(0.3, 0.2, -0.1), np.array([-0.3, 0.4, -0.5]))

    R_gripper2base = [] # Rotation of T_da_meas (Matrix B in AX=YB mapping)
    T_gripper2base = [] # Translation of T_da_meas
    R_target2cam = []   # Rotation of T_cb_i (Matrix A in AX=YB mapping)
    T_target2cam = []   # Translation of T_cb_i

    np.random.seed(42)
    noise_std = 0.05

    for _ in range(N):
        # 1. Generate T_c_from_b (Matrix B in formulation) with high excitation
        rx = (np.random.rand() - 0.5) * np.pi
        ry = (np.random.rand() - 0.5) * np.pi
        rz = (np.random.rand() - 0.5) * np.pi
        tx = (np.random.rand() - 0.5) * 4.0
        ty = (np.random.rand() - 0.5) * 4.0
        tz = (np.random.rand() - 0.5) * 4.0
        T_cb_i = Pose3(Rot3.RzRyRx(rx, ry, rz), np.array([tx, ty, tz]))

        # 2. Calculate T_d_from_a (Matrix A in formulation) with noise
        T_da_meas = T_dc_gt.compose(T_cb_i).compose(T_ba_gt).compose(
            Pose3.Expmap(np.random.normal(0, noise_std, 6))
        )

        # 3. Convert GTSAM Pose3 to OpenCV R/T format (Numpy Arrays)
        R_gripper2base.append(T_da_meas.rotation().matrix())
        T_gripper2base.append(T_da_meas.translation().reshape(3, 1))

        R_target2cam.append(T_cb_i.rotation().matrix())
        T_target2cam.append(T_cb_i.translation().reshape(3, 1))

    return (R_gripper2base, T_gripper2base, R_target2cam, T_target2cam, T_dc_gt, T_ba_gt)

def calculate_pose_error(T_gt: Pose3, T_est: Pose3) -> Tuple[float, float, float]:
    """Calculates Rotation, Translation, and Combined Logmap error norms."""
    error_pose = T_gt.between(T_est)
    log_error = Pose3.Logmap(error_pose)
    R_error_norm = np.linalg.norm(log_error[0:3])
    T_error_norm = np.linalg.norm(log_error[3:6])
    Combined_error_norm = np.linalg.norm(log_error)
    return R_error_norm, T_error_norm, Combined_error_norm

# --- 2. Main Calibration Function ---
def solve_axby_opencv():
    N = 500
    print(f"Generating {N} data points for calibration with noise_std=0.05...")

    # R_DA, T_DA are T_da_meas (Robot base to Gripper/Camera frame)
    # R_CB, T_CB are T_cb_i (Target frame to World/Calibration target frame)
    R_DA, T_DA, R_CB, T_CB, T_dc_gt, T_ba_gt = generate_calibration_data(N)

    # Dictionary of available OpenCV methods for AX=YB (RobotWorldHandEye)
    methods: Dict[str, int] = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    }

    results = []

    # ----------------------------------------------------------------------
    # Iteration through methods
    # ----------------------------------------------------------------------
    for name, method_id in methods.items():
        try:
            # OpenCV call for AX=YB (Robot-World Hand-Eye)
            # The output mapping is determined experimentally:
            # R_ba, T_ba, R_cd, T_cd = cv2.calibrateRobotWorldHandEye(R_CB, T_CB, R_DA, T_DA, method)
            R_ba_raw, T_ba_raw, R_cd_raw, T_cd_raw = cv2.calibrateRobotWorldHandEye(
                R_CB, # Input A (T_cb_i)
                T_CB, # Input A
                R_DA, # Input B (T_da_meas)
                T_DA, # Input B
                method=method_id
            )

            # 1. T_ba (Hand-Eye) is returned first and is correct
            T_ba_est_pose = Pose3(Rot3(R_ba_raw), T_ba_raw.flatten())

            # 2. T_dc (Robot-World) is returned as T_cd (T_dc inverse)
            T_cd_est_pose = Pose3(Rot3(R_cd_raw), T_cd_raw.flatten())
            T_dc_est_pose = T_cd_est_pose.inverse() # Correcting the inverse

            # Calculate errors
            R_dc_err, T_dc_err, C_dc_err = calculate_pose_error(T_dc_gt, T_dc_est_pose)
            R_ba_err, T_ba_err, C_ba_err = calculate_pose_error(T_ba_gt, T_ba_est_pose)

            total_error = C_dc_err + C_ba_err

            results.append({
                "Method": name,
                "R_dc_err": R_dc_err,
                "T_dc_err": T_dc_err,
                "R_ba_err": R_ba_err,
                "T_ba_err": T_ba_err,
                "Total_Error": total_error,
                "T_dc_est": T_dc_est_pose.translation(),
                "T_ba_est": T_ba_est_pose.translation(),
            })

        except cv2.error as e:
            print(f"Warning: Method {name} failed. Error: {e}")

    # ----------------------------------------------------------------------
    # 3. Output and Selection
    # ----------------------------------------------------------------------

    if not results:
        print("\nNo calibration results were successfully calculated.")
        return

    df = pd.DataFrame(results).sort_values(by="Total_Error").reset_index(drop=True)

    # Print detailed comparison table
    print("\n" + "="*80)
    print("🤖 OpenCV AX=YB Calibration Method Comparison (N=500, Noise=0.05)")
    print("="*80)
    print(df[["Method", "R_dc_err", "T_dc_err", "R_ba_err", "T_ba_err", "Total_Error"]].to_string(index=False, float_format="%.6f"))
    print("="*80)

    best_method = df.iloc[0]

    # Print Ground Truth
    print(f"\n--- Ground Truth (GT) ---")
    print(f"T_dc_gt Translation: {T_dc_gt.translation()}")
    print(f"T_ba_gt Translation: {T_ba_gt.translation()}")

    # Print Best Estimate
    print(f"\n--- Best Estimated Pose (Method: {best_method['Method']}) ---")
    print(f"T_dc_est Translation: {best_method['T_dc_est']}")
    print(f"T_ba_est Translation: {best_method['T_ba_est']}")

    # Final summary
    print("\n--- Summary of Best Method ---")
    print(f"Best Method: {best_method['Method']}")
    print(f"Total Combined Error: {best_method['Total_Error']:.6f}")
    print(f"T_dc (Robot-World) R/T Error: {best_method['R_dc_err']:.6f} rad / {best_method['T_dc_err']:.6f} m")
    print(f"T_ba (Hand-Eye) R/T Error: {best_method['R_ba_err']:.6f} rad / {best_method['T_ba_err']:.6f} m")

if __name__ == "__main__":
    solve_axby_opencv()
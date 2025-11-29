"""
Solve a Hand-Eye Calibration AX = YB problem.
T_d_from_a = T_d_from_c * T_c_from_b * T_b_from_a
where the notation of "T_x_from_y" is a 6 DOF transformation (3 DOF rotaion and 3 DOF translation) denoting
transforming from the coordinate system y to x. That is, if a point P is in coordinate system y,
then T_x_from_y*P brings P from y to coordinate system x. We have N measurements, each of which
gives me the T_d_from_a and T_c_from_b for each of the measurements. The unknowns:
1. T_d_from_c
2. T_b_from_a

"""

import gtsam
import numpy as np
from gtsam import Pose3, Rot3, symbol, CustomFactor


class AXBYFactor(gtsam.CustomFactor):
    """
    Revised AXBYFactor using a closure to robustly bind measurement data
    to the error function, bypassing C++ binding limitations.
    """

    # 1. Constructor: Defines the error function as a closure
    def __init__(self, key_T_dc, key_T_ba, T_cb_meas, T_da_meas, noiseModel):

        # Define the error function (closure) *inside* the constructor.
        # This function captures T_cb_meas and T_da_meas.
        def error_closure(
            this_factor: gtsam.CustomFactor, v: gtsam.Values, H: list | None
        ) -> np.ndarray:

            # Since the data is captured, we don't need to access 'this_factor' attributes!

            # Get variable keys (still accessed from the factor object)
            key_T_dc = this_factor.keys()[0]
            key_T_ba = this_factor.keys()[1]

            T_dc = v.atPose3(key_T_dc)
            T_ba = v.atPose3(key_T_ba)

            # Use the captured measurement data
            T_cb = T_cb_meas  # <-- Captured from __init__ scope
            T_da_meas_cap = T_da_meas  # <-- Captured from __init__ scope

            # Compute the predicted T_da
            T_da_predicted = T_dc.compose(T_cb).compose(T_ba)

            # Compute the residual
            residual = T_da_predicted.localCoordinates(T_da_meas_cap)

            if H is not None:
                H[0] = np.zeros((6, 6))  # Numerical placeholder
                H[1] = np.zeros((6, 6))  # Numerical placeholder

            return residual

        # Call the base class constructor, passing the closure function
        gtsam.CustomFactor.__init__(
            self,
            noiseModel,
            gtsam.KeyVector([key_T_dc, key_T_ba]),
            error_closure,  # Pass the function object that contains the captured data
        )


def main():
    # Solve T_da = T_dc * T_cb * T_ba
    # knowns: T_da & T_cb
    # Unknowns: T_dc & T_ba
    # --- Example Data ---
    N = 500  # Increased from 50 to 500

    # --- Ground Truth ---
    T_dc_gt = Pose3(Rot3.RzRyRx(0.1, -0.1, 0.2), np.array([0.1, 0.2, 0.3]))
    T_ba_gt = Pose3(Rot3.RzRyRx(0.3, 0.2, -0.1), np.array([-0.3, 0.4, -0.5]))

    T_dc_KEY = symbol("X", 0)
    T_ba_KEY = symbol("X", 1)

    graph = gtsam.NonlinearFactorGraph()
    # Noise remains reasonable for realistic data
    factor_noise = gtsam.noiseModel.Isotropic.Sigmas(
        np.array([0.05] * 6)
    )  # 5cm/5deg std

    # --- Add Factors ---
    np.random.seed(42)  # For reproducible results
    for i in range(N):
        # Generate measurements T_c_from_b with high excitation (large, varied rotation/translation)

        # Rotations: Random values between -pi/2 and +pi/2 (approx +/- 1.57 rad)
        rx = (np.random.rand() - 0.5) * np.pi
        ry = (np.random.rand() - 0.5) * np.pi
        rz = (np.random.rand() - 0.5) * np.pi

        # Translations: Random values between -2.0 and +2.0 meters
        tx = (np.random.rand() - 0.5) * 4.0
        ty = (np.random.rand() - 0.5) * 4.0
        tz = (np.random.rand() - 0.5) * 4.0

        T_cb_i = Pose3(Rot3.RzRyRx(rx, ry, rz), np.array([tx, ty, tz]))

        # Calculate T_d_from_a and add measurement noise
        T_da_meas = (
            T_dc_gt.compose(T_cb_i)
            .compose(T_ba_gt)
            .compose(Pose3.Expmap(np.random.normal(0, 0.05, 6)))  # Add noise
        )

        # Use the custom factor
        new_factor = AXBYFactor(T_dc_KEY, T_ba_KEY, T_cb_i, T_da_meas, factor_noise)
        graph.add(new_factor)

    # --- Initial Estimates ---
    initial_estimate = gtsam.Values()
    # Use a noisy guess for both
    T_dc_guess = T_dc_gt.compose(Pose3.Expmap(np.random.normal(0, 0.01, 6)))
    T_ba_guess = T_ba_gt.compose(Pose3.Expmap(np.random.normal(0, 0.01, 6)))

    initial_estimate.insert(T_dc_KEY, T_dc_guess)
    initial_estimate.insert(T_ba_KEY, T_ba_guess)

    # --- Optimize ---
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    # --- Output ---
    T_dc_est = result.atPose3(T_dc_KEY)
    T_ba_est = result.atPose3(T_ba_KEY)

    print("--- Optimization Summary ---")
    print(f"Number of Measurements (N): {N}")
    print(f"Residual Error Norm (Pre-Optimization): {graph.error(initial_estimate):.4f}")
    print(f"Residual Error Norm (Post-Optimization): {graph.error(result):.4f}")

    print("\n" + "="*50 + "\n")
    print("T_d from c (Unknown 1)")
    print(f"Ground Truth:\n{T_dc_gt}")
    print(f"Estimated:\n{T_dc_est}")

    error_dc_pose = T_dc_gt.between(T_dc_est) # This is a Pose3 object (the error transformation)
    error_dc = Pose3.Logmap(error_dc_pose)   # Call Logmap static function with the pose

    print(f"Estimation Error (Logmap):\n{error_dc}")
    print(f"Total error norm: {np.linalg.norm(error_dc):.6f}")

    print("\n" + "="*50 + "\n")
    print("T_b from a (Unknown 2)")
    print(f"Ground Truth:\n{T_ba_gt}")
    print(f"Estimated:\n{T_ba_est}")
    error_ba_pose = T_ba_gt.between(T_ba_est)
    error_ba = Pose3.Logmap(error_ba_pose)   # Call Logmap static function with the pose

    print(f"Estimation Error (Logmap):\n{error_ba}")
    print(f"Total error norm: {np.linalg.norm(error_ba):.6f}")
    print("\n" + "="*50)


if __name__ == "__main__":
    main()

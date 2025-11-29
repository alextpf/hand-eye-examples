import numpy as np
import math
import gtsam
import pyceres as ceres
from gtsam import Pose3, Rot3

# --- 1. SE(3) Manifold Definition for Ceres ---

# The deprecated ceres.LocalParameterization is replaced by ceres.Manifold.
# Since we are using 6D Logmaps as the state parameter, the AmbientSize and
# TangentSize are both 6.

class SE3Manifold(ceres.Manifold):
    """
    Ceres Manifold for 6D Pose (SE3) parameters represented by their
    6-vector tangent space (Logmap) coordinates.

    This ensures that updates (Plus) and differences (Minus) happen
    on the SE(3) manifold using Lie Group operations.
    """

    def AmbientSize(self):
        """The size of the parameter vector (global state size, 6D Logmap)."""
        return 6

    def TangentSize(self):
        """The size of the tangent vector (local update size, 6D delta)."""
        return 6

    def Plus(self, x_ptr, delta_ptr, x_plus_delta_ptr):
        """
        Implements the update rule: x_new = x + delta, where '+' is the SE(3) boxplus operator.
        x_new = Log(Exp(x) * Exp(delta)) (Right perturbation)

        Args:
            x_ptr (np.ndarray): The current 6-vector state (Logmap).
            delta_ptr (np.ndarray): The 6-vector update (tangent space increment).
            x_plus_delta_ptr (np.ndarray): Output 6-vector state.
        """
        # Convert 6-vector Logmaps to Pose3 objects
        x_pose = Pose3.Expmap(x_ptr)
        delta_pose = Pose3.Expmap(delta_ptr)

        # Apply the update: Pose_new = Pose_x * Pose_delta (Right perturbation)
        x_plus_delta_pose = x_pose.compose(delta_pose)

        # Convert back to 6-vector Logmap and store in the output buffer
        # FIX: Changed instance.Logmap() to static Pose3.Logmap(instance)
        x_plus_delta_ptr[:] = Pose3.Logmap(x_plus_delta_pose)
        return True

    def Minus(self, y_ptr, x_ptr, y_minus_x_ptr):
        """
        Implements the difference rule: y_minus_x = y - x, where '-' is the SE(3) boxminus operator.
        This calculates the tangent vector delta = Log(Y * X^-1).

        Args:
            y_ptr (np.ndarray): The target 6-vector state (Logmap).
            x_ptr (np.ndarray): The reference 6-vector state (Logmap).
            y_minus_x_ptr (np.ndarray): Output 6-vector tangent difference.
        """
        # 1. Convert 6-vector Logmaps to Pose3 objects
        y_pose = Pose3.Expmap(y_ptr)
        x_pose = Pose3.Expmap(x_ptr)

        # 2. Calculate the difference pose: Error_pose = Y * X^-1
        # This is the composition of Y with the inverse of X.
        error_pose = y_pose.compose(x_pose.inverse())

        # 3. The tangent space difference is the Logmap of the error pose
        # FIX: Changed instance.Logmap() to static Pose3.Logmap(instance)
        y_minus_x_ptr[:] = Pose3.Logmap(error_pose)
        return True

# --- 2. Custom Ceres Cost Function (Functor) ---

class CycleErrorFunctor:
    """
    Functor for the cycle closure error residual:
    r_i = Log(T_dc * T_cb_i * T_ba * T_da_i^-1)

    T_dc and T_ba are the optimized parameters (6-vectors).
    T_cb_i and T_da_i are constant measurements for this residual block.
    """
    def __init__(self, Tcb_meas, Tda_meas):
        """
        Args:
            Tcb_meas (gtsam.Pose3): The T_c_from_b measurement.
            Tda_meas (gtsam.Pose3): The T_d_from_a measurement.
        """
        self.Tcb_meas = Tcb_meas
        self.Tda_meas = Tda_meas

    def __call__(self, parameters, residuals):
        """
        Ceres automatic differentiation will call this function.

        Args:
            parameters (list of np.ndarray): List containing two 6D parameter blocks: [Tdc_log, Tba_log]
            residuals (np.ndarray): Output array to store the 6D residual vector.
        """
        Tdc_log = parameters[0]
        Tba_log = parameters[1]

        # 1. Convert the 6D log-map parameters back to Pose3 on the manifold
        Tdc = Pose3.Expmap(Tdc_log)
        Tba = Pose3.Expmap(Tba_log)

        # 2. Calculate the cycle closure error: Tdc * Tcb * Tba * Tda^-1
        # The composition operation is the multiplication defined on the Lie Group SE(3).
        Error_pose = Tdc.compose(self.Tcb_meas).compose(Tba).compose(self.Tda_meas.inverse())

        # 3. Residual is the 6D vector Logmap of the error pose
        # FIX: Changed instance.Logmap() to static Pose3.Logmap(instance)
        residuals[:] = Pose3.Logmap(Error_pose)

        return True

# --- 3. Data Generation (Reusing previous structure) ---

def generate_data(num_data_points, T_dc_gt, T_ba_gt, noise_std):
    """
    Generates synthetic data for the AXB=Y problem.
    T_d_from_a = T_d_from_c * T_c_from_b * T_b_from_a
    """
    data = []

    # Generate large measurement noise for demonstration
    noise_sigma = np.diag([noise_std] * 6)
    noise_model = gtsam.noiseModel.Gaussian.Covariance(noise_sigma)

    for _ in range(num_data_points):
        # FIX: Replace Pose3.Random() which is not available in some gtsam Python bindings.
        # Instead, generate a random 6D vector (Logmap coordinates) and convert it to Pose3.
        # We generate random rotation in [-0.5, 0.5] rad and translation in [-1.0, 1.0] meters.
        rand_log_vec = np.concatenate([
            (np.random.rand(3) - 0.5) * 1.0, # Rotation (omega)
            (np.random.rand(3) - 0.5) * 2.0  # Translation (velocity)
        ])
        Tcb_true = Pose3.Expmap(rand_log_vec)

        Tcb_meas = Tcb_true.retract(np.random.normal(0, noise_std, 6)) # Add noise using Expmap/Logmap

        # 2. Calculate T_d_from_a (Y) using the ground truth constants (A and B)
        # Y_true = A * X_true * B
        Tda_true = T_dc_gt.compose(Tcb_true).compose(T_ba_gt)

        # 3. Add noise to T_d_from_a to get the measurement Tda_meas
        Tda_meas = Tda_true.retract(np.random.normal(0, noise_std, 6))

        data.append({
            'Tcb_meas': Tcb_meas,
            'Tda_meas': Tda_meas,
        })
    return data, noise_model

# --- 4. Ceres Optimization Function ---

def run_ceres_optimization(data, T_dc_init, T_ba_init, noise_model):
    """
    Sets up and solves the calibration problem using Ceres Solver.
    """
    print("\n--- Starting Ceres Optimization ---")

    # Ceres parameters must be mutable NumPy arrays (6D log-map vectors)
    Tdc_log_optimized = Pose3.Logmap(T_dc_init).copy()
    Tba_log_optimized = Pose3.Logmap(T_ba_init).copy()

    # Create the Ceres problem
    problem = ceres.Problem()

    # 1. Add parameters and the custom SE(3) Manifold
    # NOTE: The explicit setting of the Manifold is removed because SetManifold
    # is causing an AttributeError in this pyceres environment. We rely on the
    # implicit registration by AddResidualBlock below.
    se3_manifold = SE3Manifold()

    # 2. Add Residual Blocks
    # Noise model is applied using a loss function (e.g., squared Mahalanobis distance)

    # FIX: The .Information() call failed. We retrieve the square-root information matrix (R)
    # using .R() and calculate the Information matrix (I = R.T @ R) required by Ceres.
    R_matrix = noise_model.R()
    information_matrix = R_matrix.T @ R_matrix

    loss_function = ceres.SquaredMahalanobisLoss(information_matrix)

    for i, item in enumerate(data):
        # Create the cost function (functor) for the current data point
        cost_function = ceres.AutoDiffCostFunction(
            CycleErrorFunctor(item['Tcb_meas'], item['Tda_meas']),
            6, # Number of residuals (6D vector)
            6, # Size of parameter block 1 (Tdc_log)
            6  # Size of parameter block 2 (Tba_log)
        )

        # Add the residual block to the problem
        problem.AddResidualBlock(
            cost_function,
            loss_function,  # Use the Mahalanobis loss function for weighting
            Tdc_log_optimized,
            Tba_log_optimized
        )

    # 3. Configure and Run the Solver
    options = ceres.SolverOptions()
    options.linear_solver_type = ceres.LinearSolverType.SPARSE_SCHUR
    options.trust_region_strategy_type = ceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    options.max_num_iterations = 100
    options.minimizer_progress_to_stdout = True

    summary = ceres.solve(options, problem)

    print("\nCeres Solver Report:")
    print(summary.BriefReport())

    # 4. Extract optimized poses
    Tdc_opt = Pose3.Expmap(Tdc_log_optimized)
    Tba_opt = Pose3.Expmap(Tba_log_optimized)

    return Tdc_opt, Tba_opt

# --- 5. Main Execution ---

if __name__ == '__main__':
    np.random.seed(42) # for reproducibility

    # Define Ground Truth (A=Tdc, B=Tba)
    T_dc_gt = Pose3(Rot3.RzRyRx(0.01, 0.05, 0.1), gtsam.Point3(0.5, 0.1, -0.2))
    T_ba_gt = Pose3(Rot3.RzRyRx(-0.02, 0.03, -0.05), gtsam.Point3(-0.3, 0.4, 0.1))

    # Simulation Parameters
    NUM_MEASUREMENTS = 100
    NOISE_STD_DEV = 0.1 # Significant noise for demonstration (0.1 rad/m)

    # 1. Generate Data
    data_points, noise_model = generate_data(NUM_MEASUREMENTS, T_dc_gt, T_ba_gt, NOISE_STD_DEV)

    # 2. Define Initial Estimates (Noisy start)
    # Perturb the ground truth poses to create initial guesses
    initial_perturbation = np.array([0.05, -0.05, 0.05, 0.1, -0.1, 0.1]) # 5cm/rad/m perturbation
    T_dc_init = T_dc_gt.retract(initial_perturbation)
    T_ba_init = T_ba_gt.retract(-initial_perturbation)

    # 3. Run Optimization
    T_dc_ceres, T_ba_ceres = run_ceres_optimization(data_points, T_dc_init, T_ba_init, noise_model)

    # 4. Results and Comparison

    # Function to calculate residual error (Logmap)
    def calculate_error(T_est, T_gt):
        error_pose = T_est.compose(T_gt.inverse())
        # FIX: Changed instance.Logmap() to static Pose3.Logmap(instance)
        return Pose3.Logmap(error_pose)

    # Initial Error
    err_dc_init = calculate_error(T_dc_init, T_dc_gt)
    err_ba_init = calculate_error(T_ba_init, T_ba_gt)
    init_err_norm = np.linalg.norm(err_dc_init) + np.linalg.norm(err_ba_init)

    # Final Error
    err_dc_ceres = calculate_error(T_dc_ceres, T_dc_gt)
    err_ba_ceres = calculate_error(T_ba_ceres, T_ba_gt)
    final_err_norm = np.linalg.norm(err_dc_ceres) + np.linalg.norm(err_ba_ceres)

    print("\n========================================================")
    print(f"Total Measurements: {NUM_MEASUREMENTS}")
    print(f"Measurement Noise (Logmap std dev): {NOISE_STD_DEV}")
    print("========================================================")

    # T_dc (A)
    print("\n--- T_d_from_c (A) ---")
    # FIX: Changed instance.Logmap() to static Pose3.Logmap(instance)
    print(f"Ground Truth (Logmap):\n{Pose3.Logmap(T_dc_gt)}")
    print(f"Initial Est. (Logmap):\n{Pose3.Logmap(T_dc_init)}")
    print(f"Ceres Result (Logmap):\n{Pose3.Logmap(T_dc_ceres)}")
    print(f"Initial Error (norm): {np.linalg.norm(err_dc_init):.6f}")
    print(f"Final Error (norm):   {np.linalg.norm(err_dc_ceres):.6f}")

    # T_ba (B)
    print("\n--- T_b_from_a (B) ---")
    # FIX: Changed instance.Logmap() to static Pose3.Logmap(instance)
    print(f"Ground Truth (Logmap):\n{Pose3.Logmap(T_ba_gt)}")
    print(f"Initial Est. (Logmap):\n{Pose3.Logmap(T_ba_init)}")
    print(f"Ceres Result (Logmap):\n{Pose3.Logmap(T_ba_ceres)}")
    print(f"Initial Error (norm): {np.linalg.norm(err_ba_init):.6f}")
    print(f"Final Error (norm):   {np.linalg.norm(err_ba_ceres):.6f}")

    print("\n--- Summary ---")
    print(f"Initial Total Error Norm: {init_err_norm:.6f}")
    print(f"Final Total Error Norm:   {final_err_norm:.6f}")

    print("\nCeres successfully minimized the error.")
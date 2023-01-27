"""Implementation of inverse solvers with different methods to compute
derivatives.

Solver Methods:
- Levenberg-Marquardt Algorithm;

Derivative Methods:
- Analytical formulation;
- Automatic Differentitation.
"""

import jax
import jax.numpy as jnp
from jax import jacfwd

import circle_mesh_eit
import direct_solver


def levenberg_marquardt_analytic(true_voltages,
                                 measurement_setup,
                                 lm_parameter,
                                 descent_step,
                                 max_it=100):
    """Solves inverse problem with analytical Levenberg-Marquard method

    Args:
        true_voltages: Array of shape
            (nmb_electrodes * (nmb_electrodes - 1),) containing voltage
            measurements with respect to different current patterns
            concatenated into a 1D array.
        max_it: Maximum number of interations
        lm_parameter: Levenberg-Marquardt parameter for regularization of
            Hessian matrix.

    Returns:
        anomaly: Array of shape (5, ) describing the circle parameterization
            of the conductivity with anomaly.
    """
    domain_mesh = circle_mesh_eit.CircleUniform2DMeshEit(
        measurement_setup["radius"],
        measurement_setup["nmb_electrodes"],
        measurement_setup["length_electrodes"],
        initial_edge_length = 0.042 * measurement_setup["radius"])

    eit_model = direct_solver.EitModel(domain_mesh,
                                       measurement_setup["current_amplitude"],
                                       measurement_setup["contact_impedances"],
                                       measurement_setup["current_pattern"])

    approx_anomaly = jnp.array([0.1*measurement_setup["radius"], 0., 0.,
                                1.4, 0.7])

    # Compute voltages and jacobian
    coefficient_solution, sim_voltages = eit_model.solver(approx_anomaly)

    diff_voltages = sim_voltages - true_voltages
    loss = jnp.linalg.norm(diff_voltages)/jnp.linalg.norm(true_voltages)

    k = 0

    while loss > 1e-3 and k < max_it:
        jacobian_matrix = eit_model.jacobian_analytic_non_opt(
            coefficient_solution, approx_anomaly)[:, :3]

        gradient = jnp.matmul(jacobian_matrix.T, diff_voltages)
        hessian = jnp.matmul(jacobian_matrix.T, jacobian_matrix) + \
                lm_parameter * jnp.diag(jnp.ones(3))

        delta, _ = jax.scipy.sparse.linalg.bicgstab(hessian, -gradient)

        approx_anomaly = approx_anomaly.at[:3].add(descent_step * delta)
        k = k + 1

        coefficient_solution, sim_voltages = eit_model.solver(approx_anomaly)
        diff_voltages = sim_voltages - true_voltages
        loss = jnp.linalg.norm(diff_voltages)/jnp.linalg.norm(true_voltages)

    if k < max_it:
        print("Convergence was achieved in " + str(k) + \
              "-th iterations, with a loss of: " + str(loss))
    else:
        print("More iterations are needed! Current loss: " + str(loss))

    print("Analytic Anomaly: ")
    print(approx_anomaly)

    return approx_anomaly[:3], loss, k


def levenberg_marquardt_auto_diff(true_voltages,
                                  measurement_setup,
                                  lm_parameter,
                                  descent_step,
                                  max_it=100):
    """Solves inverse problem with analytical Levenberg-Marquard method

    Args:
        true_voltages: Array of shape
            (nmb_electrodes * (nmb_electrodes - 1),) containing voltage
            measurements with respect to different current patterns
            concatenated into a 1D array.
        max_it: Maximum number of interations
        lm_parameter: Levenberg-Marquardt parameter for regularization of
            Hessian matrix.

    Returns:
        anomaly: Array of shape (5, ) describing the circle parameterization
            of the conductivity with anomaly.
    """
    domain_mesh = circle_mesh_eit.CircleUniform2DMeshEit(
        measurement_setup["radius"],
        measurement_setup["nmb_electrodes"],
        measurement_setup["length_electrodes"],
        initial_edge_length=0.042 * measurement_setup["radius"])


    eit_model = direct_solver.EitModel(domain_mesh,
                                       measurement_setup["current_amplitude"],
                                       measurement_setup["contact_impedances"],
                                       measurement_setup["current_pattern"])

    approx_anomaly = jnp.array([0.1*measurement_setup["radius"], 0., 0.])

    # Compute voltages and jacobian
    sim_voltages = eit_model.direct_operator(approx_anomaly)

    diff_voltages = sim_voltages - true_voltages
    loss = jnp.linalg.norm(diff_voltages)/jnp.linalg.norm(true_voltages)

    k = 0

    while loss > 1e-3 and k < max_it:
        jacobian_matrix = eit_model.jacobian_ad(approx_anomaly)

        gradient = jnp.matmul(jacobian_matrix.T, diff_voltages)
        hessian = jnp.matmul(jacobian_matrix.T, jacobian_matrix) + \
                lm_parameter * jnp.diag(jnp.ones(3))

        delta, _ = jax.scipy.sparse.linalg.bicgstab(hessian, -gradient)

        approx_anomaly = approx_anomaly.at[:].add(descent_step * delta)
        k = k + 1

        sim_voltages = eit_model.direct_operator(approx_anomaly)
        diff_voltages = sim_voltages - true_voltages
        loss = jnp.linalg.norm(diff_voltages)/jnp.linalg.norm(true_voltages)

    if k < max_it:
        print("Convergence was achieved in " + str(k) + \
              "-th iterations, with a loss of: " + str(loss))
    else:
        print("More iterations are needed! Current loss: " + str(loss))

    print("AD Anomaly: ")
    print(approx_anomaly)

    return approx_anomaly, loss, k

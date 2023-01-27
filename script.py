"""Test run of effectiveness between analytic and AD differentiation."""

import os
from absl import logging, app

from time import time

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

import circle_mesh_eit as mesher
import direct_solver
import inverse_solvers


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_platform_name", "gpu")

def main(_):

    nmb_electrodes = 32
    radius = 1

    # Measurement Setup
    measurement_setup = {
        "radius": radius,
        "nmb_electrodes": nmb_electrodes,
        "length_electrodes": radius * jnp.pi/(2 * nmb_electrodes),
        "current_amplitude": 3.,
        "contact_impedances": (5e-5) * jnp.ones(nmb_electrodes),
        "current_pattern": "trignometric"
        }

    domain_mesh = mesher.CircleUniform2DMeshEit(
        measurement_setup["radius"],
        measurement_setup["nmb_electrodes"],
        measurement_setup["length_electrodes"],
        initial_edge_length=0.04 * measurement_setup["radius"]
        )

    sim_model = direct_solver.EitModel(domain_mesh,
                                       measurement_setup["current_amplitude"],
                                       measurement_setup["contact_impedances"],
                                       measurement_setup["current_pattern"])

    print("Points = "+ str(domain_mesh.nmb_points) + " triangles = " + \
            str(domain_mesh.nmb_triangles) + "\n")
    print("Simulate measurements! \n")

    #current_directory = os.getcwd()
    #results_path = os.path.join(current_directory, "figures/")

    # Random Generation of Anomalies
    nmb_experiments = 10
    key = random.split(random.PRNGKey(12), 5)

    center_angles = random.uniform(key[0], (nmb_experiments,),
                                   minval=0, maxval=2 * jnp.pi)
    center_rad = random.uniform(key[1], (nmb_experiments,),
                                minval=0., maxval=measurement_setup["radius"])

    c_x = center_rad * jnp.cos(center_angles)
    c_y = center_rad * jnp.sin(center_angles)

    diff_radius = measurement_setup["radius"] - center_rad
    anomaly_radius = random.uniform(key[2], (nmb_experiments,), minval=0.1,
                                    maxval=diff_radius)

    conductivity_inside = random.uniform(key[3], (nmb_experiments,),
                                         minval=0.7, maxval=1.6)
    conductivity_outside = random.uniform(key[4], (nmb_experiments,),
                                          minval=0.7, maxval=1.6)

    anomalies = jnp.array([anomaly_radius, c_x, c_y,
                           1.4*jnp.ones(nmb_experiments),
                           0.7*jnp.ones(nmb_experiments)]).T
    """anomalies = jnp.array([anomaly_radius, c_x, c_y,
                           conductivity_inside,
                           conductivity_outside]).T"""

    meas_error = np.zeros((nmb_experiments, 2))
    time_elapsed = np.zeros((nmb_experiments, 2))
    anomalies_recon = np.zeros((nmb_experiments, 2, 3))
    anomaly_error = np.zeros((nmb_experiments, 2, 3))
    it = np.zeros((nmb_experiments, 2))


    lm_parameter = 10
    descent_step = 1.
    max_it = 20


    for k in range(nmb_experiments):
        print("Exp " + str(k))
        print(anomalies[k, :])
        voltages = sim_model.direct_operator(anomalies[k, :3])

        start_time = time()
        anomalies_recon[k, 0, :], meas_error[k, 0], it[k, 0] = \
            inverse_solvers.levenberg_marquardt_analytic(
                voltages, measurement_setup,
                lm_parameter,
                descent_step,
                max_it)
        end_time = time()
        time_elapsed[k, 0] = end_time - start_time
        anomaly_error[k, 0, :] = jnp.abs(
            anomalies[k,:3] - anomalies_recon[k, 0])

        start_time = time()
        anomalies_recon[k, 1, :], meas_error[k, 1], it[k, 1] = \
            inverse_solvers.levenberg_marquardt_auto_diff(
                voltages, measurement_setup,
                lm_parameter,
                descent_step,
                max_it)
        end_time = time()
        time_elapsed[k, 1] = end_time - start_time
        anomaly_error[k, 1, :] = jnp.abs(anomalies[k,:3] - \
                                         anomalies_recon[k, 1])

    with open("statistics_experiments_partial.txt", "w", encoding="utf-8") as f:
        f.write("LM method for Radius and Center - Non-optimal Analytic!\n")
        f.write("\n")

        f.write("Analytic Statistics: \n\n")
        f.write("Measurement loss:\n ")
        f.write(np.array2string(meas_error[:, 0]))
        f.write("\n\n")
        f.write("Average Measurement loss: " + str(np.mean(meas_error[:, 0])))
        f.write("\n\n Time Elapsed: \n")
        f.write(np.array2string(time_elapsed[:, 0]))
        f.write("\n\n Average Time Elapsed: " + \
                str(np.mean(time_elapsed[:, 0])))
        f.write("\n\n Anomaly loss: \n")
        f.write(np.array2string(anomaly_error[:, 0]))
        f.write("\n\n Average Anomaly Difference: " + \
                str(np.array2string(np.mean(anomaly_error[:, 0, :], axis=0))))

        f.write("\n \n Auto-Diff Statistics: \n\n")
        f.write("Measurement loss:\n ")
        f.write(np.array2string(meas_error[:, 1]))
        f.write("\n\n")
        f.write("\n\n Average Measurement loss: " + \
                str(np.mean(meas_error[:, 1])))
        f.write("\n\n Time Elapsed: \n")
        f.write(np.array2string(time_elapsed[:, 1]))
        f.write("\n\n Average Time Elapsed: " + \
                 str(np.mean(time_elapsed[:, 1])))
        f.write("\n\n Anomaly loss: \n")
        f.write(np.array2string(anomaly_error[:, 1]))
        f.write("\n\n Average Anomaly Difference: " + \
                str(np.array2string(np.mean(anomaly_error[:, 1, :], axis=0))))

    with open("anomalies_experiments_partial.txt", "w", encoding="utf-8") as g:
        g.write("True Anomalies // Analytic Anomaly// AD Anomaly\n\n")

        for k in range(nmb_experiments):
            g.write(np.array2string(anomalies[k, :3]))
            g.write("  //  ")
            g.write(np.array2string(anomalies_recon[k, 0, :]))
            g.write("  //  ")
            g.write(np.array2string(anomalies_recon[k, 1, :]))
            g.write("\n\n")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)

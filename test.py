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

    nmb_electrodes = 16

    # Measurement Setup
    measurement_setup = {
        "radius": 1.,
        "nmb_electrodes": nmb_electrodes,
        "length_electrodes": jnp.pi/(2 * nmb_electrodes),
        "current_amplitude": 3.,
        "contact_impedances": (5e-5) * jnp.ones(nmb_electrodes),
        "current_pattern": "trignometric"
        }

    domain_mesh = mesher.CircleUniform2DMeshEit(
        measurement_setup["radius"],
        measurement_setup["nmb_electrodes"],
        measurement_setup["length_electrodes"],
        initial_edge_length=0.04
        )

    sim_model = direct_solver.EitModel(domain_mesh,
                                       measurement_setup["current_amplitude"],
                                       measurement_setup["contact_impedances"],
                                       measurement_setup["current_pattern"])

    print("Points = "+ str(domain_mesh.nmb_points) + " triangles = " + \
            str(domain_mesh.nmb_triangles) + "\n")
    print("Simulate measurements! \n")

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
                           conductivity_inside,
                           conductivity_outside]).T

    ad_jacobian = jax.jit(jax.jacfwd(sim_model.direct_operator_))

    time_elapsed = np.zeros((nmb_experiments, 3))
    max_error = np.zeros((nmb_experiments))

    for k in range(nmb_experiments):
        fem_sol, _ = sim_model.solver(anomalies[k, :])
        start_time = time()
        j_analytic = sim_model.jacobian_analytic_non_opt(
            fem_sol, anomalies[k, :])
        end_time = time()
        an_time = end_time - start_time
        print("\n Non-Optimized Analytic Time: " + str(an_time))
        start_time = time()
        j_analytic = sim_model.jacobian_analytic(fem_sol, anomalies[k, :])
        end_time = time()
        opt_an_time = end_time-start_time
        print("\n Optimized Analytic Time: " + str(opt_an_time))
        start_time = time()
        j_ad = ad_jacobian(anomalies[k, :])
        end_time = time()
        ad_time = end_time-start_time
        print("\n Auto-Diff Time: " + str(ad_time))

        print("\n Supremum Norm: ")
        print(jnp.max(jnp.abs(j_ad-j_analytic)))
        max_error[k] = jnp.max(jnp.abs(j_ad - j_analytic) )
        time_elapsed[k, :] = jnp.array([an_time, opt_an_time, ad_time])

    with open("Jacobian_experiments.txt", "w", encoding="utf-8") as f:
        f.write("Anomalies || Max_Error || Analytic || Opt-Analytic || AD: \n")
        for k in range(nmb_experiments):
            f.write(np.array2string(anomalies[k, :]))
            f.write(str(max_error[k]))
            f.write(" || ")
            f.write(np.array2string(time_elapsed[k, :]))
            f.write("\n")

        f.write("Average Error = " + str(np.mean(max_error[1:])))
        f.write("Average Times = " + \
            np.array2string(np.mean(time_elapsed[1:,:], axis=0)))


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)

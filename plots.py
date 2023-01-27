"""Test run of effectiveness between analytic and AD differentiation."""

import os
from absl import logging, app

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import circle_mesh_eit as mesher
import conductivity


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
        initial_edge_length=0.042
        )

    print("Points = "+ str(domain_mesh.nmb_points) + " triangles = " + \
            str(domain_mesh.nmb_triangles) + "\n")
    print("Simulate measurements! \n")

    anomaly = jnp.array([[0.16138829, 0.36461025, 0.34408662],
                         [0.17612889, 0.30777156, 0.26452026],
                         [0.17709824, 0.30604666, 0.2632364 ]])


    # Splits coordinates of mesh points.
    x_pt, y_pt = domain_mesh.points.T[:]

    fig, axes = plt.subplots((1, 3))

    fig.suptitle("Conductivity Anomalies in Disk of Radius 1")
    axes[0].set_title("True = [0.161, 0.365, 0.344]")
    axes[1].set_title("Analytic = [0.176, 0.308, 0.265]")
    axes[2].set_title("Auto-Diff = [0.177, 0.306, 0.263]")
    # Plot the elements of the mesh and color them based on conductivity value
    #axes.triplot(x_pt, y_pt, mesh.mesh_triangles, "-k", linewidth=0.2)
    for k in range(3):
        conductivity_array = conductivity.create_conductivity_array(
            anomaly[k, :], domain_mesh)
        colors = axes[0, k].tripcolor(x_pt,
                            y_pt,
                            domain_mesh.mesh_triangles,
                            facecolors=conductivity_array,
                            cmap="GnBu")

        # Add the electrodes to the plot.
        for index_ele in domain_mesh.electrode_pts:
            axes[0, k].plot(domain_mesh.points[index_ele, 0],
                    domain_mesh.points[index_ele, 1],
                    "k",
                    linewidth=3)

    plt.colorbar(colors, ax=axes)

    output_directory = os.getcwd()
    file_name = "Experiment_partial_1.png"
    fig.savefig(os.path.join(output_directory, file_name))

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)

"""Definition of conductivity through a circular anomaly in 2D.

Here, we define the conductivity in a disk of radius 1 with an anomaly.
The anomaly is circular (for now) and is defined by three parameters
[r, center_x, center_y], which correspond the radius and center coordinates.

We define it by level-set theory with the help of the Heaviside function and
the level-set function r^2 - ((x - center_x)^2+(y - center_y)^2). This function
is positive if the point (x, y) is inside the anomaly, negative if outside
and 0 if it is on the boundary.

See more on the following links:
https://en.wikipedia.org/wiki/Level_set
https://en.wikipedia.org/wiki/Level-set_method

Furthermore, the conductivity is defined in a piece-wise constant manner, by
setting a value for each element of a two-dimensional mesh.
Hereby, the mesh is an object type and contains the following attributes:
 - mesh.points: Array of shape (N x 2) with the coordinates of the N
    2-dimensional points of the mesh.
 - mesh.elements: Array of shape (K x 3) that contains the
    the indexes of the points defining each triangular element of the mesh.
 - mesh.pts_electrodes: list of indexes for the points on each electrode.
"""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt


def smoothened_heaviside(x_pt):
    """Smoothed and differentiable Heaviside function.

    Here we provide an approximation of the Heaviside function through a
    smoothing procedure close to the jump at x_pt=0. The smoothing is done by:

                 H(x_pt) ~= 1 / pi * arctan(x_pt / smoothing_length) + 1 / 2.

    This function is infinitely differentiable and as the parameter
    smoothing_length goes to zero we get closer to the Heaviside function.

    Args:
        x_pt: 1D array of floats.
        smoothing_length: Length over which the jump at x_pt = 0 is smoothed.
            It must be a positive number.

    Returns:
        heaviside_values: 1D Array with smoothed heaviside function
            evaluated at input x_pt.
    """

    smoothing_length = 0.01

    heaviside_values = (1 / jnp.pi) * jnp.arctan2(x_pt,
                                                  smoothing_length) + 1 / 2

    return heaviside_values


def derivative_heaviside(x_pt):
    """Derivative of approximate heaviside function.

    (H(x_pt))' = (1 / (pi*smoothing_length)) *
                    (1 / (1 + ( x_pt / smoothing_length)^2 ) )

    Args:
        x_pt: 1D array of floats.
        smoothing_length: Length over which the jump at x_pt = 0 is
          smoothed. It must be a positive number.

    Returns:
        derivative_values: 1D Array with derivative values of smoothed
            heaviside function evaluated at input x_pt.
    """

    smoothing_length = 0.01

    derivative_values = (1 / (jnp.pi * smoothing_length)) * \
            (1 / (1 + jnp.power(x_pt / smoothing_length, 2)))

    return derivative_values


def compute_derivative_conductivity(mesh, anomaly):
    """Computes derivative of conductivity over elements wrt anomaly.

    Args:
        mesh: Object of type Mesh, with attributes corresponding to
            points (mesh.pts - (N x 2) array) and elements
            (mesh.element - (K x 3) array).
        anomaly: Array of shape (5,) with parameterization of circular
            conductivity anomaly.

    Returns:
        jacobian_conductivity: Array of shape (mesh.nmb_triangles, 5) containing
            conductivity derivative over each element with respect to anomaly
            parameters.
    """

    jacobian_conductivity = jnp.zeros((mesh.nmb_triangles, 5))

    # Evaluate smooth heaviside function through level set at element centroids
    element_pts = mesh.points[mesh.mesh_triangles, :]
    centroids = jnp.mean(element_pts, axis=1)
    level_set_values = circle_level_set(anomaly, centroids)
    heaviside_values = smoothened_heaviside(level_set_values)

    # Derivatives wrt to conductivity in and out.
    jacobian_conductivity = jacobian_conductivity.at[:, 3].set(heaviside_values)
    jacobian_conductivity = jacobian_conductivity.at[:, 4].set(1 -
                                                               heaviside_values)

    heaviside_derivative = derivative_heaviside(level_set_values)

    center_xy_derivative = 2 * (centroids - anomaly[1:3])

    # Chain-rule by hand for radius
    jacobian_conductivity = jacobian_conductivity.at[:, 0].set(
        (anomaly[3] - anomaly[4]) * heaviside_derivative * 2 * anomaly[0])

    # Chain-rule by hand for center_x, center_y
    jacobian_conductivity = jacobian_conductivity.at[:, 1:3].set(
        (anomaly[3] - anomaly[4]) *
        (center_xy_derivative.T * heaviside_derivative).T)

    return jacobian_conductivity


def circle_level_set(anomaly, points):
    """Level-set function of the circle of radius r and center (px, py).

    A level-set is any function that is positive inside the domain it
    describes, negative outside of it and 0 on its boundary.
    Here, in particular, it is zero over the circumference.

    Args:
        anomaly: namedtuple composed of the radius and center of circular
            anomaly.
        points: array of shape (N x 2) with 2D points..

    Returns:
        1D Array with level-set function evaluated at input points.
    """

    # TODO(Ivan) Check that anomaly is inside the mesh.

    return anomaly[0]**2 - jnp.sum(
        jnp.power(points - jnp.array([anomaly[1], anomaly[2]]), 2), 1)


#@partial(jax.jit, static_argnums=(1, ))
def create_conductivity_array(anomaly, mesh):
    """Definition of a piece-wise conductivity over the mesh elements.

    Conductivity values are pre-determined inside and outside anomaly.

    Args:
        anomaly: namedtuple composed of the radius and center of circular
            anomaly, conductivity values inside and outside the anomaly.
        smoothing_length: Length over which the jump at x = 0 is smoothed.
            It must be a positive number.
        mesh: Object of type Mesh, with attributes corresponding to
            points (mesh.pts - (N x 2) array) and elements
            (mesh.element - (K x 3) array).

    Returns:
        1D Array with shape (K x 1) that defines the piece-wise constant
        conductivity values over the mesh elements.
    """

    # Create three dimensional array of shape: (Number_of_elements, N, 2)
    # which contains for each element its N points that are 2D points.
    element_pts = mesh.points[mesh.mesh_triangles, :]

    # Compute the centroid (middle point) of each element through
    # the mean of its points.
    centroids = jnp.mean(element_pts, axis=1)

    # Evaluate the center points through level-set function.
    level_set_values = circle_level_set(anomaly, centroids)

    # Check through signal of level_set_values if elements are inside or
    # outside the anomaly. The function smoothed_heaviside applied to
    # level_set_values outputs 1 for elements clearly inside the anomaly,
    # 0 for the ones outside, and a value in between for elements near the
    # boundary.
    heaviside_values = smoothened_heaviside(level_set_values)

    # Chooses the value of conductivity for each element from
    # conductivity_in, conductivity_out.
    return anomaly[4] * (1 - heaviside_values) + \
        anomaly[3] * heaviside_values


def plot_conductivity(mesh, conductivity, file_name, output_directory=None):
    """Outputs a figure of piece-wise conductivity over mesh elements.

    In here, we give a plot of the piece-wise constant conductivity over
    mesh elements.

    Args:
        mesh: Object of type Mesh, with attributes corresponding to
            points (mesh.points), elements (mesh.mesh_triangles), and
            elements defining electrodes (mesh.ix_ele).
        conductivity: 1D Array with conductivity values over mesh elements.
        output_path: string path where the figure is saved.
    """

    # Splits coordinates of mesh points.
    x_pt, y_pt = mesh.points.T[:]

    fig, axes = plt.subplots()

    axes.set_title("Plot of piece-wise conductivity")
    # Plot the elements of the mesh and color them based on conductivity value
    #axes.triplot(x_pt, y_pt, mesh.mesh_triangles, "-k", linewidth=0.2)
    colors = axes.tripcolor(x_pt,
                            y_pt,
                            mesh.mesh_triangles,
                            facecolors=conductivity,
                            cmap="GnBu")

    # Add the electrodes to the plot.
    for index_ele in mesh.electrode_pts:
        axes.plot(mesh.points[index_ele, 0],
                  mesh.points[index_ele, 1],
                  "k",
                  linewidth=3)

    plt.colorbar(colors, ax=axes)

    if output_directory is not None:
        fig.savefig(os.path.join(output_directory, file_name))
    else:
        plt.show()

    plt.close()


def plot_anomaly(mesh, anomaly, file_name, output_directory=None):
    """Outputs a figure of piece-wise conductivity over mesh elements.

    In here, we give a plot of the piece-wise constant conductivity over
    mesh elements.

    Args:
        mesh: Object of type Mesh, with attributes corresponding to
            points (mesh.points), elements (mesh.mesh_triangles), and
            elements defining electrodes (mesh.ix_ele).
        anomaly: Array of shape (5,) with conductivity parameterization.
        file_name: Name of .png file to save in image.
        output_path: string path where the figure is saved.
    """

    conductivity = create_conductivity_array(anomaly, mesh)

    # Splits coordinates of mesh points.
    x_pt, y_pt = mesh.points.T[:]

    fig, axes = plt.subplots()

    axes.set_title("Plot of piece-wise conductivity")
    # Plot the elements of the mesh and color them based on conductivity value
    #axes.triplot(x_pt, y_pt, mesh.mesh_triangles, "-k", linewidth=0.2)
    colors = axes.tripcolor(x_pt,
                            y_pt,
                            mesh.mesh_triangles,
                            facecolors=conductivity,
                            cmap="GnBu")

    # Add the electrodes to the plot.
    for index_ele in mesh.electrode_pts:
        axes.plot(mesh.points[index_ele, 0],
                  mesh.points[index_ele, 1],
                  "k",
                  linewidth=3)

    plt.colorbar(colors, ax=axes)

    if output_directory is not None:
        fig.savefig(os.path.join(output_directory, file_name))
    else:
        plt.show()

    plt.close()

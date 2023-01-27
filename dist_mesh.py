""" Defines a mesh (triangular and 2D) through the DistMesh algorithm.

Here, we provide the DistMesh algorithm created by Per-Olof Persson (Berkeley)
and Gilbert-Strang (MIT).

This algorithm uses a level-set function, i.e.,
a function that is negative inside the region, and positive outside
(or vice-versa), to create an unstructed triangular mesh in 2D. For the
actual mesh generation we use Delaunay triangulation routine and try to
optimize the points locations by a force-based smoothing procedure. That is,
we update until we achieve equilibrium of forces at each points..

For such, we regularly update the topology with Delaunay and force the
boundary points to move tangentially to the boundary by projecting with
the level-set function. This iterative procedure typically results in
very well-shaped meshes.

References:
http://persson.berkeley.edu/distmesh/


In the end, the DistMesh algorithm outputs an array of shape (N x 2) that
contains all the points that define the mesh and an array of shape (K x 3)
that contains the indexes of points that define each element.
"""

import math

import numpy as np
from numpy.linalg import norm

import scipy.sparse
import scipy.spatial


class MeshControlParameters:
    """Defines the parameters that control the dist_mesh_2d algorithm flow.

    Attributes:
        exit_tolerance: tolerance for the exiting condition.
        triangulate_tolerance: tolerance for the movement of the points.
            If movement is large, then triangulate again with Delaunay.
        force_scale: scale to enforce repulsive forces at each point.
        euler_time_step: time step in the Euler method for updating the
            new points.
        geometry_tolerance: tolerance of geometry evaluations.
        derivative_step: spatial step to compute derivative of level_set
            function of domain.

    Other args:
        initial_edge_length: parameter that defines initial edge length
            from which some of the control parameters depended.
    """

    def __init__(
        self,
        initial_edge_length,
        exit_tolerance=0.001,
        triangulate_tolerance=0.1,
        force_scale=1.2,
        euler_time_step=0.2,
    ):

        self.exit_tolerance = exit_tolerance
        self.triangulate_tolerance = triangulate_tolerance
        self.force_scale = force_scale
        self.euler_time_step = euler_time_step

        self.geometry_tolerance = 0.001 * initial_edge_length
        self.derivative_step = np.sqrt(np.finfo(np.double).eps) * \
            initial_edge_length


def dist_mesh_2d(control_parameters, level_set, relative_edge_length,
                 initial_edge_length, domain_box, pts_fixed):
    # pylint: disable=W1401
    """Defines a mesh through the DistMesh algorithm.


    Steps of Algorithm:
    1. Creates in the domain box a rectangular grid of evenly spaced points in x
    and y with steps initial_edge_length and sqrt(3)/2 * initial_edge_length,
    respectively.

    Shifts odd lines of x by initial_edge_length to obtain a grid that forms
    equilateral triangles by connecting the points. That is,

                    . . . . .         . . . . .
                    . . . . .          . . . . .
                    . . . . .   --->  . . . . .
                    . . . . .          . . . .
                    . . . . .         . . . . .

    NOTE: Here, the forces at each inner point are zero. But by
    picking only the ones inside the domain, this will not happen anymore and
    an update is needed. The forces are given as:

    (a)   Forces(point) = total_internal_Forces(point) + external_Forces(point),

    where the internal_Forces arise from the connections with other points and
    the external_Forces arise from reactions with the boundary.


    2. First, selects points that are inside the desired domain under a
    certain [geometric tolerance], by checking if their level_set values are
    negative.
    Secondly, establishes a random selection criterium based on the
    desired density arising from the relative_length_function.

    NOTE: This function allows us to take out points in regions where we
    do not need so much, to have higher density in troublesome regions.

    E.g. For relative_length_function = 1, i.e. constant everywhere,
    it selects all of the points generated.

    Stacks with the set of desired fixed points. In order to start the loop.

    3. Enters `while True` loop that only breaks when forces at each point are
    0.

    At the first iteration or when the update moved at least one point more than
    a certain fraction of the [initial_edge_length], controled by
    [triangulate_tolerance], we need to triangulate the points again.

    The triangulation is done by the re_delaunay function. For such, we use
    the inner Delaunay triangulation of scipy. Then we keep only the inner
    triangles and select the set of edges that belong to the triangles and
    form the mesh.

                              . ___ .
                             / \   / \    <- Represents
                            /   \ /   \   <- an edge
         Point of mesh --> . ___ . ___ .
                            \   / \   /
                             \./___\./

    --> Each triangle is represented by three points.
    --> An edge can belong to more than one triangle.

    4. With the indexes of the points that form the edges, we compute the
    2d vectors and their lengths that define the direction of each edge.

    From this, we are able to compute the force exerted by each edge on its
    points. This correspond to the inner forces, which are computed in
    terms of "desired edge length" l0, see below. Here l is the current length.

                     { (l0 - l) ,  if l < l0
    forces(l, l0) =  {
                     { 0,              if l >= l0

    NOTE: In order to obligate the points spread across the geometry, most of
    the bars should give repulsive forces, i.e., forces > 0. For such, we
    choose l0 to be a bit larger than the edge length actually desired. A good
    default in 2D is 20%, which is controlled by [force_scale]
    (by default = 1.2).

    5. After computing the set of inner forces, we can join them together to
    have the set of total inner forces actuating at each point. This is what
    gives the internal forces in equation (a) at each point.

    Here, we do not yet have the external forces. But, we already take a step
    to balance the forces at each point. For such, we solve

    (b)                  total_inner_Forces(points) = 0,

    by introducing artificial time, that is:

    (c)                  d points / d t = total_inner_Forces(points).
                         points(0) = initial_points

    If a stationary solution of (c) is found, then equation (b) is immediately
    fulfilled. Thus, we can use forward Euler Method to update the set of points

    (d) new_points = old_points +
                [euler_time_step] * total_inner_Forces(old_points)

    NOTE: Recall that a stationary solution represents that the points are not
    moving in time anymore. The existence of such solution is achieved by the
    choice of the forces function that brings the equilibrium to the desired
    edge length.

    6. Finally, the external forces take part of the equilibrium in the
    following way: all points that end up outside in the last update are
    moved back to the closest boundary point. This conforms with the forces
    acting normal to the boundary and makes the points only able to move
    along it and not go outside.
                                 .
                                 |
                                 v
                              . _._ .
                             / \   / \          Points outside moved
                            /   \ /   \         to boundary.
                      . -->. ___ . ___ .
                            \   / \   /
                             \./___\./


    7. After the update, we check if the movement of the internal points
    was large enough or if  we have already attained total_forces(points) = 0.

    This can be done with [euler_time_step] * total_inner_forces[old_points],
    since this represents the movement at each point.

    Args:
        level_set: function handle for level-set of boundary.
        relative__edge_length: function handle for relative edge length
            distribution over domain.
        initial_edge_length: initial edge length of points distribution.
        domain_box: Array of shape (2 x 2) that defines a bounding square for
            domain.
        pts_fixed: Array of shape (M x 2) with set of fixed points.

    Returns:
        points: Array of shape (N x 2) of final mesh points.
        mesh_triangles: Array of shape (K x 3) with indexes of points that
        define each element.
    """
    # pylint: enable=W1401

    # 1. Creates grid with points forming equilateral triangle in bounding box.

    # Forms evenly space rectangular grid
    x, y = np.meshgrid(
        np.arange(domain_box[0, 0], domain_box[0, 1], initial_edge_length),
        np.arange(domain_box[1, 0], domain_box[1, 1],
                  initial_edge_length * math.sqrt(3) / 2.0))

    # Shift odd lines to the right, so that points form equilateral triangles
    x[1::2, :] += initial_edge_length / 2.0
    points = np.array([x.ravel(), y.ravel()]).T

    # 2. Selects points based on domain and density

    # Selects the points inside the domain and outside but very near
    # the boundary.
    points = points[level_set(points) < control_parameters.geometry_tolerance]

    # Randomly selects points to define regions with more or less density.
    prob = 1 / relative_edge_length(points)**2
    selection = np.random.rand(points.shape[0], 1) <= (prob / np.max(prob))
    points = points[selection[:, 0], :]

    # Stack fixed points with the ones generated.
    points = np.vstack((pts_fixed, points))  # Add fixed points
    nmb_pts = len(points)
    pts_old = np.inf * np.ones((nmb_pts, 2))  # Initialize previous points

    # 3. Enters updating loop to triangulate points and equilibrate forces.

    # Termination Criterium: all interior points move less than a defined
    # tolerance.
    while True:

        # Re-triangulates if at least the movement of one points
        # was large enough (scaled).
        if max_l2_norm(points - pts_old) / initial_edge_length > \
            control_parameters.triangulate_tolerance:

            pts_old = points
            # Re-Delaunay and obtain the mesh_triangles and edges.
            mesh_triangles, edges = re_delaunay(
                points, level_set, control_parameters.geometry_tolerance)

        # 4. Computes the set of inner force vectors that are spread
        # through the mesh, based on the edges.

        # Calculates the vectors corresponding to each edge and their
        # respective lengths. Recall that the forces will depend on the
        # current on edges length and vector.
        edges_vector, edges_length = calculate_vector_and_length_edges(
            points, edges)

        # Finds set of inner forces applied to every point,
        # in the strength and vector form, for convenience.
        inner_forces, inner_forces_vector = find_inner_forces(
            points, edges, relative_edge_length, edges_vector, edges_length,
            control_parameters.force_scale)

        # 5. Computes total inner forces and update points position.

        # For each point, adds all of the forces applied to it into
        # a single force. Array of shape (N x 2) where each line is
        # the vector of the internal force at the corresponding point.
        total_inner_forces = compute_total_inner_forces(inner_forces,
                                                        inner_forces_vector,
                                                        nmb_pts, edges)

        # Sets forces at fixed points to 0. As its name says, they are
        # fixed, thus we do not want them to move.
        total_inner_forces[0:len(pts_fixed), :] = 0

        # Forward Euler-Method iteration. Here, time is artificially
        # introduced to obtain an ODE solved by the following method.
        points = points + control_parameters.euler_time_step * \
            total_inner_forces

        # 6. Projects points ending up outside to the boundary. This
        # is the effect of the external forces, that is, forces exerted
        # by the boundary.
        points = project_inside(points, level_set,
                                control_parameters.derivative_step)

        boundary_distance = level_set(points)

        # 7. Checks if total inner forces caused enough movement or
        # we have total_inner_forces[points]  ~= 0.
        # Excludes boundary movements.
        norm_added_forces = max_l2_norm(
            control_parameters.euler_time_step * \
            total_inner_forces[boundary_distance < - \
            control_parameters.geometry_tolerance])

        # Checks termination criterium.
        if norm_added_forces / initial_edge_length < \
           control_parameters.exit_tolerance:

            break

    # Sorts indexes of each triangle (increasing) and
    # return the unique ones.
    mesh_triangles = np.unique(np.sort(mesh_triangles, 1), axis=0)

    return points, mesh_triangles


# pylint: disable=W1401
def compute_total_inner_forces(inner_forces, inner_forces_vector, nmb_pts,
                               edges):
    """Adds up the forces present at each point.

    We use scipy.sparse.csr_matrix due to its inherent summation
    property for duplicated indices.
    For simplicity, we provide an example:

    Let us assume that we have three points with indexes {i, s, t},
    (ordered), with edges=[[i, s], [i, t], [s, t]].

                        i ._______. s
                           \     /
                            \   /
                             \./ t

    Forces at point i: Force_is, Force_it; (represent forces arising from
    the edges is and it);
    Forces at point s: -Force_is, Force_st;
    Forces at point t: -Force_it, -Force_st.

    Objetive is to obtain:

    total_inner_forces_at_points[i, :] = Force_is + Force_it
    total_inner_forces_at_points[s, :] = - Force_is + Force_st
    total_inner_forces_at_points[t, :] = - Force_it - Force_is

    We introduce the array_of_forces as an horizontal stack like follows:

    array_of_forces = [[Force_is, -Force_is],
                       [Force_it, -Force_it],
                       [Force_st, -Force_st]]

    Further, the array of rows and cols are helpers to add up the functions.
    In 2D, they follow the following syntax:

    rows = [[i, i, s, s],               cols = [[0, 1, 0, 1],
            [i, i, t, t],                       [0, 1, 0, 1],
            [s, s, t, t]].                      [0, 1, 0, 1]].

    Finally, this is where csr_matrix comes in handy.
    First, we flatten the above arrays. Recall each force is 2D.
    The csr_matrix does the following process:

        total_inner_forces[rows[k], cols[k]] = array_of_forces[k],

    If two pairs are equal, [rows[k], cols[k]] = [rows[j], cols[j]],
    the csr_matrix function adds up to the previous position, i.e.,
    array_of_forces[k] + array_of_forces[j].

    k=0: [rows[0], cols[0]] = [i, 0] --> array_of_forces[0] = Force_is[0]
    k=1: [rows[1], cols[1]] = [i, 1] --> array_of_forces[1] = Force_is[1]
    k=2: [rows[2], cols[2]] = [s, 0] --> array_of_forces[2] = - Force_is[0]
    k=3: [rows[3], cols[3]] = [s, 1] --> array_of_forces[3] = - Forces_is[1]

    Current total_inner_forces_at points:
        total_inner_forces_at_points[i, :] = Force_is
        total_inner_forces_at_points[s, :] = - Force_is
        total_inner_forces_at_points[t, :] = 0

    k=4: [rows[4], cols[4]] = [i, 0] --> array_of_forces[4] = Forces_it[0]
    k=5: [rows[5], cols[5]] = [i, 1] --> array_of_forces[5] = Forces_it[1]
    k=6: [rows[6], cols[6]] = [t, 0] --> array_of_forces[6] = - Forces_it[0]
    k=7: [rows[7], cols[7]] = [t, 1] --> array_of_forces[7] = - Forces_it[1]

    Current total_inner_forces_at points:
        total_inner_forces_at_points[i, :] = Force_is + Force_it
        total_inner_forces_at_points[s, :] = - Force_is
        total_inner_forces_at_points[t, :] = - Force_it

    etc...

    In the end, we transform the csr_matrix into an array to be proper to use.

    Args:
        inner_forces: Array of shape (J x 1) with the force value applied
            by a certain edge.
        inner_forces_vector: Array of shape (J x 2) with the force vector
            applied by a certain edge.
        nmb_pts: total number of points currently in the mesh.
        edges: Array of shape (J x 2) with indexes of points forming each
            edge of the mesh.

    Returns:
        Array of shape (nmb_pts x 2) with the total_inner_forces at each
        point.
    """
    # pylint: enable=W1401

    # data is an array of shape (J x 4) where each line is
    # inner forces vector and - inner forces vector.
    data = np.hstack([inner_forces_vector, -inner_forces_vector])

    # rows is an array of shape (J x 4) where rows[k, :] = [s, s, j, j] for
    # edges[k, :] = [s, j], with s, j indexes.
    rows = edges[:, [0, 0, 1, 1]]

    # cols is an array of shape (J x 4) where each line is [0, 1, 0, 1]
    cols = np.dot(np.ones(np.shape(inner_forces)), np.array([[0, 1, 0, 1]]))

    # total_inner_forces_matrix[rows[k], cols[k]] = data[k]. When we have
    # (rows[k], cols[k])=(rows[j], cols[j]), we sum up in the corresponding
    # position data[k] + data[j]. THIS IS THE MAIN REASON TO USE CSR_MATRIX.
    total_inner_forces_matrix = scipy.sparse.csr_matrix(
        (data.ravel(), [rows.ravel(), cols.ravel()]), shape=(nmb_pts, 2))

    return total_inner_forces_matrix.toarray()


def find_inner_forces(points, edges, relative_edge_length, edges_vector,
                      edges_length, force_scale):
    """Computes the inner forces that arise from each edge.

    Args:
        points: Array of shape (N x 2) with the points in the mesh.
        edges: Array of shape (J x 2) with indexes of points forming edges.
        relative_edge_length: function handle for relative edge length
            distribution over domain.
        edges_vector: Array of shape (J x 2) with the vectors in 2D for
            each edge.
        edges_length: Array of shape (J x 1) with lengths of each edge.
        force_scale: scale to enforce repulsive forces at each point.

    Returns:
        Arrays of shape (J x 1) and (J x 2) that contain the strength of
        each force and its vectorial form.
    """

    # 1. Computes desired lengths for each edge.
    # First, we check for the desired density of the edge at each point in
    # the mesh (relative_edge_length).
    #middle_point_of_edge = np.mean(points[edges, :], axis=1)
    probability_edges = relative_edge_length(np.mean(points[edges, :], axis=1))
    # Determines scaling factor to ensure repulsive forces and density control.
    scaling_factor = np.sqrt(norm(edges_length)**2 / norm(probability_edges)**2)
    des_length = scaling_factor * force_scale * probability_edges

    # 2. Computes the forces through f(l, l0) = l0 - l, if l0 > l, where l0, l
    # are the desired length and current length of the edges, respectively.
    forces = np.maximum(des_length - edges_length,
                        np.zeros((len(edges_length), 1)))

    # Determines the x, y components by multiplying the force
    # by the normalized bar_vectors.
    forces_vector = np.multiply((forces / edges_length) * [1., 1.],
                                edges_vector)

    return forces, forces_vector


def calculate_vector_and_length_edges(points, edges):
    """Computes the vectors and length from a indexes of edges and
    respective points.

    Args:
        points: array of shape (N x 2) with coordinates of 2D points.
        edges: array of shape (K x 2) with indexes of the points
            that form edges.

    Returns:
        Array of shape (K x 2) with the coordinates of the vector that
        each edge forms and and array of shape (K x 1) with their
        respective lengths.

    """

    edges_vector = points[edges[:, 0], :] - points[edges[:, 1], :]
    edges_length = norm(edges_vector, axis=1).reshape(len(edges_vector), 1)

    return edges_vector, edges_length


def project_inside(points, level_set, derivative_step):
    """Projects the outside points back to the boundary.

    Here, we project the points that end outside the geometry after the last
    update back to the closest point on the boundary. This corresponds to a
    reaction force normal to the boundary.

    To do so, we compute the numerical gradient of level_set function on the
    outside points, which gives the (negative) direction to the closest
    boundary point.

    E.g.

                                 .
                                 |
                                 v
                              . _._ .
                             / \   / \          Points outside moved
                            /   \ /   \         to boundary.
                      . -->. ___ . ___ .
                            \   / \   /
                             \./___\./


    Args:
        points: Array of shape (N x 2) with all the points in the mesh.
        level_set: function handle for level-set of boundary
        derivative_step: Derivative step.

    Returns:
        Array of shape (N x 2) with the outside points moved to the closest
        boundary point. The points inside remain unchanged.
    """

    boundary_distance = level_set(points)  # Calculates distance to boundary
    # Selects points outside the domain
    index_outside_pts = boundary_distance > 0
    pts_outside = points[index_outside_pts]
    nmb_out = len(pts_outside)

    # Array with derivative step for derivative in x, y coordinates.
    derivative_step_array_x = np.hstack((derivative_step * \
        np.ones((nmb_out, 1)), np.zeros((nmb_out, 1))))

    derivative_step_array_y = np.hstack((np.zeros(
        (nmb_out, 1)), derivative_step * np.ones((nmb_out, 1))))

    # Computes gradients at outside points
    grad_x = (level_set(points[index_outside_pts] + derivative_step_array_x) \
             - boundary_distance[index_outside_pts]) / derivative_step

    grad_y = (level_set(points[index_outside_pts] + derivative_step_array_y) \
             - boundary_distance[index_outside_pts]) / derivative_step

    # Project points to the boundary (knowing the distance to closest point).
    points[index_outside_pts] -= np.vstack(
        (boundary_distance[index_outside_pts] * grad_x,
         boundary_distance[index_outside_pts] * grad_y)).T

    return points


def re_delaunay(points, level_set, geometry_tolerance):
    """Computes by Delaunay the triangles from a set of points.

    Here, we provide an helper function for the DistMesh routine that
    computes the elements and edges from a set of points.

    Args:
        points: Array of shape (N x 2) with points to Delaunay.
        level_set: function handle for level-set of boundary.
        geps: Tolerance to keep interior elements.

    Returns:
        mesh_triangles: Array of shape (K x 3) with indexes of points that
            define each element.
        edges: Array of shape (E x 2) with the edges that form elements.

    """

    # Creates (K x 3) Array of the triangular elements.
    mesh_triangles = scipy.spatial.Delaunay(points).simplices

    # Define (K x 3 x 2) Array of 2D coordinates for points in
    # each element
    mesh_triangles_pts = points[mesh_triangles, :]

    # Computes the centroid (middle point) of each element through
    # the mean of its ponts.
    centroids = np.mean(mesh_triangles_pts, axis=1)

    # Keeps interior elements
    mesh_triangles = mesh_triangles[level_set(centroids) < -geometry_tolerance]

    # Defines set of edges for each element, sort indexes and
    # eliminate repeated edges.
    edges = np.vstack((mesh_triangles[:, [0, 1]], mesh_triangles[:, [0, 2]],
                       mesh_triangles[:, [1, 2]]))
    edges = np.unique(np.sort(edges, 1), axis=0)

    return mesh_triangles, edges


def max_l2_norm(points):
    """ Returns maximum l2-norm of a set of points.

    Args:
        points: Array of shape (N x 2) of points.

    Returns:
        Maximum of l2-norms.
    """

    return np.max(norm(points, axis=1))

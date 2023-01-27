"""Computes the stiffness matrix of the Complete Electrode Model in 2D.

With a mesh defined, the electrodes fixed on the boundary and a
conductivity established inside, we can compute and assemble all
of the elements of the stiffness matrix for the Complete Electrode
Model.

For the developer convenience and for easier understanding of the code ahead
we remit to the introduction present in:

    /impedance/docs/eit_fem.md

We just remind that the stiffness matrix is composed of four blocks,

                    A = [[B1 + B2, C], [C.T, D]].

where B1 is the only one dependant of the conductivity values.

NOTE: In order to solve inverse problem or simulate voltages for many
conductivities it is required to solve the direct problem many times.
Thus, everything that remains constant between iterations can be pre-computed
once and saved for re-use. We structure our code with this in mind. The only
part that needs to be re-computed is the B1 matrix which depends on the
conductivity values. See [README](docs/eit_fem.md) for a deep dive!
"""

import numpy as np
import jax
import jax.numpy as jnp


# pylint: disable = W1401
def compute_global_triangles_integrals(eit_mesh):
    """Computes the global matrices for all triangles in mesh.

    This function computes for each triangle Tk = [i, j, l] in the
    mesh the integrals of B^1 with all index combinations. That is,

    Math:

        triangle_integrals_matrix[s, t] =
            \int_Tk \nabla \phi_s \cdot \nabla \phi_t dxdy

    with s, t = {i, j, l}. Here, each triangle_integrals_matrix is (3 x 3).
    This matrix is defined as $GM$ and was introduced in the
    [README](docs/eit_fem.md).


    Args:
        eit_mesh: Object of type CircleUniform2DMeshEit which defines an uniform
            triangular mesh on a circular domain. It contains attributes
            corresponding to points (eit_mesh.points - (nmb_points x 2) array)
            and triangles (eit_mesh.mesh_triangles - (nmb_triangles x 3) array),
            which define the mesh.

    Returns:
        global_triangle_integrals_matrix: Array of shape
        (eit_mesh.nmb_triangles, 3, 3) which contains for each mesh triangle its
        triangle_integrals_matrix, defined above, of shape (3 x 3).
    """
    # pylint: enable = W1401

    global_triangle_integrals_matrix = np.zeros((eit_mesh.nmb_triangles, 3, 3))

    # Iterates through the triangles of the mesh and computes for each the
    # respective global matrix.
    for k, triangle in enumerate(eit_mesh.mesh_triangles):
        # Sets the coordinates of each point of current triangle
        triangle_points = eit_mesh.points[triangle, :]

        triangle_integrals_matrix = \
            compute_triangle_integrals_matrix(triangle_points)
        global_triangle_integrals_matrix[k, :] = triangle_integrals_matrix

    return global_triangle_integrals_matrix


def compute_triangle_integrals_matrix(triangle_points):
    """Computes the triangle_integrals_matrix for a triangle.

    Here, we compute the triangle integrals matrix for a specific triangle.
    For a triangle $T_k = [i, j, l]$ this matrix consists of the following
    integrals:

    Math: Let s, t = {i, j, l} be one of the indexes defining the triangle.

        \int_Tk \nabla \phi_s \cdot \nabla \phi_t dxdy,

    (s and t can be equal). This establishes a (3 x 3) matrix with all
    the possible combinations of s and t.

    The computation of the integral is done by passing the global triangle
    to a local one by applying a change of variables and transforming the
    global basis functions into the local basis functions which are easier
    to treat.

    A proper math explanation is presented in

                /impedance/docs/eit_fem.md

    Args:
        triangle_points: Array of shape (3 x 2) with coordinates of
            the vertices of a certain global triangle.

    Returns:
        Array of shape (3 x 3) with the global matrix of the triangle.
    """

    # Initialize the matrix J defined as
    # J = [[x_j - x_i, x_l - x_i], [y_j - y_i, y_l - y_i]].
    # Recall that this matrix defines a 2D transformation by:
    # x = x_i + (x_j - x_i)\xi + (x_l - x_i)\zeta
    # y = y_i + (y_j - y_i)\xi + (y_l - y_i)\zeta  or
    # (x, y) = (x_i, y_i) + J(\xi, \zeta).
    # Only J is important for the integration.
    change_variables_matrix = np.zeros((2, 2))

    # Uses triangle_points = [[x_i, y_i], [x_j, y_j], [x_l, y_l]] and
    # multiplies by a proper matrix to obtain J.
    change_variables_matrix = np.matmul(triangle_points.T,
                                        np.array([[-1, -1], [1, 0], [0, 1]]))

    # Defines the gradient of local basis functions, the basis functions in the
    # local triangle.
    gradients_local_basis_functions = np.array([[-1, 1, 0], [-1, 0, 1]])

    # Computes: (J^T)^-1 \nabla local_basis_function_s, s in {00, 01, 10}
    gradients_global_basis_functions = np.matmul(
        np.linalg.inv(change_variables_matrix).T,
        gradients_local_basis_functions)

    # Computes the inner products of all combinations of column vectors.
    gradients_inner_products_matrix = (gradients_global_basis_functions.T
                                      ).dot(gradients_global_basis_functions)

    global_triangle_area = 0.5 * np.abs(np.linalg.det(change_variables_matrix))

    return global_triangle_area * gradients_inner_products_matrix


def assemble_b1_matrix(eit_model, conductivity_array):
    """Computes the B1 matrix which represents the terms inside the domain.

    The B1 matrix is defined through the integrals over the domain included
    in the global_triangle_integrals_matrix with a proper multiplication with
    the conductivity value.

    B1[i, j] is the sum over all triangles k for which ij is an edge of

    Math:

            conductivity[k] * \int_Tk \nabla \phi_i \cdot \nabla \phi_j dxdy.

    To add all the terms, we iterate over the triangles and use the
    global_triangle_integrals_matrix to add all of the above integrals over a
    current triangle to their right position in B1.

    A proper math explanation is presented in

                    /impedance/fem_jax/README.md

    Args:
        eit_model: Object of class type EitModel that inherits an object
            of type CircleUniform2DMeshEit designated by eit_mesh and contains
            the global_triangle_integrals_matrix of shape
            (eit_model.eit_mesh.nmb_triangles, 3, 3) which contains the
            integrals on each triangle to be added to B1.
        conductivity: Array of shape (eit_model.eit_mesh.nmb_triangles, 1)
            with the conductivity values on each mesh triangle.

    Returns:
        Square matrix of size (eit_model.eit_mesh.nmb_points +
        eit_model.eit_mesh.nmb_electrodes - 1), where the block
        (eit_model.eit_mesh.nmb_points, eit_model.eit_mesh.nmb_points) contains
        the B1 matrix.
    """

    # Initializes matrix b1 with the size of the stiffness matrix.
    # Here, we only update the block
    # [0: eit_model.eit_mesh.nmb_points, 0: eit_model.eit_mesh.nmb_points],
    # but this is usefull to add to the other terms of the stiffness matrix.
    matrix_b1 = jnp.zeros(
        (eit_model.eit_mesh.nmb_points + eit_model.eit_mesh.nmb_electrodes - 1,
         eit_model.eit_mesh.nmb_points + eit_model.eit_mesh.nmb_electrodes - 1))

    pre_computation = jnp.einsum("i, ikl->ikl", conductivity_array,
                                 eit_model.global_triangle_integrals_matrix)

    mesh_triangles = jnp.array(eit_model.eit_mesh.mesh_triangles)

    def body_fun(index, matrix_b1):
        """Adds to all relevant index combinations the respective integral
           computations
        """

        triangle = mesh_triangles[index, :]

        # Creates indexing matrix for the triangle. For a triangle with indexes
        # triangle = [i, j, l] this leads to
        # mesh_x = [i, i, i, j, j, j, l, l, l]
        # and mesh_y = [i, j, l, i, j, l, i, j, l].
        mesh_x, mesh_y = jnp.meshgrid(triangle, triangle, indexing="ij")

        # Add respective pre-computation
        matrix_b1 = matrix_b1.at[mesh_x, mesh_y].add(pre_computation[index, :])

        return matrix_b1

    # Jaxian loop in order to jit in one go and not desenroll it,
    # like a normal jit.
    matrix_b1 = jax.lax.fori_loop(0, eit_model.eit_mesh.nmb_triangles, body_fun,
                                  matrix_b1)

    return matrix_b1


def assemble_b2_and_c_matrix(eit_mesh, contact_impedances):
    """Computes the B2 and C matrix, simultaneously.

    Here, we assemble the matrices B2 and C by computing the corresponding
    integrals. The computation is simultaneous since the integrals are
    computed over the same edges with different integrands. This brings
    similarities in the computation that we can exploit.

    The B2 matrix is obtained by computation of integrals
    over electrodes of two basis functions multiplied. To be specific:

    Math:

        \int_{E_l} \phi_i \phi_j ds.

    Accordingly, the matrix B^2 will be very sparse since most basis functions
    are zero at the boundary and in particular at the electrodes. We take
    advantage of this by just looking at the basis functions with vertice on
    the electrodes.

    The entries of C (see README) can be defined in two split ways according to
    the position of the index i. If i is inside the electrode E_0, then

    Math:

        C_ij = -(1 / contact_impedance[0]) \int_{E_0} \phi_i ds,

        for all j in 0, ..., L - 2.

    Otherwise, if the index i belongs to another electrode E_{n+1} it is

    Math:

        C_in = (1 / contact_impedance[n+1]) \int_{E_{n+1}} \phi_i dx,

        and 0 in the rest of the line.

    Args:
        eit_mesh: Object of type CircleUniform2DMeshEit which defines a uniform
            triangular mesh on a circular domain. It contains attributes
            corresponding to points (eit_mesh.points - (nmb_points x 2) array)
            and triangles (eit_mesh.mesh_triangles - (nmb_triangles x 3) array),
            which define the mesh.
        contact_impedances: Array of shape (nmb_electrodes x 1) with the contact
            impedance value on each electrode. The array only contains positive
            values.

    Returns:
        Matrix B2 and C with shape (eit_mesh.nmb_points, eit_mesh.nmb_points)
        and  (eit_mesh.nmb_points, nmb_electrodes - 1), respectively.
    """

    assert contact_impedances.all() > 0

    matrix_b2 = np.zeros((eit_mesh.nmb_points, eit_mesh.nmb_points))
    matrix_c = np.zeros((eit_mesh.nmb_points, eit_mesh.nmb_electrodes - 1))

    # Iterates of the edges of E_0, and updates the lines of C, for which
    # the indexes that define the edges are in E_0.
    for edge in eit_mesh.electrode_edges[0]:

        # Computes distance between the two points of the edge.
        edge_vector = eit_mesh.points[edge[1], :] - \
                eit_mesh.points[edge[0], :]
        edge_length = np.linalg.norm(edge_vector)

        # Updates the matrix_c
        matrix_c[edge[0], :] += -0.5 * (1 / contact_impedances[0]) * edge_length
        matrix_c[edge[1], :] += -0.5 * (1 / contact_impedances[0]) * edge_length

        # Update the matrix_b2
        # Adds up the integration of different basis functions multiplied.
        matrix_b2[edge[0], edge[1]] += (1 / (6 * contact_impedances[0])) * \
            edge_length
        matrix_b2[edge[1], edge[0]] += (1 / (6 * contact_impedances[0])) * \
            edge_length

        # Adds up the integration of the basis function squared.
        matrix_b2[edge[0], edge[0]] += (1 / (3 * contact_impedances[0])) * \
            edge_length
        matrix_b2[edge[1], edge[1]] += (1 / (3 * contact_impedances[0])) * \
            edge_length

    # Iterates over the rest of electrodes and corresponding edges, to update
    # the particular entries of C.
    for j in range(1, eit_mesh.nmb_electrodes):
        for edge in eit_mesh.electrode_edges[j]:

            # Computes distance between the two points of the edge.
            edge_vector = eit_mesh.points[edge[1], :] - \
                eit_mesh.points[edge[0], :]
            edge_length = np.linalg.norm(edge_vector)

            # Update the matrix_c
            matrix_c[edge[0], j - 1] += 0.5 * (1 / contact_impedances[j]) * \
                edge_length
            matrix_c[edge[1], j - 1] += 0.5 * (1 / contact_impedances[j]) * \
                edge_length

            # Updates the matrix_b2
            # Adds up the integration of different basis functions multiplied.
            matrix_b2[edge[0], edge[1]] += (1 / (6 * contact_impedances[j])) * \
                edge_length
            matrix_b2[edge[1], edge[0]] += (1 / (6 * contact_impedances[j])) * \
                edge_length

            # Adds up the integration of the basis function squared.
            matrix_b2[edge[0], edge[0]] += (1 / (3 * contact_impedances[j])) * \
                edge_length
            matrix_b2[edge[1], edge[1]] += (1 / (3 * contact_impedances[j])) * \
                edge_length

    return matrix_b2, matrix_c


def assemble_d_matrix(eit_mesh, contact_impedances):
    """Computes the D matrix.

    Here, we assemble the D matrix that is given as:

    Math:

    D_ij = |E_0| / contact_impedance_E_0,  for i different j

    D_ij = |E_0| / contact_impedance_E_0 + |E_{i}| / contact_impedance_E_{i}
        for i = j.

    Thus we can start defining the matrix D by |E_0| / contact_impedance_E_0 in
    every entry and just add up the corresponding terms on the diagonal.

    Here |E_i| is the electrode length, but we assume all electrodes are equal.

    Args:
        eit_mesh: Object of type CircleUniform2DMeshEit which defines a uniform
            triangular mesh on a circular domain. It contains attributes
            corresponding to points (eit_mesh.points - (nmb_points x 2) array)
            and triangles (eit_mesh.mesh_triangles - (nmb_triangles x 3) array),
            which define the mesh.
        contact_impedances: Array of shape (nmb_electrodes x 1) with the contact
            impedance value on each electrode.

    Returns:
        Matrix D with shape (nmb_electrodes - 1, nmb_electrodes - 1).
    """

    # Initializes definition of D.
    general_term_matrix_d = (
        eit_mesh.electrode_arc_length / contact_impedances[0]) * np.ones(
            (eit_mesh.nmb_electrodes - 1, eit_mesh.nmb_electrodes - 1))

    # Defines the extra terms at the diagonal corresponding to each electrode.
    extra_diagonal_term_matrix_d = np.diag(eit_mesh.electrode_arc_length /
                                           contact_impedances[1:])

    # Returns their addition that defines the matrix_d
    return general_term_matrix_d + extra_diagonal_term_matrix_d


def assemble_constant_term_stiffness_matrix(eit_mesh, contact_impedances):
    """Assembles the constant term of the stiffness matrix.

    Here, we assemble the constant term of the stiffness matrix given as:

                            [B^2 C; C^T D]

    Args:
        eit_mesh: Object of type circle_mesh_eit, with attributes corresponding
            to points (eit_mesh.points - (nmb_points x 2) array) and triangles
            (eit_mesh.mesh_triangles - (nmb_triangles x 3) array).
        contact_impedances: Array of shape (nmb_electrodes x 1) with the
            contact impedance value on each electrode.

    Returns:
        Array that defines the constant terms of the stiffness matrix with shape
        (nmb_points + nmb_electrodes - 1, nmb_points + nmb_electrodes - 1).
    """

    constant_matrix = np.zeros(
        (eit_mesh.nmb_points + eit_mesh.nmb_electrodes - 1,
         eit_mesh.nmb_points + eit_mesh.nmb_electrodes - 1))

    # Assemble B2, C and D separately.
    matrix_b2, matrix_c = assemble_b2_and_c_matrix(eit_mesh, contact_impedances)
    matrix_d = assemble_d_matrix(eit_mesh, contact_impedances)

    # Add all terms together into a matrix with the same shape as the
    # stiffness matrix.
    constant_matrix[0:eit_mesh.nmb_points, 0:eit_mesh.nmb_points] = matrix_b2

    constant_matrix[0:eit_mesh.nmb_points, eit_mesh.nmb_points:] = matrix_c

    constant_matrix[eit_mesh.nmb_points:, 0:eit_mesh.nmb_points] = matrix_c.T

    constant_matrix[eit_mesh.nmb_points:, eit_mesh.nmb_points:] = matrix_d

    return constant_matrix


def define_voltage_basis_functions(nmb_electrodes):
    """Defines the basis functions over the electrodes.

    The voltage basis functions are basis functions over the electrodes used
    for the CEM. Part of the FEM solution will be a vector of coefficients that
    multiplied by each respective basis function the voltages simulation over
    the electrodes.

    The basis functions are fixed with respect to the number of electrodes
    and can be pre-computed at the start.

    Math Description:
    All vector have length equal to the number of electrodes.

            \eta_1 = [1, -1, 0, ..., 0]^T
            \eta_2 = [1, 0, -1, ..., 0]^T
            ...
            \eta_{L-1} = [1, 0, ..., 0, -1]^T

    Args:
        nmb_electrodes: number of electrodes we are using.

    Returns:
        Matrix of shape (nmb_electrodes, nmb_electrodes - 1) where each column
        represents a basis function at the electrodes.
    """

    voltage_basis_matrix = np.zeros((nmb_electrodes, nmb_electrodes - 1))

    voltage_basis_matrix[0, :] = 1
    voltage_basis_matrix[1:, :] = np.diag(-np.ones(nmb_electrodes - 1))

    return voltage_basis_matrix

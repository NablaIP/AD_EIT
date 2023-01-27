"""Direct solver for the FEM system of equation of EIT in 2D.

Here, we piece together all of the pieces needed to build the stiffness
matrix for a certain conductivity and solve the algebraic system of equations
that was obtained from the FEM for the Complete Electrode Model of EIT in 2D.

We establish a class that defines the EitModel being used, that is, mesh,
current patterns and stiffness matrix.

Use Conjugate Gradients to solve the FEM system

                A theta = hat{I},

where A is the stiffness matrix and hat{I} is defined through the choice of
current pattern. theta is the solution coefficients that are used to obtain
the voltages at the electrodes as

                U = H theta,

where H is the voltage basis matrix, that contains the basis functions at the
electrodes defining a basis of the spaces of possible voltages.
"""
import math
from functools import partial

import numpy as np
import jax
from jax.scipy.sparse import linalg
import jax.numpy as jnp

import conductivity
import stiffness_matrix


class EitModel:
    """Defines all variables that can be pre-computed and can be re-used on
    solving the inverse problem.

    This class contains all pre-computations that are essential for the FEM to
    run without an heavy burden, and that either depend on the mesh/domain and
    on the choice of currents.

    Recall that, we can pre-compute:
        - the constant term of the stiffness matrix A^{constant};
        - the integrals taking part of the block B^1, that is used to define
        the stiffness matrix A;
        - the basis function at the electrodes, which are required to obtain
        the voltages in the end;
        - the right-hand side of the algebraic system of equations obtained by
        FEM, that only depends on the current patterns applied at the electrodes
        and which are defined when starting a measuring procedure.

    We are using an independent mesh that has a certain number of points and
    triangles, (nmb_points and nmb_triangles, respectively) with a set of
    electrodes on attached to the boundary (nmb_electrodes).

    Attributes:
        global_triangles_integrals_matrix: Array of shape
            (nmb_triangles, 3, 3) which contains for each mesh triangle
            its triangle_integrals_matrix, defined above, of shape (3 x 3).
        constant_term_stiffness_matrix: Array that defines the constant terms of
            the stiffness matrix with shape (nmb_points + nmb_electrodes - 1,
            nmb_points + nmb_electrodes - 1).
        voltage_basis_matrix: Matrix of shape (nmb_electrodes,
            nmb_electrodes - 1) where each column represents a basis function at
            the electrodes.
        ext_voltage_basis_matrix: Matrix of shape (nmb_electrodes, nmb_points +
            nmb_electrodes - 1) which is an extension of voltage_basis_matrix
            where the first nmb_points collumns are 0-vectors.
    	fem_eq_right_side: Array of shape
            (nmb_points + nmb_electrodes -1, nmb_electrodes - 1) where each
            j-th column represents the right-hand side of the FEM system
            with respect to j-th current pattern applied. The first nmb_points
            are zero.
    """

    def __init__(self, eit_mesh, current_amplitude, contact_impedances,
                 current_pattern):
        self.eit_mesh = eit_mesh

        self.global_triangle_integrals_matrix = \
            stiffness_matrix.compute_global_triangles_integrals(eit_mesh)

        self.constant_term_stiffness_matrix = \
            stiffness_matrix.assemble_constant_term_stiffness_matrix(
                eit_mesh, contact_impedances)

        self.voltage_basis_matrix = \
            stiffness_matrix.define_voltage_basis_functions(
                eit_mesh.nmb_electrodes)

        # Creates Extended Matrix M = [0 voltage_basis_matrix]
        self.ext_voltage_basis_matrix = np.zeros(
            (self.eit_mesh.nmb_electrodes,
             self.eit_mesh.nmb_points + self.eit_mesh.nmb_electrodes - 1))
        self.ext_voltage_basis_matrix[:, self.eit_mesh.nmb_points:] = \
                    self.voltage_basis_matrix

        self.fem_eq_right_side = \
            compute_fem_eq_right_side(
                eit_mesh.nmb_points, eit_mesh.nmb_electrodes,
                eit_mesh.electrode_arc_length, current_pattern,
                current_amplitude)

    @partial(jax.jit, static_argnums=(0,))
    def solver(self, anomaly):
        """Solves the direct problem with FEM for the CEM.

        This function solves the algebraic system of equations:

                A theta = hat{I},

        where A is the stiffness matrix for a given conductivity, hat{I}
        corresponds to the current pattern applied and theta is the solution.
        Recall that, A is given as

                    A = [B1 + B2   C; C.T   D]

        where

                    A^constant = [B2  C; C.T  D]

        is constant for a certain mesh and does not depend on the
        conductivity. Therefore, we use the pre-compute A^constant
        and update A with the conductivity by changing A.

        Args:
            self: Object of type EitModel that contains the mesh of the
                domain and all pre-computations of the FEM model.
            conductivity: Array of shape (self.eit_mesh.nmb_triangles, 1)
                with the conductivity values on each mesh triangle.
        Return:
            voltages_matrix: Array of shape (self.eit_mesh.nmb_electrodes,
                self.eit_mesh.nmb_electrodes - 1) that contains on each
                column the voltage measurements corresponding to a current
                pattern applied.
        """

        # Generate from anomaly a conductivity array
        conductivity_array = conductivity.create_conductivity_array(
            anomaly, self.eit_mesh)

        # Compute the B1 matrix, that depends on the conductivity values.
        matrix_b1 = stiffness_matrix.assemble_b1_matrix(self,
                                                        conductivity_array)

        # Update the stiffness matrix (with the pre-compute constant term).
        matrix_a = jnp.add(self.constant_term_stiffness_matrix, matrix_b1)

        vectorized_cg = jax.vmap(linalg.cg, in_axes=(None, 1), out_axes=1)

        coefficient_solution, _ = vectorized_cg(matrix_a,
                                                self.fem_eq_right_side)

        # Recall that, the voltages are expanded in terms of the basis functions
        # like U = \sum \eta_k \beta_k, where \eta_k is a basis function and
        # \beta=[\beta_1, ...,\beta_{L-1}] are coefficients that are part of the
        # solution to the FEM system.
        voltage_coefficients = coefficient_solution[self.eit_mesh.nmb_points:]
        voltages_matrix = self.voltage_basis_matrix @ voltage_coefficients

        return coefficient_solution, voltages_matrix.T.reshape(-1)

    def solver_ad(self, anomaly):
        """Solves the direct problem with FEM for the CEM.

        This function solves the algebraic system of equations:

                A theta = hat{I},

        where A is the stiffness matrix for a given conductivity, hat{I}
        corresponds to the current pattern applied and theta is the solution.
        Recall that, A is given as

                    A = [B1 + B2   C; C.T   D]

        where

                    A^constant = [B2  C; C.T  D]

        is constant for a certain mesh and does not depend on the
        conductivity. Therefore, we use the pre-compute A^constant
        and update A with the conductivity by changing A.

        Args:
            self: Object of type EitModel that contains the mesh of the
                domain and all pre-computations of the FEM model.
            conductivity: Array of shape (self.eit_mesh.nmb_triangles, 1)
                with the conductivity values on each mesh triangle.
        Return:
            voltages_matrix: Array of shape (self.eit_mesh.nmb_electrodes,
                self.eit_mesh.nmb_electrodes - 1) that contains on each
                column the voltage measurements corresponding to a current
                pattern applied.
        """

        # Generate from anomaly a conductivity array
        conductivity_array = conductivity.create_conductivity_array(
            anomaly, self.eit_mesh)

        # Compute the B1 matrix, that depends on the conductivity values.
        matrix_b1 = stiffness_matrix.assemble_b1_matrix(self,
                                                        conductivity_array)

        # Update the stiffness matrix (with the pre-compute constant term).
        matrix_a = jnp.add(self.constant_term_stiffness_matrix, matrix_b1)

        vectorized_cg = jax.vmap(linalg.cg, in_axes=(None, 1), out_axes=1)

        coefficient_solution, _ = vectorized_cg(matrix_a,
                                                self.fem_eq_right_side)

        # Recall that, the voltages are expanded in terms of the basis functions
        # like U = \sum \eta_k \beta_k, where \eta_k is a basis function and
        # \beta=[\beta_1, ...,\beta_{L-1}] are coefficients that are part of the
        # solution to the FEM system.
        voltage_coefficients = coefficient_solution[self.eit_mesh.nmb_points:]
        voltages_matrix = self.voltage_basis_matrix @ voltage_coefficients

        return voltages_matrix.T.reshape(-1)

    @partial(jax.jit, static_argnums=(0, ))
    def direct_operator(self, anomaly):
        """Computes voltage measurements from an anomaly.

        This is specifically a simplification of the solver to be used with AD.

        Args:
            self: Object of type EitModel that contains the mesh of the
                domain and all pre-computations of the FEM model.
            anomaly: Array of shape (5,) with parameterization of circular
                conductivity anomaly.

        Returns:
            voltages: Array of shape
            (self.eit_mesh.nmb_electrodes*(self.eit_mesh.nmb_electrodes -1),)
            that contains voltage measurements with respect to different
            current patterns concatenated into a 1D array.
        """

        voltages = self.solver_ad(anomaly)

        return voltages

    def direct_operator_(self, anomaly):
        """Computes voltage measurements from an anomaly.

        This is specifically a simplification of the solver to be used with AD.

        Args:
            self: Object of type EitModel that contains the mesh of the
                domain and all pre-computations of the FEM model.
            anomaly: Array of shape (5,) with parameterization of circular
                conductivity anomaly.

        Returns:
            voltages: Array of shape
            (self.eit_mesh.nmb_electrodes*(self.eit_mesh.nmb_electrodes -1),)
            that contains voltage measurements with respect to different
            current patterns concatenated into a 1D array.
        """

        voltages = self.solver_ad(anomaly)

        return voltages

    @partial(jax.jit, static_argnums=(0, ))
    def jacobian_ad(self, anomaly):
        return jax.jacfwd(self.direct_operator_)(anomaly)

    def jacobian_analytic_non_opt(self, fem_solution, anomaly):
        """Compute jacobian of direct operator with analytical formulation.

        The direct operator is given as:

                Sim: anomaly ---> V = voltage_basis_matrix * beta,

        where beta is part of solution theta = [alpha, beta] of the FEM,

                A * theta = \tilde{I}.

        As such, the derivatives for each parameter w of the anomaly given as:

        partial V / partial w = - gamma.T * partial A / partial w * theta,

        where gamma is the solution of the adjoint system:

                A * gamma = [0 M].T

        Args:
            self: Object of type EitModel that contains the mesh of the
                domain and all pre-computations of the FEM model.
            coefficient_solution: Array of shape (self.eit_mesh.nmb_points
                + self.eit_mesh.nmb_electrodes - 1,
                self.eit_mesh.nmb_electrodes - 1) with solution of FEM.
            anomaly: Array of shape (5,) with parameterization of circular
                conductivity anomaly.
        Returns:
            jacobian: Array of shape
                (self.eit_mesh.nmb_electrodes *
                (self.eit_mesh.nmb_electrodes - 1) , 5) defines the
                jacobian matrix of direct operator through analytic
                formulation for FEM - EIT.
        """

        # Computes conductivity derivative at elements wrt anomaly parameters
        conductivity_derivative = conductivity.compute_derivative_conductivity(
            self.eit_mesh, anomaly)

        # Computes matrix B1 jacobian through chain-rule
        matrix_b1_jacobian = derivative_matrix_b1(self, conductivity_derivative)

        # Solve adjoint system
        adjoint_solution = adjoint_system_solver(self, anomaly)

        jacobian = jnp.zeros(
            (self.eit_mesh.nmb_electrodes * (self.eit_mesh.nmb_electrodes - 1),
             5))

        nmb_electrodes = self.eit_mesh.nmb_electrodes

        # Join everything together
        for k in range(nmb_electrodes - 1):
            theta = fem_solution[:self.eit_mesh.nmb_points, k]
            for index in range(5):
                jacobian = jacobian.at[
                    k * nmb_electrodes:(k + 1) * nmb_electrodes,
                    index].set(-jnp.matmul(
                        adjoint_solution[:, :self.eit_mesh.nmb_points],
                        (jnp.matmul(matrix_b1_jacobian[:, :, index], theta))))

        return jacobian


    @partial(jax.jit, static_argnums=(0, ))
    def jacobian_analytic(self, fem_solution, anomaly):
        """Compute jacobian of direct operator with analytical formulation.

        The direct operator is given as:

                Sim: anomaly ---> V = voltage_basis_matrix * beta,

        where beta is part of solution theta = [alpha, beta] of the FEM,

                A * theta = \tilde{I}.

        As such, the derivatives for each parameter w of the anomaly given as:

        partial V / partial w = - gamma.T * partial A / partial w * theta,

        where gamma is the solution of the adjoint system:

                A * gamma = [0 M].T

        Args:
            self: Object of type EitModel that contains the mesh of the
                domain and all pre-computations of the FEM model.
            coefficient_solution: Array of shape (self.eit_mesh.nmb_points
                + self.eit_mesh.nmb_electrodes - 1,
                self.eit_mesh.nmb_electrodes - 1) with solution of FEM.
            anomaly: Array of shape (5,) with parameterization of circular
                conductivity anomaly.
        Returns:
            jacobian: Array of shape
                (self.eit_mesh.nmb_electrodes *
                (self.eit_mesh.nmb_electrodes - 1) , 5) defines the
                jacobian matrix of direct operator through analytic
                formulation for FEM - EIT.
        """

        # Computes conductivity derivative at elements wrt anomaly parameters
        conductivity_derivative = conductivity.compute_derivative_conductivity(
            self.eit_mesh, anomaly)

        # Computes matrix B1 jacobian through chain-rule
        matrix_b1_jacobian = derivative_matrix_b1(self, conductivity_derivative)

        # Solve adjoint system
        adjoint_solution = adjoint_system_solver(self, anomaly)
        adjoint_solution = adjoint_solution[:, :self.eit_mesh.nmb_points]

        jacobian = jnp.zeros(
            (self.eit_mesh.nmb_electrodes * (self.eit_mesh.nmb_electrodes - 1),
             5))

        nmb_electrodes = self.eit_mesh.nmb_electrodes
        alpha = fem_solution[:self.eit_mesh.nmb_points, :]
        slice_ = jnp.zeros((nmb_electrodes, nmb_electrodes - 1), dtype=int)
        for k in range(nmb_electrodes-1):
            slice_ = slice_.at[:, k].set(
                jnp.arange(k * nmb_electrodes, (k+1)*nmb_electrodes))

        def body_fun(index, jacobian):
            temp_array = jnp.einsum("ijk, j->ik",
                matrix_b1_jacobian,
                alpha[:, index])

            jacobian_voltage_index = jnp.einsum(
                "ni, ik->nk",
                adjoint_solution,
                temp_array)

            jacobian = jacobian.at[slice_[:, index], :].add(
                -jacobian_voltage_index)

            return jacobian

        jacobian = jnp.zeros((nmb_electrodes*(nmb_electrodes - 1), 5))
        jacobian = jax.lax.fori_loop(0, nmb_electrodes - 1, body_fun,
                                  jacobian)

        return jacobian


def adjoint_system_solver(eit_model, anomaly):
    """ Solves adjoint system with stiffness matrix A and different right side.

    Args:
        eit_model: Object of type EitModel that contains the mesh of the
                    domain and all pre-computations of the FEM model.
        anomaly: Array of shape (5,) with parameterization of circular
            conductivity anomaly.
    Returns:
        coefficient_solution: Array of shape
            (eit_model.eit_mesh.nmb_electrodes,
            eit_model.eit_mesh.nmb_points + eit_model.eit_mesh.nmb_electrodes
            - 1) with the adjoint system solution.
    """

    # Re-compute stiffness matrix -- ATENTION CAN BE BYPASSED THROUGH MEMORY
    conductivity_array = conductivity.create_conductivity_array(
        anomaly, eit_model.eit_mesh)
    matrix_b1 = stiffness_matrix.assemble_b1_matrix(eit_model,
                                                    conductivity_array)
    matrix_a = jnp.add(eit_model.constant_term_stiffness_matrix, matrix_b1)

    # Solving Adjoint System
    vectorized_cg = jax.vmap(linalg.cg, in_axes=(None, 1), out_axes=1)
    adjoint_solution, _ = vectorized_cg(matrix_a.T,
                                        eit_model.ext_voltage_basis_matrix.T)

    return adjoint_solution.T


def derivative_matrix_b1(eit_model, conductivity_derivative):
    """ Computes the derivative with respect to anomaly of matrix B1.

    The derivative of the matrix B1 with respect to the anomaly parameters
    can be computed through chain-rule of differentiation like:

    partial B^1 / partial w = sum_{k=0}^K
        partial B^1 / partial sigma_k) * (partial sigma_k / partial w),

    where K is the number of mesh elements and the derivative of B1 with
    respect to each value sigma_k is given by the integrals of the function
    over that element.

    Args:
        eit_model: Object of type EitModel that contains the mesh of the
            domain and all pre-computations of the FEM model.
        conductivity_derivative: Derivative of conductivity_array with
            respect to anomaly parameters.

    Returns:
        matrix_b1_jacobian: Array of shape (eit_model.eit_mesh.nmb_points,
            eit_model.eit_mesh.nmb_points, 5) that contains the derivative
            values of matrix b1 evaluated at the current anomaly.
    """

    matrix_b1_jacobian = jnp.zeros(
        (eit_model.eit_mesh.nmb_points, eit_model.eit_mesh.nmb_points, 5))

    for index_k, triangle in enumerate(eit_model.eit_mesh.mesh_triangles):

        mesh_x, mesh_y = jnp.meshgrid(triangle, triangle, indexing="ij")

        anomaly_element_derivative = jnp.einsum(
            "ij, k -> ijk", eit_model.global_triangle_integrals_matrix[index_k],
            conductivity_derivative[index_k, :])

        # Should output a mesh_x, mesh_y, 5 to compute \partial B1/\partial_w
        matrix_b1_jacobian = matrix_b1_jacobian.at[mesh_x, mesh_y, :].add(
            anomaly_element_derivative)

    return matrix_b1_jacobian


def adjacent_current_pattern_matrix(nmb_electrodes, current_amplitude):
    """Defines the adjacent pattern matrix.

    Given (nmb_elctrodes) electrodes we define (nmb_electrodes - 1)
    current patterns of adjacent type, which are linearly independent
    satisfy Kirchoff's law.

    The adjacent current pattern is characterized by inserting positive
    current in one electrode and negative on the one on its right, e.g.,

                I = [0,..., 1, -1, 0, ...,0].

    The general set of current patterns required for the inverse problem,
    starts inserting current on the first electrode, and each new current
    pattern moves on electrode to the right. This will form a set of
    (nmb_electrodes - 1) linearly independent current patterns.

    Args:
        nmb_electrodes: number of electrodes.
        current_amplitude: current amplitude in mA.

    Returns:
        Array of shape (nmb_electrodes, nmb_electrodes - 1) where each
        column represents an adjacent current pattern.
    """

    current_pattern_matrix = np.zeros((nmb_electrodes, nmb_electrodes - 1))

    for k in range(nmb_electrodes - 1):
        current_pattern_matrix[k:k + 2, k] = np.array(
            [current_amplitude, -current_amplitude])

    return current_pattern_matrix


def trignometric_current_pattern_matrix(nmb_electrodes, electrode_arc_length,
                                        current_amplitude):
    """Defines the trignometric current pattern.

    Given (nmb_electrodes) electrodes we define (nmb_electrodes - 1)
    current patterns of trignometric type, which are linearly independent
    and that satisfy Kirchoff's law.

    In order to simulate the voltages, we need to apply a electrical
    current pattern to the set of electrodes around the object. This
    electrical current pattern needs not to have any special form.

    However, when we want to either solve the inverse problem or match
    the simulated voltages with the measured ones, we need more than one
    current pattern. In this sense, the trignometric patterns are the most
    important (but not the most practical in real-life) and are defined as
    follows:

    Math:
    Let j be the index of the current pattern and i the index of the electrode.
    Then I_j^i is the current applied to the electrode i-th from the current
    pattern j-th defined by:

        if l <= nmb_electrodes / 2:

            I_j^i = A * cos(j * angle_center_of_electrode_i )

        else:

            I_j^i = A * sin((j - nmb_electrodes / 2) * \
                             angle_center_of_electrode_i )

    where A is the amplitude of the current being applied.

    This forms a set of (nmb_electrodes - 1) linearly independent current
    patterns.


    Args:
        nmb_electrodes: number of electrodes
        electrode_arc_length: angular length of each electrode
        current_amplitude: current amplitude in mA

    Returns:
        Array of shape (nmb_electrodes, nmb_electrodes - 1) where each
        column represents a trignometric current pattern.
    """

    current_pattern_matrix = np.zeros((nmb_electrodes, nmb_electrodes - 1))

    electrode_center_angles = (2 * math.pi / nmb_electrodes) * \
        np.arange(nmb_electrodes) + electrode_arc_length / 2

    for j in range(nmb_electrodes - 1):
        if j + 1 <= nmb_electrodes / 2:

            current_pattern_matrix[:, j] = current_amplitude *  \
                np.cos((j + 1) * electrode_center_angles)
        else:
            current_pattern_matrix[:, j] = current_amplitude * \
                np.sin((j + 1 - nmb_electrodes / 2) * electrode_center_angles)

    return current_pattern_matrix


def compute_fem_eq_right_side(mesh_nmb_points, nmb_electrodes,
                              electrode_arc_length, current_pattern,
                              current_amplitude):
    """Defines right-hand sides of the FEM system.

    The right-hand side is defined for a single current pattern,
    [I_1, ..., I_{nmb_electrodes}], as

        hat{I} = [vector_of_zeros, I_1 - I_2, ..., I_1 - I_{nmb_electrodes}],

    where vector_of_zeros has dimension (mesh_nmb_points).

    Here, we compute simultaneously all right-hand sides for the respective
    current patterns applied.

    Args:
        mesh_nmb_points: number of points in the mesh
        nmb_electrodes: number of electrodes
        current_pattern: string defining which set of current patterns to use.
            The possibilities are {trignometric, adjacent}. Trignometric applies
            current throughout all electrodes, adjacent only does it over two
            electrodes.
    Returns:
        Array of shape (mesh_nmb_points + nmb_electrodes -1, nmb_electrodes - 1)
        where each j-th collumn represents the right-hand side of the FEM system
        when the current pattern j-th is applied. The first (mesh_nmb_points)
        are zero.
    """

    # Choose the set of current patterns to apply!
    if current_pattern == "trignometric":
        current_pattern_matrix = trignometric_current_pattern_matrix(
            nmb_electrodes, electrode_arc_length, current_amplitude)
    elif current_pattern == "adjacent":
        current_pattern_matrix = adjacent_current_pattern_matrix(
            nmb_electrodes, current_amplitude)

    constant_terms_fem_system = np.zeros(
        (mesh_nmb_points + nmb_electrodes - 1, nmb_electrodes - 1))

    # Defines the right-hand side of the system of equations in FEM
    # corresponding to each current pattern applied.
    for j in range(nmb_electrodes - 1):
        constant_terms_fem_system[j + mesh_nmb_points, :] = \
            current_pattern_matrix[0, :] - current_pattern_matrix[j + 1, :]

    return constant_terms_fem_system

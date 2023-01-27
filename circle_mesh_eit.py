"""Circle mesh for the Electrical Impedance Tomography in 2D.

Here, we devise an unstructured triangular mesh with the DistMesh algorithm
on a circular domain. This domain is centered at the origin and is defined
only by its radius. Further, in this class we define a set of attributes
related with the set of electrodes that will be attached to the boundary and
are essential to EIT.

This class is a layout for the computation of the stiffness matrix in EIT
direct problem.

Assumptions:
Our domain is a circle centered at the origin with a certain radius.
Further, our model of EIT uses equidistant electrodes with a fixed
arc length. This assumption arises from the real-world application where
the electrodes injecting current and measure voltage over a small arc of the
surface.

For implementation purposes, each electrode is represented by
three points/angles:

Electrode = {angle_; angle_ + electrode_arc_length / 2;
             angle_ + electrode_arc_length},

where angle_ is the angle of the left-hand side extremity (anti-clockwise order)
of the electrode. These angles are also equidistant. Notice that all
points/angles in the boundary which end up between the left and right extremity
are also contained in the electrode.

See: /impedance/figures/electrodes_mesh_illustrated.png
"""

import os
import math

import numpy as np
import matplotlib.pyplot as plt

from dist_mesh import dist_mesh_2d
from dist_mesh import MeshControlParameters


class CircleUniform2DMeshEit:
    """Defines a uniform triangular mesh on a circular domain through DistMesh
    algorithm and with equidistant electrodes.

    This class allows the user to define the arc length and number of electrodes
    in order to generate a mesh with equidistant electrodes and a desired
    resolution of a circular domain. Each electrode is defined by its middle
    point and its two extremes. These set of points serves as input to the
    DistMesh algorithm by the fixed points attribute.

    This class also contains the boundary points and edges at each electrode.

    Attributes:
        radius: radius of circular domain
        nmb_electrodes: number of electrodes
        length_electrodes: angle that defines length of electrodes
        pts_fixed: Array of shape (nmb_electrodes * 3 x 2) that contains the
            points on the boundary defining each electrode.
        points: Array of shape (N x 2) with points that define the mesh.
            By definition, this array is a combination of the fixed points
            (defining the electrodes) and the ones generated, in this exact
            order.
        mesh_triangles: Array of shape (K x 3) with indexes that define each
            element of the mesh.
        nmb_points: Number of points in array points, for convenience.
        nmb_triangles: Number of triangles in the mesh, for convenence.
        index_boundary_pts: Array of shape (S x 1) with indexes of boundary
            points over the general array of points pts.
        electrode_pts: List of lists with indexes points for each electrode.
        electode_edges: List of lists with edges for each electrode.
    """

    def __init__(
        self,
        radius,
        nmb_electrodes,
        length_electrodes,
        initial_edge_length=0.1,
    ):
        electrode_arc_length = length_electrodes / radius

        self.radius = radius
        self.nmb_electrodes = nmb_electrodes
        self.electrode_arc_length = electrode_arc_length

        # Generates points that define the electrodes, which
        # will be fixed points in the mesh generation.
        self.pts_fixed = generate_equidistant_electrodes(
            radius, nmb_electrodes, electrode_arc_length)

        # Creates particular level-set function with a fixed radius,
        # from the general-one.
        circle_level_set = generate_circle_level_set(radius)

        # Creates mesh with equally distributed nodes and with the points
        # that define the electrodes fixed.

        # Defines the control parameters for DistMesh Algorithm by default.
        control_parameters = MeshControlParameters(
            initial_edge_length=initial_edge_length)

        domain_box = np.array([[-radius, radius], [-radius, radius]])

        self.points, self.mesh_triangles = dist_mesh_2d(
            control_parameters, circle_level_set, constant_relative_edge_length,
            initial_edge_length, domain_box, self.pts_fixed)

        # Define number of points and triangles
        self.nmb_points = len(self.points)
        self.nmb_triangles = len(self.mesh_triangles)

        # Defines the attributes boundary_pts and index_boundary_pts.
        # Because of self.pts structure, the points defining the electrodes will
        # be first, i.e., boundary_pts = [pts_fixed, generated_boundary_pts]
        self.index_boundary_pts = self.find_boundary_pts(
            circle_level_set, initial_edge_length)

        # Creates list of lists with the points of mesh on each electrode.
        self.electrode_pts = self.find_electrode_pts()

        # Defines list of lists with all edges on each electrode.
        self.electrode_edges = self.find_electrode_edges()

    def find_boundary_pts(self, level_set, initial_edge_length):
        """Finds all the points of the mesh on the boundary.

        Args:
            level_set: function handle that represents the
                level-set function of the boundary.

        Returns:
            Array of shape (S x 1) that contains the indexes
            of points on the boundary.
        """

        # Defines which points are near the boundary.
        close_to_boundary_condition = np.abs(level_set(self.points)) <= \
            1e-4 * initial_edge_length

        # Defines the indexes of the points in the boundary.
        index_boundary_pts = np.where(close_to_boundary_condition)[0]

        return index_boundary_pts

    def find_electrode_pts(self):
        """Determines the points inside each electrode in anti-clockwise
            order.

        Here, we find the points inside each electrode based on the
        angles of the boundary points. Our electrode is defined by the
        three points, as follows:

        Electrode = {left_extreme_angle,
                     left_extreme_angle + electrode_arc_length / 2,
                     left_extreme_angle + electrode_arc_length}.

        But notice, that all the angles between the three angles are
        also inside the respective electrode.

        Thus, we use purely the information about the angles of
        the boundary points to verify if they are inside the electrodes.

        Further, we know that the fixed points defining the electrodes will
        be the first on the boundary_pts. Consequently, they will be added
        first to their respective list. This allows for a simple
        ordering procedure, since only the last elements of each list will
        require ordering.

        Returns:
            List with lists of points inside each electrode sorted by
            their angle in anti-clockwise order.
        """

        lists_pts_in_electrode = [[3 * i, 3 * i + 1, 3 * i + 2] \
                                   for i in range(self.nmb_electrodes)]

        # Defines center angle of electrodes
        center_angles = center_angles_electrodes(self.nmb_electrodes,
                                                 self.electrode_arc_length)

        index_rest_bd_pts = \
            self.index_boundary_pts[3 * self.nmb_electrodes:]
        rest_boundary_pts = self.points[index_rest_bd_pts, :]

        #  Defines the angles of boundary points in [-pi, pi]
        angles_rest_boundary_pts = angles_0_2pi(rest_boundary_pts)

        # Checks if each boundary point is inside one of the electrodes.
        for bd_index, angle_bd in enumerate(angles_rest_boundary_pts):
            for ele_index, ext_angle in enumerate(center_angles):

                if abs(angle_bd - ext_angle) <= self.electrode_arc_length / 2:

                    lists_pts_in_electrode[ele_index] = \
                        self.__insert_angle_sorted(
                        angle_bd,
                        index_rest_bd_pts[bd_index],
                        lists_pts_in_electrode[ele_index])

                    break

        # Sort each inside list by anti-clockwise order of angles
        return lists_pts_in_electrode

    def __insert_angle_sorted(self, new_angle, new_angle_index,
                              electrode_pts_list):
        """ Inserts index of angle into list of points inside
        an electrode, in a sorted manner.

        Args:
            new_angle: angle used to compare with the ones the
                list represents.
            new_angle_index: index of new angle in the general
                order of the mesh.
            electrode_pts_list: List with points already inside
                each electrode.

        Returns:
            List with the index of the new angle into place.
        """

        electrode_pts = self.points[electrode_pts_list, :]
        angles_electrode_pts_list = angles_0_2pi(electrode_pts)

        for index, angle in enumerate(angles_electrode_pts_list):

            if new_angle < angle:
                electrode_pts_list.insert(index, new_angle_index)
                break

        return electrode_pts_list

    def find_electrode_edges(self):
        """ Forms the edges for each electrode based on their inside points.

        An electrode can contain more that its defining points. For example,
        let [a, b, c, d] be a list of the indexes of points inside each
        electrode. From here, the set of edges for this electrode is given
        as [[a, b], [b, c], [c, d]].

        Returns:
            List with a list for each electrode that contains the edges
            that are inside of it.
        """

        # List of Edges in each electrode
        edges = [[] for j in range(self.nmb_electrodes)]

        # Iterates through each electrode.
        for index_ele, ele in enumerate(self.electrode_pts):
            # Iterates through the points in each electrode
            for index_pt, point in enumerate(ele[0:len(ele) - 1]):
                # Adds the edge with the current point and the next one.
                edges[index_ele].append([point, ele[index_pt + 1]])

        return edges

    def plot_mesh(self, output_directory=None):
        """ Plots the mesh with all the points and elements in
            mesh highlighted. """

        # Split x, y-coordinates of points
        pts_x, pts_y = self.points.T[:]
        pts_fixed_x, pts_fixed_y = self.pts_fixed.T[:]

        fig, axes = plt.subplots()
        axes.set_title("Plot of Domain Mesh")

        axes.triplot(pts_x, pts_y, self.mesh_triangles)
        axes.plot(pts_x, pts_y, ".")
        axes.plot(pts_fixed_x, pts_fixed_y, "x")

        if output_directory is not None:
            fig.savefig(os.path.join(output_directory, "mesh.png"))
        else:
            plt.show()

        plt.close()


def center_angles_electrodes(nmb_electrodes, electrode_arc_length):
    """Determines angles of electrodes center. """

    angle_step = (2 * math.pi / nmb_electrodes)
    center_angle =  angle_step * np.arange(nmb_electrodes) + \
        electrode_arc_length / 2

    return center_angle


def angles_0_2pi(pts):
    """Finds the angles of a set of points in the range [0, 2*pi]

    Args:
        pts: Array of shape (K x 2) with 2D points.

    Returns:
        angles: Array of shape (K x 1) with the respective angles.
    """
    # Finds angles in interval [-pi, pi]
    x_pt, y_pt = pts.T[:]

    angles = np.arctan2(y_pt, x_pt)
    # Determines indexes of angles in [-pi, 0[
    negative_angle_indexes = np.where(angles < 0)[0]

    # Shifts to [pi, 2*pi[
    angles[negative_angle_indexes] += 2 * math.pi

    return angles


def constant_relative_edge_length(pts):
    """Defines the relative edge length distribution over domain used
        on DistMesh.

    For certain applications there might be a need of denser regions of
    points in the mesh. This function describes equally distributed
    meshes, where the desired edge length will be the same everywhere.

    Returns:
        Array of 1's with length of input. Represents the probability
        to keep each of these points.
    """

    return np.ones((len(pts), 1))


def generate_circle_level_set(radius):
    """Defines the level-set function for a circle with a
    certain radius.

    Args:
        radius: radius of circular domain.

    Returns:
        Function that takes as input an Array with shape
        (N x 2) and returns an Array of shape (N x 1) with the
        level-set function of the circular domain with the defined
        radius evaluated at input points.
    """

    def circle_level_set(pts):
        return np.linalg.norm(pts, axis=1) - radius

    return circle_level_set


def generate_equidistant_electrodes(radius, nmb_electrodes,
                                    electrode_arc_length):
    """ Creates set of 2D points on boundary that define the
    equidistant electrodes.

    We define the equidistant electrodes by dividing the circle in
    nmb_electrodes parts. Each electrode is represented by three points:
    left-extreme (alpha), center of electrode (alpha + length_ele / 2)
    and the right-extreme (alpha + length_ele)

    Args:
        radius: radius of the circular domain.
        nmb_electrodes: number of electrodes.
        electrode_arc_length: angle length of each electrode.

    Returns:
        Array of shape ( nmb_electrodes * 3, 2) that represents
        2D points that define the electrodes on the boundary.
    """

    pts_ele = np.zeros((nmb_electrodes * 3, 2))

    center_angles = center_angles_electrodes(nmb_electrodes,
                                             electrode_arc_length)

    electrode_angles = np.array([
        center_angles - electrode_arc_length / 2, center_angles,
        center_angles + electrode_arc_length / 2
    ]).T.flatten()

    pts_ele = radius * np.array(
        [np.cos(electrode_angles),
         np.sin(electrode_angles)]).T

    return pts_ele

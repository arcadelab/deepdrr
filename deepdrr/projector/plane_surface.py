from typing import Tuple

from deepdrr import geo
import numpy as np

class PlaneSurface:
    plane_vector: np.ndarray
    surface_origin: np.ndarray
    basis_1: np.ndarray
    basis_2: np.ndarray
    bounds: np.ndarray
    orthogonal: bool
    

    def __init__(
        self,
        plane_vector: np.ndarray,
        surface_origin: geo.Point3D,
        basis: Tuple[geo.Vector3D, geo.Vector3D],
        bounds: np.ndarray,
        orthogonal: bool
    ):
        """A representation of a rectangular region on a 2D plane embedded in 3D space.

        Args:
            plane_vector (np.ndarray): a 1x4 matrix uniquely identifying the plane.  The first three elements are the normal vector, and the fourth element is the distance between the plane and the origin of the coordinate axes.  \\vec{m} = (n_x, n_y, n_z, d), where \\hat{n} is the normal vector to the plane, and 'd' is the distance away from the origin
            surface_origin (geo.Point3D): a point on the plane that is used as a reference point for every other position on the surface.  Points on the plane are only on the surface if the point is [surface_origin] + [in-bounds coefficients] * [basis]
            basis (Tuple[geo.Vector3D, geo.Vector3D]): an orthogonal (NOT necessarily orthonormal) basis for the plane.  All points on the plane are represented by [surface_origin] + [linear combination of basis]
            bounds (np.ndarray): a 2x2 matrix that provides the in-bounds coefficients for the linear combination of the basis vectors.  bounds[0,0] and bounds[0,1] give the lower and upper bounds (inclusive) on the acceptable coefficient for the first basis vector, and bounds[1,0] and bounds[1,1] behave the same for the second basis vector
            orthogonal (bool): if True, then the user guarantees that the provided basis is orthogonal.
        """
        self.plane_vector = plane_vector.copy()
        self.surface_origin = geo.Point3D.from_any(surface_origin)
        self.basis_1 = geo.Vector3D.from_any(basis[0])
        self.basis_2 = geo.Vector3D.from_any(basis[1])
        self.bounds = bounds.copy()
        self.orthogonal = orthogonal

        # Check that the basis is orthogonal if it is purported to be:
        if self.orthogonal:
            b1_dot_b2 = (self.basis_1[0] * self.basis_2[0]) + (self.basis_1[1] * self.basis_2[1]) + (self.basis_1[2] * self.basis_2[2])
            assert np.isclose(0, b1_dot_b2)

    def check_ray_intersection(
        self,
        pos: geo.Point3D,
        direction: geo.Vector3D
    ) -> np.float32:
        """Calculates whether or not a photon at the specified position, travelling in the specified direction, will hit the plane of the PlaneSurface object.

        It is imperative that all of the arguments are in the same coordinate system (unchecked).

        Args:
            pos (geo.Point3D): the position of the photon
            dir (geo.Vector3D): the direction that the photon is travelling in

        Returns:
            np.float32: if there will be an intersection, the distance to the intersection.  If no intersection, returns a negative number (the negative number does not necessarily have a geometrical meaning) 
        """
        # (\vec{pos} + \alpha * \vec{dir}) \cdot \vec{m} = 0, then (\vec{pos} + \alpha * \vec{dir}) is the point of intersection.
        # Want to return None if \alpha < 0, since then the photon is actually travelling the wrong way to hit the plane.
        r_dot_m = (pos.data) @ self.plane_vector
        if 0 == r_dot_m:
            # 'pos' is already on the plane
            return 0
        d_dot_m = (direction.data) @ self.plane_vector
        if 0 == d_dot_m:
            # 'direction' is perpendicular to the normal vector of the plane ==> will never intersect
            return -1

        return (-1 * r_dot_m) / d_dot_m

    def point_on_surface(
        self,
        point: geo.Point3D
    ) -> bool:
        """Returns whether the given point, which is assumed to be on the plane of the surface, is within the bounds of the surface.

        Args:
            point (geo.Point3D): the point to check
        
        Returns:
            bool: True if the point is in-bounds, False otherwise
        """
        coef_1, coef_2 = self.get_lin_comb_coefs(point)

        if (coef_1 < self.bounds[0,0]) or (coef_1 > self.bounds[0,1]):
            return False

        return (coef_2 >= self.bounds[1,0]) and (coef_2 <= self.bounds[1,1])

    def point_on_surface_checking(
        self,
        point: geo.Point3D
    ) -> bool:
        """Returns whether the given point, which is not assumed to be on the plane of the surface, is within the bounds of the surface.

        Args:
            point (geo.Point3D): the point to check
        
        Returns:
            bool: True if the point is on the plane and in-bounds, False otherwise
        """
        pvec = point.data
        assert (4,) == pvec.shape
        assert 1 == pvec[-1]

        if (0 == np.dot(pvec, self.plane_vector)):
            return self.point_on_surface(point)
        return False

    def get_lin_comb_coefs(
        self,
        point: geo.Point3D
    ) -> Tuple[np.float32, np.float32]:
        """Returns the 'coordinates' of the point in the plane, where the 'origin' in the plane is PlaneSurface.surface_origin, 
        and where the coordinate axes correspond to the PlaneSurface basis vectors.

        Args:
            point (geo.Point3D): the point to check
        
        Returns:
            Tuple[np.float32, np.float32]: the coefficients for the two basis vectors
        """
        if self.orthogonal:
            return self._get_lin_comb_orthogonal(point)
        else:
            return self._get_lin_comb_general(point)

    def _get_lin_comb_orthogonal(
        self,
        point: geo.Point3D
    ) -> Tuple[np.float32, np.float32]:
        """Returns the 'coordinates' of the point in the plane, where the 'origin' in the plane is PlaneSurface.surface_origin, 
        and where the coordinate axes correspond to the PlaneSurface basis vectors.  Assumes that the basis vectors are orthogonal.

        Args:
            point (geo.Point3D): the point to check
        
        Returns:
            Tuple[np.float32, np.float32]: the coefficients for the two basis vectors
        """
        # Working in 3D (i.e., non-homogeneous) coordinates:
        # Let \vec{x} := point, \vec{s} := surface_origin. Thus, (\vec{x} - \vec{s}) = \alpha_1 * b_1 + \alpha_2 * b_2, 
        # where {b_1, b_2} is the orthonormal basis.  In other terms,
        #
        #       (\vec{x} - \vec{s}) = (b_1 b_2) (\alpha_1 \alpha_2)^T = B \vec{\alpha}
        #
        # Accordingly,
        # 
        #       \vec{\alpha} = B^{-1} (\vec{x} - \vec{s})
        # 
        # where B^{-1} is the left-inverse of the 3x2 matrix B = (b_1 b_2).  Since the basis {b_1, b_2} is orthogonal,
        # we have:
        # 
        #       B^{-1} = ( b_1^T / magnitude(b_1)^2
        #                  b_2^T / magnitude(b_2)^2 )
        #
        # Once we have \vec{\alpha}, we just need to check against the acceptable bounds, given by self.bounds

        disp_np_3 = (point - self.surface_origin).data[0:3] # \vec{x} - \vec{s}, only 3 values (non-homogeneous)

        b_1_np_3 = self.basis_1.data[0:3] # b_1, only 3 values (non-homogeneous)
        alpha_1 = np.dot(b_1_np_3, disp_np_3) / np.dot(b_1_np_3, b_1_np_3)

        b_2_np_3 = self.basis_2.data[0:3] # b_2, only 3 values (non-homogeneous)
        alpha_2 = np.dot(b_2_np_3, disp_np_3) / np.dot(b_2_np_3, b_2_np_3)

        return alpha_1, alpha_2

    def _get_lin_comb_general(
        self,
        point: geo.Point3D
    ) -> Tuple[np.float32, np.float32]:
        """Returns the 'coordinates' of the point in the plane, where the 'origin' in the plane is PlaneSurface.surface_origin, 
        and where the coordinate axes correspond to the PlaneSurface basis vectors.  Does not assume that the basis vectors are orthogonal.

        Args:
            point (geo.Point3D): the point to check
        
        Returns:
            Tuple[np.float32, np.float32]: the coefficients for the two basis vectors
        """
        # Working in 3D (i.e., non-homogeneous) coordinates:
        # Let \vec{x} := point, \vec{s} := surface_origin. Thus, (\vec{x} - \vec{s}) = \alpha_1 * b_1 + \alpha_2 * b_2, 
        # where {b_1, b_2} is the orthonormal basis.  In other terms,
        #
        #       (\vec{x} - \vec{s}) = (b_1 b_2) (\alpha_1 \alpha_2)^T = B \vec{\alpha}
        #
        # Accordingly,
        # 
        #       \vec{\alpha} = B^{-1} (\vec{x} - \vec{s})
        # 
        # where B^{-1} is the left-inverse of the 3x2 matrix B = (b_1 b_2).  Since the basis vectors are linearly independent,
        # (B^T B) is invertible.  Accordingly, ((B^T B)^{-1} B^T) is the left-inverse of B. 
        # 
        # Let D_ij := dot_product(b_i, b_j) be the dot product between the i-th and j-th basis vectors.
        # 
        #       (B^T B) = [D_11 D_12]
        #                 [D_21 D_22]
        # 
        #       (B^T B)^{-1} = (1 / (D_11 D_22 - D_12 D_21)) [ D_22 -D_12]
        #                                                    [-D_21  D_11]
        #
        #                    = (1 / det(B^T B)) [ D_22 -D_12]
        #                                       [-D_21  D_11]
        #
        # Thus, \vec{\alpha} = (B^T B)^{-1} (B^T (\vec{x} - \vec{s})) by the associative property of matrix multiplication.
        # 
        #   \vec{\beta} = B^T (\vec{x} - \vec{s})
        #
        #   \vec{\alpha} = (B^T B)^{-1} \vec{\beta}
        #                = (1 / det(B^T B)) [ D_22 -D_12] [beta_1]
        #                                   [-D_21  D_11] [beta_2]
        #
        # Once we have \vec{\alpha}, we just need to check against the acceptable bounds, given by self.bounds

        disp_np_3 = (point - self.surface_origin).data[0:3] # \vec{x} - \vec{s}, only 3 values (non-homogeneous)

        b_1_np_3 = self.basis_1.data[0:3] # b_1, only 3 values (non-homogeneous)
        b_2_np_3 = self.basis_2.data[0:3] # b_2, only 3 values (non-homogeneous)

        D_11 = np.dot(b_1_np_3, b_1_np_3)
        D_22 = np.dot(b_2_np_3, b_2_np_3)
        D_cross = np.dot(b_1_np_3, b_2_np_3)
        det = (D_11 * D_22) - (D_cross * D_cross)

        beta_1 = np.dot(b_1_np_3, disp_np_3)
        beta_2 = np.dot(b_2_np_3, disp_np_3)

        alpha_1 = (D_22 * beta_1 - D_cross * beta_2) / det

        alpha_2 = (-D_cross * beta_1 + D_11 * beta_2) / det

        return alpha_1, alpha_2

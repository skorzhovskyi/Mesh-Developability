
import numpy as np
import open3d as o3d
from scipy.sparse import lil_matrix, vstack
from scipy.optimize import least_squares

def compute_laplacian_matrix(mesh):
    """
    Compute the Laplacian matrix of the mesh for smoothness regularization.
    """
    n_vertices = len(mesh.vertices)
    L = lil_matrix((n_vertices, n_vertices))

    adjacency_list = [[] for _ in range(n_vertices)]
    triangles = np.asarray(mesh.triangles)

    for tri in triangles:
        for i in range(3):
            vi = tri[i]
            vj = tri[(i + 1) % 3]
            adjacency_list[vi].append(vj)
            adjacency_list[vj].append(vi)

    for i in range(n_vertices):
        neighbors = adjacency_list[i]
        n_neighbors = len(neighbors)
        L[i, i] = n_neighbors
        for j in neighbors:
            L[i, j] = -1

    return L

def assign_triangles_to_planes(normals, centroids, planes, weights=(1.0, 1.0)):
    """
    Assign each triangle to the best-fitting plane based on angular and positional deviation.
    """
    w_theta, w_d = weights
    assignments = []
    for n_t, c_t in zip(normals, centroids):
        min_deviation = np.inf
        best_plane_idx = -1

        for idx, (n_p, p_p) in enumerate(planes):
            # Angular deviation
            cos_theta = np.clip(np.dot(n_t, n_p), -1.0, 1.0)
            theta = np.arccos(cos_theta)

            # Positional deviation
            distance = abs(np.dot(n_p, c_t - p_p))

            # Total deviation
            total_deviation = w_theta * theta + w_d * distance

            if total_deviation < min_deviation:
                min_deviation = total_deviation
                best_plane_idx = idx

        assignments.append(best_plane_idx)
    return assignments

def energy_function(vertex_positions, mesh, planes, assignments, laplacian, lambda_smooth):
    """
    Energy function to minimize.
    """
    vertices = vertex_positions.reshape((-1, 3))
    triangles = np.asarray(mesh.triangles)
    n_vertices = len(vertices)

    # Plane deviation terms
    plane_errors = []
    for tri_idx, tri in enumerate(triangles):
        plane_idx = assignments[tri_idx]
        n_p, p_p = planes[plane_idx]

        v1, v2, v3 = vertices[tri]
        centroid = (v1 + v2 + v3) / 3
        # Distance from centroid to plane
        distance = np.dot(n_p, centroid - p_p)
        plane_errors.append(distance)

    plane_errors = np.array(plane_errors)

    # Smoothness terms
    L = laplacian.tocsr()
    delta_V = L.dot(vertices)
    smoothness_errors = lambda_smooth * delta_V.flatten()

    # Total energy
    energy = np.concatenate([plane_errors, smoothness_errors])

    return energy

def main():
    # Load or create a mesh
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
    mesh.compute_vertex_normals()

    # Define a set of planes (normal vector and a point on the plane)
    planes = [
        (np.array([1, 0, 0]), np.array([0, 0, 0])),  # YZ plane
        (np.array([0, 1, 0]), np.array([0, 0, 0])),  # XZ plane
        (np.array([0, 0, 1]), np.array([0, 0, 0])),  # XY plane
    ]

    planes = [
        (np.array([1, 0, 0]), np.array([1, 0, 0])),  # YZ plane
        (np.array([0, 1, 0]), np.array([0, 1, 0])),  # XZ plane
        (np.array([0, 0, 1]), np.array([0, 0, 1])),  # XY plane
        (np.array([-1, 0, 0]), np.array([-1, 0, 0])),  # YZ plane
        (np.array([0, -1, 0]), np.array([0, -1, 0])),  # XZ plane
        (np.array([0, 0, -1]), np.array([0, 0, -1])),  # XY plane
    ]

    # Compute normals and centroids of each triangle
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    normals = []
    centroids = []

    for tri in triangles:
        v1, v2, v3 = vertices[tri]
        n = np.cross(v2 - v1, v3 - v1)
        n /= np.linalg.norm(n)
        centroid = (v1 + v2 + v3) / 3
        normals.append(n)
        centroids.append(centroid)

    normals = np.array(normals)
    centroids = np.array(centroids)

    # Assign each triangle to the best-fitting plane
    assignments = assign_triangles_to_planes(normals, centroids, planes, weights=(1.0, 0.1))

    # Compute Laplacian matrix for smoothness regularization
    laplacian = compute_laplacian_matrix(mesh)

    # Set regularization weight
    lambda_smooth = 0.1

    # Initial vertex positions
    vertex_positions = vertices.flatten()

    # Define the residual function for least squares optimization
    residual_function = lambda x: energy_function(
        x, mesh, planes, assignments, laplacian, lambda_smooth
    )

    # Run the optimization
    result = least_squares(
        residual_function,
        vertex_positions,
        method='lm',
        verbose=2,
        xtol=1e-6,
        ftol=1e-6,
        max_nfev=200
    )

    # Update mesh vertices with optimized positions
    optimized_vertices = result.x.reshape((-1, 3))
    mesh.vertices = o3d.utility.Vector3dVector(optimized_vertices)

    o3d.io.write_triangle_mesh("adjusted_mesh.ply", mesh)

if __name__ == "__main__":
    main()
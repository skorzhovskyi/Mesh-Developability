import trimesh
import numpy as np
import torch
import torch_scatter
import pymeshfix

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to compute normals of the triangles
def compute_triangle_normals(V, F):
    # Compute the normals of all triangles in the mesh
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # Specify dim=1
    normals = normals / torch.norm(normals, dim=1, keepdim=True)  # Normalize the normals
    return normals

def laplacian_smoothing(V, F, smoothing_factor=0.01):
    N = V.shape[0]
    V_np = V.cpu().numpy()
    F_np = F.cpu().numpy()

    # Compute the adjacency list
    neighbors = [[] for _ in range(N)]
    for face in F_np:
        for i in range(3):
            v0 = face[i]
            v1 = face[(i + 1) % 3]
            v2 = face[(i + 2) % 3]
            neighbors[v0].extend([v1, v2])

    # Convert to set to remove duplicates and back to list
    neighbors = [list(set(neigh)) for neigh in neighbors]

    # Apply Laplacian smoothing
    V_smoothed = V_np.copy()
    for i in range(N):
        if len(neighbors[i]) > 0:
            neighbor_vertices = V_np[neighbors[i], :]
            V_smoothed[i, :] = V_np[i, :] * (1 - smoothing_factor) + neighbor_vertices.mean(axis=0) * smoothing_factor

    # Convert back to torch tensor
    V_smoothed = torch.tensor(V_smoothed, dtype=V.dtype, device=V.device)
    return V_smoothed

# Function to collapse short edges
def collapse_short_edges(V, F, length_threshold):
    """
    Collapse edges shorter than length_threshold.
    V: (N, 3) tensor of vertices
    F: (M, 3) tensor of face indices
    Returns updated V and F.
    """
    # Extract edges from faces
    edges = torch.cat([
        F[:, [0, 1]],
        F[:, [1, 2]],
        F[:, [2, 0]]
    ], dim=0)  # Shape: (3*M, 2)

    # Sort edges to ensure uniqueness (since edges are undirected)
    edges_sorted = torch.sort(edges, dim=1)[0]  # Shape: (3*M, 2)

    # Get unique edges
    edges_unique = torch.unique(edges_sorted, dim=0)  # Shape: (num_edges, 2)

    # Compute edge lengths
    V_i = V[edges_unique[:, 0], :]  # Shape: (num_edges, 3)
    V_j = V[edges_unique[:, 1], :]  # Shape: (num_edges, 3)
    edge_lengths = torch.norm(V_i - V_j, dim=1)  # Shape: (num_edges,)

    # Find edges to collapse
    collapse_mask = edge_lengths < length_threshold  # Shape: (num_edges,)

    edges_to_collapse = edges_unique[collapse_mask]  # Shape: (num_edges_to_collapse, 2)

    N = V.shape[0]

    parent = torch.arange(N, device=V.device)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root != v_root:
            parent[v_root] = u_root

    for edge in edges_to_collapse:
        vi_idx = edge[0].item()
        vj_idx = edge[1].item()
        union(vi_idx, vj_idx)

    # Update parent to find root of each vertex
    for i in range(N):
        parent[i] = find(i)

    # Map old indices to new indices
    unique_parents, new_indices = torch.unique(parent, return_inverse=True)

    V_new = V[unique_parents]  # (num_unique_vertices, 3)
    F_new = new_indices[F]  # Update face indices

    # Remove degenerate faces
    valid_faces_mask = (F_new[:, 0] != F_new[:, 1]) & (F_new[:, 1] != F_new[:, 2]) & (F_new[:, 2] != F_new[:, 0])
    F_new = F_new[valid_faces_mask]

    return V_new, F_new

# Function to flip edges in thin triangles
def flip_edges_in_thin_triangles(V, F, angle_threshold_degrees):
    """
    Flip the longest edge of triangles that are too thin.
    V: (N, 3) tensor of vertices
    F: (M, 3) tensor of face indices
    angle_threshold_degrees: float, threshold angle in degrees
    Returns updated F (V remains unchanged)
    """
    angle_threshold_radians = angle_threshold_degrees * np.pi / 180.0

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    l0 = torch.norm(e0, dim=1)
    l1 = torch.norm(e1, dim=1)
    l2 = torch.norm(e2, dim=1)

    # Compute angles at each vertex using the Law of Cosines
    cos_theta0 = (l2**2 + l0**2 - l1**2) / (2 * l2 * l0)
    cos_theta1 = (l0**2 + l1**2 - l2**2) / (2 * l0 * l1)
    cos_theta2 = (l1**2 + l2**2 - l0**2) / (2 * l1 * l2)

    # Clamp cos_theta to [-1,1] to avoid numerical errors
    cos_theta0 = torch.clamp(cos_theta0, -1.0, 1.0)
    cos_theta1 = torch.clamp(cos_theta1, -1.0, 1.0)
    cos_theta2 = torch.clamp(cos_theta2, -1.0, 1.0)

    angle0 = torch.acos(cos_theta0)
    angle1 = torch.acos(cos_theta1)
    angle2 = torch.acos(cos_theta2)

    angles = torch.stack([angle0, angle1, angle2], dim=1)  # Shape: (M, 3)

    min_angles, _ = torch.min(angles, dim=1)  # Shape: (M,)

    thin_triangles_mask = min_angles < angle_threshold_radians

    thin_triangle_indices = torch.nonzero(thin_triangles_mask).squeeze(1)

    if thin_triangle_indices.numel() == 0:
        # No thin triangles
        return F

    # For each thin triangle, find its longest edge
    l_stack = torch.stack([l0, l1, l2], dim=1)  # Shape: (M, 3)
    longest_edge_indices = torch.argmax(l_stack, dim=1)  # Shape: (M,)

    # For each thin triangle, get its longest edge
    triangle_edges = torch.stack([
        F[:, [0,1]],
        F[:, [1,2]],
        F[:, [2,0]]
    ], dim=1)  # Shape: (M, 3, 2)

    # Edges to flip
    edges_to_flip = triangle_edges[thin_triangle_indices, longest_edge_indices[thin_triangle_indices], :]  # Shape: (num_thin_triangles, 2)

    # Build edge to face mapping
    edges = triangle_edges.reshape(-1, 2)  # Shape: (3*M, 2)
    face_indices = torch.arange(F.shape[0], device=V.device).repeat_interleave(3)  # Shape: (3*M,)
    edge_keys = torch.sort(edges, dim=1)[0]  # Shape: (3*M, 2)

    edge_to_face = {}

    for i in range(edge_keys.shape[0]):
        edge = tuple(edge_keys[i].cpu().numpy())
        face_idx = face_indices[i].item()
        if edge in edge_to_face:
            edge_to_face[edge].append(face_idx)
        else:
            edge_to_face[edge] = [face_idx]

    modified_faces = set()

    for k in range(edges_to_flip.shape[0]):
        edge = edges_to_flip[k]
        edge_sorted, _ = torch.sort(edge)
        edge_key = tuple(edge_sorted.cpu().numpy())
        face_list = edge_to_face.get(edge_key, [])
        if len(face_list) != 2:
            # Cannot flip edge, not shared by exactly two faces
            continue
        f1_idx, f2_idx = face_list
        if f1_idx in modified_faces or f2_idx in modified_faces:
            # Faces already modified
            continue

        # Get the two faces
        f1 = F[f1_idx]
        f2 = F[f2_idx]

        needs_flip = False

        if (f1[[0, 1]] == edge).all() or (f1[[1, 2]] == edge).all() or (f1[[2, 0]] == edge).all():
            needs_flip = True

        # Get the vertices opposite the shared edge in each face
        f1_vertices = set(f1.cpu().numpy().tolist())
        f2_vertices = set(f2.cpu().numpy().tolist())
        shared_vertices = set(edge.cpu().numpy().tolist())
        v1_opposite = list(f1_vertices - shared_vertices)[0]
        v2_opposite = list(f2_vertices - shared_vertices)[0]

        # Check if the new edge would create a valid face
        if v1_opposite == v2_opposite:
            # Degenerate face
            continue

        if needs_flip:
            v1_opposite, v2_opposite = v2_opposite, v1_opposite

        # Create new faces with the flipped edge
        new_f1 = torch.tensor([v1_opposite, v2_opposite, edge[0].item()], device=V.device)
        new_f2 = torch.tensor([v2_opposite, v1_opposite, edge[1].item()], device=V.device)

        # Update the faces
        F[f1_idx] = new_f1
        F[f2_idx] = new_f2

        modified_faces.update([f1_idx, f2_idx])

    return F

# Updated optimize_mesh function with vectorized code
def optimize_mesh(V_init, F, num_epochs=1000, learning_rate=1e-2, smoothing_interval=100, regularization_weight=0.01,
                  operation_interval=100, length_threshold=0.01, angle_threshold_degrees=20.0):
    # Ensure V_init and F are NumPy arrays
    V_init = np.asarray(V_init)
    F_cpu = np.asarray(F)

    # Convert to tensors on the GPU
    V = torch.tensor(V_init, dtype=torch.float32, requires_grad=True, device=device)
    F = torch.tensor(F_cpu, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam([V], lr=learning_rate)

    for epoch in range(num_epochs):
        # Update N and M after any potential modifications to V and F
        N = V.shape[0]
        M = F.shape[0]

        optimizer.zero_grad()

        # Recompute triangle normals based on current vertex positions
        normals = compute_triangle_normals(V, F)  # (M, 3)

        # Build the mapping from vertices to adjacent faces
        vertex_face_indices = F.reshape(-1)  # Shape: (M*3,)
        face_indices_repeated = torch.arange(M, device=device).repeat_interleave(3)  # Shape: (M*3,)

        # Normals per face vertex
        normals_per_vertex = normals[face_indices_repeated]  # Shape: (M*3, 3)

        # Compute outer products of normals
        outer_products = normals_per_vertex.unsqueeze(2) * normals_per_vertex.unsqueeze(1)  # (M*3, 3, 3)

        # Reshape outer_products to (M*3, 9)
        outer_products_flat = outer_products.reshape(-1, 9)  # Shape: (M*3, 9)

        # Use torch_scatter.scatter_add to accumulate outer products per vertex
        S_flat = torch_scatter.scatter_add(
            outer_products_flat,
            vertex_face_indices.unsqueeze(1).expand(-1, 9),
            dim=0,
            dim_size=N
        )

        # Reshape S_flat back to (N, 3, 3)
        S = S_flat.reshape(N, 3, 3)

        # Compute counts per vertex using torch_scatter
        counts = torch_scatter.scatter_add(
            torch.ones_like(vertex_face_indices, dtype=torch.float32),
            vertex_face_indices,
            dim=0,
            dim_size=N
        )

        n_i = counts  # (N,)

        # Avoid division by zero
        mask = n_i > 1

        # Compute covariance matrices without in-place operations
        C_i = torch.where(
            mask.view(-1, 1, 1),
            S / (n_i.view(-1, 1, 1) - 1.0),
            torch.zeros_like(S)
        )

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(C_i)  # (N, 3)

        # Zero out eigenvalues where mask is False
        eigenvalues = torch.where(
            mask.view(-1, 1),
            eigenvalues,
            torch.zeros_like(eigenvalues)
        )

        # Get smallest eigenvalues
        lambda_i = eigenvalues[:, 0]  # (N,)

        # Compute the objective function value
        f = lambda_i.sum()

        # Regularization to make neighbor distances as similar as possible
        # Extract edges from faces
        edges = torch.cat([
            F[:, [0, 1]],
            F[:, [1, 2]],
            F[:, [2, 0]]
        ], dim=0)  # Shape: (3*M, 2)

        # Sort edges to ensure uniqueness (since edges are undirected)
        edges_sorted = torch.sort(edges, dim=1)[0]  # Shape: (3*M, 2)

        # Get unique edges
        edges_unique = torch.unique(edges_sorted, dim=0)  # Shape: (num_edges, 2)

        edge_vertices = edges_unique  # Shape: (num_edges, 2)

        # Compute edge lengths
        V_i = V[edge_vertices[:, 0], :]  # Shape: (num_edges, 3)
        V_j = V[edge_vertices[:, 1], :]  # Shape: (num_edges, 3)
        edge_lengths = torch.norm(V_i - V_j, dim=1)  # Shape: (num_edges,)

        # Prepare to compute per-vertex mean edge lengths
        vertex_indices_i = edge_vertices[:, 0]  # Shape: (num_edges,)
        vertex_indices_j = edge_vertices[:, 1]  # Shape: (num_edges,)

        vertex_indices_all = torch.cat([vertex_indices_i, vertex_indices_j], dim=0)  # Shape: (2*num_edges,)
        edge_lengths_all = torch.cat([edge_lengths, edge_lengths], dim=0)  # Shape: (2*num_edges,)

        edge_counts = torch.ones_like(edge_lengths_all)  # Shape: (2*num_edges,)

        # Compute per-vertex sum of edge lengths and counts
        sum_lengths = torch_scatter.scatter_add(edge_lengths_all, vertex_indices_all, dim=0, dim_size=N)
        counts = torch_scatter.scatter_add(edge_counts, vertex_indices_all, dim=0, dim_size=N)

        mean_lengths = sum_lengths / counts  # Shape: (N,)

        # Now compute per-edge deviations from per-vertex mean lengths
        mean_lengths_i = mean_lengths[vertex_indices_i]  # Shape: (num_edges,)
        mean_lengths_j = mean_lengths[vertex_indices_j]  # Shape: (num_edges,)

        deviations_i = (edge_lengths - mean_lengths_i) ** 2
        deviations_j = (edge_lengths - mean_lengths_j) ** 2

        # Collect deviations per vertex
        vertex_indices_all = torch.cat([vertex_indices_i, vertex_indices_j], dim=0)  # (2*num_edges,)
        deviations_all = torch.cat([deviations_i, deviations_j], dim=0)  # (2*num_edges,)

        sum_deviations = torch_scatter.scatter_add(deviations_all, vertex_indices_all, dim=0, dim_size=N)  # Shape: (N,)

        variance = sum_deviations / counts  # Shape: (N,)

        # Compute the regularization term
        regularization_term = variance.sum()  # Scalar

        # Total loss
        f_total = f + regularization_weight * regularization_term

        f_total.backward()
        optimizer.step()

        # Perform edge collapse and edge flip operations at specified intervals
        if epoch > 0 and epoch % operation_interval == 0:
            with torch.no_grad():
                # Collapse short edges
                V, F = collapse_short_edges(V, F, length_threshold)
                # Flip edges in thin triangles
                F = flip_edges_in_thin_triangles(V, F, angle_threshold_degrees)
                # Ensure V requires grad
                V.requires_grad_()
                # Re-initialize optimizer with new V
                optimizer = torch.optim.Adam([V], lr=learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Objective Function Value = {f.item()}, Regularization = {regularization_term.item()}")

    optimized_V = V.detach().cpu().numpy()
    optimized_F = F.detach().cpu().numpy()
    return optimized_V, optimized_F

# Generate a sphere mesh using trimesh
def generate_sphere_mesh(radius=1.0, subdivisions=3):
    # Create a sphere mesh with the specified radius and subdivisions
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    vertices = sphere.vertices
    faces = sphere.faces
    return vertices, faces

# Save the mesh to disk
def save_mesh(vertices, faces, file_path):
    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # Export the mesh to an OBJ file
    mesh.export(file_path)
    print(f"Mesh saved to {file_path}")

# Load a mesh from a PLY file using trimesh
def load_mesh_from_ply(file_path):
    # Load the mesh from a PLY file
    mesh = trimesh.load(file_path)
    vertices = mesh.vertices
    faces = mesh.faces
    return vertices, faces

# Main example usage
if __name__ == "__main__":
    # Generate a sphere mesh
    # V_init, F = generate_sphere_mesh(radius=1.0, subdivisions=3)

    print(f"Using device: {device}")

    input_ply_file = "input_mesh_6.ply"
    V_init, F = load_mesh_from_ply(input_ply_file)

    # Optimize the mesh
    optimized_V, optimized_F = optimize_mesh(
        V_init, F,
        num_epochs=1000,
        learning_rate=1e-2,
        smoothing_interval=100,
        regularization_weight=0.01,
        operation_interval=100,
        length_threshold=0.05,
        angle_threshold_degrees=20.0
    )

    save_mesh(optimized_V, optimized_F, "optimized_mesh.obj")

    # Print the initial and optimized vertices (optional)
    print("Initial vertices:\n", V_init[:5])  # Display the first 5 vertices
    print("Optimized vertices:\n", optimized_V[:5])  # Display the first 5 optimized vertices

import torch
import numpy as np
import torch.nn.functional as F


def append_edge(edge_map_, edges_, idx_a, idx_b):
    if idx_b < idx_a:
        idx_a, idx_b = idx_b, idx_a

    if not (idx_a, idx_b) in edge_map_:
        e_id = len(edges_)
        edges_.append([idx_a, idx_b])
        edge_map_[(idx_a, idx_b)] = e_id
        edge_map_[(idx_b, idx_a)] = e_id


def edge_subdivide(vertices, uvs, faces, uvfaces):
    """
    subdivides mesh based on edge midpoints. every triangle is subdivided into 4 child triangles.
    old faces are kept in array
    :param vertices: V x 3 ... vertex coordinates
    :param uvs: T x 2 ... uv coordinates
    :param faces: F x 3 face vertex idx array
    :param uvfaces: F x 3 face uv idx array
    :return:
        - vertices ... np.array of vertex coordinates with shape V + n_edges x 3
        - uvs ... np.array of uv coordinates with shape T + n_edges x 2
        - faces ... np.array of face vertex idcs with shape F + 4*F x 3
        - uv_faces ... np.array of face uv idcs with shape F + 4*F x 3
        - edges ... np.array of shape n_edges x 2 giving the indices of the vertices of each edge
        - uv_edges ... np.array of shape n_edges x 2 giving the indices of the uv_coords of each edge

        all returns are a concatenation like np.concatenate((array_old, array_new), axis=0) so that
        order of old entries is not changed and so that also old faces are still present.
    """
    n_faces = faces.shape[0]
    n_vertices = vertices.shape[0]
    n_uvs = uvs.shape[0]

    # if self.edges is None:
    # if True:
    # compute edges
    edges = []
    edge_map = dict()
    for i in range(0, n_faces):
        append_edge(edge_map, edges, faces[i, 0], faces[i, 1])
        append_edge(edge_map, edges, faces[i, 1], faces[i, 2])
        append_edge(edge_map, edges, faces[i, 2], faces[i, 0])
    n_edges = len(edges)
    edges = np.array(edges).astype(int)

    # compute edges uv space
    uv_edges = []
    uv_edge_map = dict()
    for i in range(0, n_faces):
        append_edge(uv_edge_map, uv_edges, uvfaces[i, 0], uvfaces[i, 1])
        append_edge(uv_edge_map, uv_edges, uvfaces[i, 1], uvfaces[i, 2])
        append_edge(uv_edge_map, uv_edges, uvfaces[i, 2], uvfaces[i, 0])
    uv_n_edges = len(uv_edges)
    uv_edges = np.array(uv_edges).astype(int)

    #    print('edges:', edges.shape)
    #    print('self.edge_map :', len(edge_map ))
    #
    #    print('uv_edges:', uv_edges.shape)
    #    print('self.uv_edge_map :', len(uv_edge_map ))
    #
    #    print('vertices:', vertices.shape)
    #    print('normals:', normals.shape)
    #    print('uvs:', uvs.shape)
    #    print('faces:', faces.shape)
    #    print('uvfaces:', uvfaces.shape)

    ############
    # vertices
    v = np.zeros((n_vertices + n_edges, 3))
    # copy original vertices
    v[:n_vertices, :] = vertices
    # compute edge midpoints
    vertices_edges = vertices[edges]
    v[n_vertices:, :] = 0.5 * (vertices_edges[:, 0] + vertices_edges[:, 1])

    # uvs
    f_uvs = np.zeros((n_uvs + uv_n_edges, 2))
    # copy original uvs
    f_uvs[:n_uvs, :] = uvs
    # compute edge midpoints
    uvs_edges = uvs[uv_edges]
    f_uvs[n_uvs:, :] = 0.5 * (uvs_edges[:, 0] + uvs_edges[:, 1])

    # new topology
    f = np.concatenate((faces, np.zeros((4 * n_faces, 3))), axis=0)
    f_uv_id = np.concatenate((uvfaces, np.zeros((4 * n_faces, 3))), axis=0)
    # f_uv = np.zeros((4*n_faces*3, 2))
    for i in range(0, n_faces):
        # vertex ids
        a = int(faces[i, 0])
        b = int(faces[i, 1])
        c = int(faces[i, 2])
        ab = n_vertices + edge_map[(a, b)]
        bc = n_vertices + edge_map[(b, c)]
        ca = n_vertices + edge_map[(c, a)]
        # uvs
        a_uv = int(uvfaces[i, 0])
        b_uv = int(uvfaces[i, 1])
        c_uv = int(uvfaces[i, 2])
        ab_uv = n_uvs + uv_edge_map[(a_uv, b_uv)]
        bc_uv = n_uvs + uv_edge_map[(b_uv, c_uv)]
        ca_uv = n_uvs + uv_edge_map[(c_uv, a_uv)]

        ## triangle 1
        f[n_faces + 4 * i, 0] = a
        f[n_faces + 4 * i, 1] = ab
        f[n_faces + 4 * i, 2] = ca
        f_uv_id[n_faces + 4 * i, 0] = a_uv
        f_uv_id[n_faces + 4 * i, 1] = ab_uv
        f_uv_id[n_faces + 4 * i, 2] = ca_uv

        ## triangle 2
        f[n_faces + 4 * i + 1, 0] = ab
        f[n_faces + 4 * i + 1, 1] = b
        f[n_faces + 4 * i + 1, 2] = bc
        f_uv_id[n_faces + 4 * i + 1, 0] = ab_uv
        f_uv_id[n_faces + 4 * i + 1, 1] = b_uv
        f_uv_id[n_faces + 4 * i + 1, 2] = bc_uv

        ## triangle 3
        f[n_faces + 4 * i + 2, 0] = ca
        f[n_faces + 4 * i + 2, 1] = ab
        f[n_faces + 4 * i + 2, 2] = bc
        f_uv_id[n_faces + 4 * i + 2, 0] = ca_uv
        f_uv_id[n_faces + 4 * i + 2, 1] = ab_uv
        f_uv_id[n_faces + 4 * i + 2, 2] = bc_uv

        ## triangle 4
        f[n_faces + 4 * i + 3, 0] = ca
        f[n_faces + 4 * i + 3, 1] = bc
        f[n_faces + 4 * i + 3, 2] = c
        f_uv_id[n_faces + 4 * i + 3, 0] = ca_uv
        f_uv_id[n_faces + 4 * i + 3, 1] = bc_uv
        f_uv_id[n_faces + 4 * i + 3, 2] = c_uv

    return v, f_uvs, f, f_uv_id, edges, uv_edges


"""
code heavily inspired from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/sample_points_from_meshes.html
just changed functionality such that only sampled face idcs and barycentric coordinates are returned. User 
has to do the rest 
"""


def face_vertices(vertices, faces):
    """
    :param vertices: [x size, number of vertices, 3]
    :param faces: [x size, number of faces, 3]
    :return: [x size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(
        0,
        faces[:, 1].long(),
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
        ),
    )
    normals.index_add_(
        0,
        faces[:, 2].long(),
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2],
            vertices_faces[:, 1] - vertices_faces[:, 2],
        ),
    )
    normals.index_add_(
        0,
        faces[:, 0].long(),
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
        ),
    )

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

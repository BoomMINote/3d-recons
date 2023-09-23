"""
Module to take the resulting 3D reconstructions and print it as a 3D file such as wavefront .obj or .stl
"""
import numpy as np
import os.path as osp

# import matplotlib.pyplot as plt
import PIL
from skimage import io, morphology, transform
import trimesh


base_color = np.array([200] * 3 + [255], dtype=np.uint8)


def _get_uv_coordinates(valid):
    shape = valid.shape
    y, x = np.mgrid[: shape[0], : shape[1]]
    y = 1 - y / shape[0]
    x = x / shape[1]
    return np.vstack((x[valid], y[valid])).T


# def _print_wavefront_with_texture(filename, valid, mesh, texture):
#     vertices = mesh.vertices
#     faces = mesh.faces
#     base, _ = osp.splitext(filename)
#     # shape = valid.shape
#     # y, x = np.mgrid[:shape[0], :shape[1]]
#     # y = 1 - y / shape[0]
#     # x = x / shape[1]
#     # uv = np.vstack((x[valid], y[valid])).T
#     uv = _get_uv_coordinates(valid)
#
#     vertice_str = ('v {} {} {}\n' * len(vertices))[:-1].format(*vertices.flatten())
#     uv = ('vt {} {}\n' * len(uv))[:-1].format(*uv.flatten())
#     faces_str = ('f {}/{} {}/{} {}/{}\n' * len(faces))[:-1].format(*np.repeat(faces.flatten() + 1, 2))
#
#     mtl_file = '{}.mtl'.format(base)
#     with open(filename, 'w') as f:
#         f.write('mtllib {}\n'.format(osp.basename(mtl_file)))
#         f.write('\n')
#         f.write(vertice_str)
#         f.write('\n')
#         f.write(uv)
#         f.write('\n\nusemtl Textured\n')
#         f.write(faces_str)
#
#     texture_file = "{}_texture.png".format(base);
#     io.imsave(texture_file, texture.astype(np.uint8));
#     with open(mtl_file, 'w') as f:
#         f.write("newmtl Textured\n")
#         f.write("Ka 1.000 1.000 1.000\n")
#         f.write("Kd 1.000 1.000 1.000\n")
#         f.write("Ks 0.100 0.100 0.100\n")
#         f.write("d 1.0\n")
#         f.write("illum 2\n")
#         f.write("map_Ka {}\n".format(osp.basename(texture_file)))
#         f.write("map_Kd {}\n".format(osp.basename(texture_file)))


def _valid_triangles(vertex_inds, valid, neighbourhood, is_empty=None):
    shape = valid.shape
    neighbourhood = np.asarray(neighbourhood)
    new_valid = morphology.erosion(valid, neighbourhood)
    new_valid[-1, :] = 0
    new_valid[:, -1] = 0
    if is_empty is not None:
        new_valid = new_valid & is_empty
    valid_inds = np.where(new_valid.flatten() == 1)[0]

    n = np.array(neighbourhood)[1:, 1:].astype(bool)
    first_vertex = 0 if n[0, 0] else 1
    second_vertex = shape[1] if n[1, 0] else shape[1] + 1
    third_vertex = shape[1] + 1 if n[1, 1] and second_vertex != shape[1] + 1 else 1

    triangles = np.vstack(
        (
            vertex_inds[valid_inds + first_vertex],
            vertex_inds[valid_inds + second_vertex],
            vertex_inds[valid_inds + third_vertex],
        )
    ).T
    return triangles, new_valid


max_vertices = 2**14


def mesh_it(
    reconstruction, out_3D_size=None, max_vertices=max_vertices
):  # , X, Y, threeD, valid=None, texture=None, name=None):
    grid = reconstruction["grid"]

    if len(grid.shape) == 3:
        X = grid[0]
        Y = grid[1]
        Z = grid[2]
    else:
        shape = grid.shape
        Y, X = np.mgrid[: shape[0], : shape[1]]
        X = X - shape[1] / 2
        Y = -(Y - shape[0] / 2)
        Z = grid

    mask = Z.mask if np.ma.isMaskedArray(Z) else np.isnan(Z)
    transform_factor = int(np.ceil(np.sqrt(np.sum(~mask) / max_vertices)))
    if transform_factor >= 1:
        X = transform.downscale_local_mean(X, (transform_factor, transform_factor))
        Y = transform.downscale_local_mean(Y, (transform_factor, transform_factor))
        Z = transform.downscale_local_mean(Z, (transform_factor, transform_factor))
        mask = transform.downscale_local_mean(
            mask.astype(float), (transform_factor, transform_factor)
        )

    mask = mask > 0
    shape = Z.shape
    valid = (~mask).astype(bool)
    # Can for some reason not handle values on the edge of image
    valid[[0, -1], :] = valid[:, [0, -1]] = False

    if len(grid.shape) == 2 or out_3D_size is not None:
        general_scale = 2 * out_3D_size / (np.max(shape) - 1)
        X *= general_scale / transform_factor
        Y *= general_scale / transform_factor
        Z *= general_scale / transform_factor

    vertices = np.vstack((X[valid], Y[valid], Z[valid])).T
    vertex_inds = np.cumsum(valid) - 1

    upper_right_triangles, upper_right_valid = _valid_triangles(
        vertex_inds, valid.astype(int), [[0, 0, 0], [0, 1, 1], [0, 0, 1]]
    )
    lower_left_triangles, lower_left_valid = _valid_triangles(
        vertex_inds, valid.astype(int), [[0, 0, 0], [0, 1, 0], [0, 1, 1]]
    )
    is_empty = (upper_right_valid == 0) & (lower_left_valid == 0)
    upper_left_triangles, _ = _valid_triangles(
        vertex_inds, valid.astype(int), [[0, 0, 0], [0, 1, 1], [0, 1, 0]], is_empty
    )
    lower_right_triangles, _ = _valid_triangles(
        vertex_inds, valid.astype(int), [[0, 0, 0], [0, 0, 1], [0, 1, 1]], is_empty
    )

    faces = np.vstack(
        (
            upper_right_triangles,
            lower_left_triangles,
            upper_left_triangles,
            lower_right_triangles,
        )
    )

    pImage = None
    if "texture" in reconstruction and reconstruction["texture"] is not None:
        pImage = PIL.Image.fromarray(reconstruction["texture"].astype(np.uint8))
    elif (
        "segmentation" in reconstruction and reconstruction["segmentation"] is not None
    ):
        interpolate_pixels = (reconstruction["segmentation"] != 0) & (
            reconstruction["segmentation"] != 255
        )
        if np.any(interpolate_pixels):
            # coloring pixels that have been interpolated differently
            texture = np.ones(shape + (3,)) * 255
            texture[interpolate_pixels] = [0, 255, 0]
            pImage = PIL.Image.fromarray(texture.astype(np.uint8))

    mesh = trimesh.Trimesh(vertices, faces, process=False)

    uv = _get_uv_coordinates(valid)
    material = trimesh.visual.material.PBRMaterial(
        name="Material",
        baseColorFactor=base_color,
        baseColorTexture=pImage,
        roughnessFactor=1.0,
        metallicFactor=0.0,
        doubleSided=True,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv, material)

    if "name" in reconstruction:
        mesh.metadata["name"] = reconstruction["name"]
    return mesh


# def print_it(filename, grid, texture=None, out_3D_size=None, max_vertices=2**14):
#     """
#     Function to print a 3D array as a 3D file mesh. If it has more vertices than downsampled size it will be downscaled to appropriate size.
#     Check trimesh library for which formats are supported
#     """
#
#     if len(grid.shape) == 3:
#         X = grid[0]
#         Y = grid[1]
#         Z = grid[2]
#     else:
#         shape = grid.shape
#         Y, X = np.mgrid[:shape[0], :shape[1]]
#         X =  (X - shape[1] / 2)
#         Y = -(Y - shape[0] / 2)
#         Z = grid
#
#     mask = Z.mask if np.ma.isMaskedArray(Z) else np.zeros(Z.shape)
#     transform_factor = int(np.ceil(np.sqrt(np.sum(~mask) / max_vertices)))
#     if transform_factor >= 1:
#         X = transform.downscale_local_mean(X, (transform_factor, transform_factor))
#         Y = transform.downscale_local_mean(Y, (transform_factor, transform_factor))
#         Z = transform.downscale_local_mean(Z, (transform_factor, transform_factor))
#         mask = transform.downscale_local_mean(mask.astype(float), (transform_factor, transform_factor))
#
#     mask = mask > 0
#     shape = Z.shape
#     valid = (~mask).astype(bool)
#
#     if len(grid.shape) == 2 or not out_3D_size is None:
#         general_scale = 2 * out_3D_size / (np.max(shape) - 1)
#         X *= general_scale / transform_factor
#         Y *= general_scale / transform_factor
#         Z *= general_scale / transform_factor
#
#     mesh = mesh_it(X, Y, Z, valid, texture, osp.splitext(osp.basename(filename))[0])
#
#     ext = osp.splitext(filename)[1]
#     if not texture is None and ext == '.obj':  # Special export for obj files with texture
#         _print_wavefront_with_texture(filename, valid, mesh, texture)
#     else:
#         mesh.export(filename, include_normals=True)


def csv_to_3D(reconstructed_file, ext):
    """Function not really used but it shows how a printed csv file can be converted to 3D file"""
    reconstructed_file_base, _ = osp.splitext(reconstructed_file)
    output_dir, file_name = osp.split(reconstructed_file)
    # prefix = 'reconstructed_'

    grid3d = np.loadtxt(reconstructed_file)
    grid3d = grid3d.reshape((3, int(grid3d.shape[0] / 3), grid3d.shape[1]))
    grid_mask = np.isnan(grid3d)
    grid3d[grid_mask] = 0
    grid3d = np.ma.array(grid3d, mask=grid_mask)

    texture_file = osp.join(reconstructed_file_base + "_texture.png")
    texture = None
    if osp.isfile(texture_file):
        texture = io.imread(texture_file)

    export3D(
        "{}.{}".format(reconstructed_file_base, ext),
        {"grid": grid3d, "texture": texture},
    )


def export3D(
    filename, reconstructions, out_3D_size=None, max_vertices=max_vertices, prefix=""
):
    """
    Function to export a 3D reconstruction with this package to a file
    Possible 3D formats are the same as for trimesh such as GL Transmission Format .glb
    The directory of filename is used when not glb files are exported
    Prefix is only used for single file printing

    Textures are only supported for .glb files
    reconstructions should either be a dict or list of dicts where each dict contains grid (3D coordinates in a 3xNxM matrix) optional texture and optional name of the mesh
    """

    if isinstance(reconstructions, dict):
        meshes = [mesh_it(reconstructions, out_3D_size, max_vertices)]
    elif isinstance(reconstructions, list):
        meshes = []
        for reconstruction in reconstructions:
            meshes += [mesh_it(reconstruction, out_3D_size, max_vertices)]
    else:
        raise ValueError(
            'Reconstruction input format must be dict with "grid" key or list of said dicts'
        )

    base, ext = osp.splitext(filename)
    if ext == ".glb":
        # write all meshes together
        scene = trimesh.scene.scene.Scene()
        for mesh in meshes:
            name = None
            if "name" in mesh.metadata:
                name = mesh.metadata["name"]
            scene.add_geometry(mesh, node_name=name)
        scene.export(str(filename), include_normals=True)
    else:
        # write one by one
        if len(meshes) == 1:
            meshes[0].export(filename)
        else:
            for mesh in meshes:
                name = ""
                if "name" in mesh.metadata:
                    name = mesh.metadata["name"]
                output_file = "{}{}".format(mesh.metadata["name"], ext)
                mesh.export(output_file)

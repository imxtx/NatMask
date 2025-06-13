# Based on https://github.com/yfeng95/PRNet/blob/master/utils/render.py
import os
import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from skimage.io import imsave
from torch import Tensor
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    BlendParams,
)


def create_renderer(device: str) -> MeshRenderer:
    """Create a renderer for rendering the face mesh"""
    # create renderer
    # ref: https://pytorch3d.org/tutorials/render_textured_meshes
    R, T = look_at_view_transform(at=((127.5, 127.5, 0),), eye=((127.5, 127.5, 200),))

    cameras = FoVOrthographicCameras(
        znear=200,
        zfar=0,
        min_x=-127.5,
        max_x=127.5,
        min_y=-127.5,
        max_y=127.5,
        device=device,
        R=R,
        T=T,
    )

    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = DirectionalLights(
        device=device,
        direction=((0, 0, -1),),
        ambient_color=((1, 1, 1),),
        diffuse_color=((0, 0, 0),),
        specular_color=((0, 0, 0),),
    )
    blend_params = BlendParams(background_color=[0, 0, 0])

    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params,
        ),
    )


def render(
    renderer: MeshRenderer,
    textures: Tensor,
    verts: Tensor,
    verts_uvs: Tensor,
    triangles: Tensor,
) -> Tensor:
    """Render with PyTorch3d
    Args:
        renderer (MeshRenderer): Renderer
        textures (Tensor): (N, H, W, C) Face texture
        verts (Tensor): (N, nVerts, 3) Verticies
        verts_uvs (Tensor): (nVerts, 2) UV coordinates for verticies
        triangles (Tensor): (nFaces, 3) Faces / Faces uvs
    Returns:
        Tensor: Rendered images
    """
    N, H, W, C = textures.shape
    nFaces, _ = triangles.shape
    nVerts, _ = verts_uvs.shape

    # create texture
    mask_texs = TexturesUV(
        maps=textures,
        faces_uvs=triangles.expand(N, nFaces, 3),
        verts_uvs=verts_uvs.expand(N, nVerts, 2),
    )

    # verts: (n, n_ver x 3), triangles: (n_face x 3)
    mask_meshes = Meshes(
        verts=verts,
        faces=triangles.expand(N, nFaces, 3),
        textures=mask_texs,
    )

    return renderer(mask_meshes)


def get_uv_textures_batch(face_images: NDArray, poses: NDArray) -> NDArray:
    """
    Args:
        face_images (NDArray): (N, H, W, 3) Face images [0, 255]
        poses (NDArray): (N, H, W, 3) UV position maps
    Returns:
        NDArray: (N, H, W, 3) UV texture maps
    """
    textures = []
    for i in range(face_images.shape[0]):
        tex = cv2.remap(
            face_images[i],
            poses[i, :, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0),
        )
        textures.append(tex)
    return np.stack(textures)


def get_uv_texture(face_image: NDArray, pos: NDArray) -> NDArray:
    """
    Args:
        face_image (NDArray): (H, W, 3) Face images [0, 1]
        pos (NDArray): (H, W, 3) UV position maps
    Returns:
        NDArray: (H, W, 3) UV texture maps
    """
    return cv2.remap(
        face_image,
        pos[:, :, :2].astype(np.float32),
        None,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0),
    )


def render_texture(
    vertices: Tensor,
    colors: Tensor,
    triangles: Tensor,
    h: int,
    w: int,
    c: int = 3,
    device: str = "cpu",
) -> Tensor:
    """render mesh by z buffer
    Args:
        vertices: 3 x nver
        colors: 3 x nver
        triangles: 3 x ntri
        h: height
        w: width
    """
    # initial
    image = torch.zeros((h, w, c)).to(device)

    depth_buffer = torch.zeros([h, w]) - 999999.0
    # triangle depth: approximate the depth to the average value of z in each
    # vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (
        vertices[2, triangles[0, :]]
        + vertices[2, triangles[1, :]]
        + vertices[2, triangles[2, :]]
    ) / 3.0
    tri_tex = (
        colors[:, triangles[0, :]]
        + colors[:, triangles[1, :]]
        + colors[:, triangles[2, :]]
    ) / 3.0

    for i in range(triangles.shape[1]):
        tri = triangles[:, i]  # 3 vertex indices

        # the inner bounding box
        umin = max(int(torch.ceil(torch.min(vertices[0, tri]))), 0)
        umax = min(int(torch.floor(torch.max(vertices[0, tri]))), w - 1)

        vmin = max(int(torch.ceil(torch.min(vertices[1, tri]))), 0)
        vmax = min(int(torch.floor(torch.max(vertices[1, tri]))), h - 1)

        if umax < umin or vmax < vmin:
            continue

        for u in range(umin, umax + 1):
            for v in range(vmin, vmax + 1):
                if tri_depth[i] > depth_buffer[v, u] and isPointInTriangle(
                    [u, v], vertices[:2, tri].cpu().numpy()
                ):
                    depth_buffer[v, u] = tri_depth[i]
                    image[v, u, :] = tri_tex[:, i]
    return image


def isPointInTriangle(point: NDArray, tri_points: NDArray) -> NDArray:
    """Check if a point is in a triangle"""
    tp = tri_points

    # vectors
    v0 = tp[:, 2] - tp[:, 0]
    v1 = tp[:, 1] - tp[:, 0]
    v2 = point - tp[:, 0]

    # dot products
    dot00 = np.matmul(v0.T, v0)
    dot01 = np.matmul(v0.T, v1)
    dot02 = np.matmul(v0.T, v2)
    dot11 = np.matmul(v1.T, v1)
    dot12 = np.matmul(v1.T, v2)

    # barycentric coordinates
    if dot00 * dot11 - dot01 * dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)


def render_texture_numpy(
    vertices: NDArray,
    colors: NDArray,
    triangles: NDArray,
    h: int,
    w: int,
    c: int = 3,
) -> NDArray:
    """render mesh by z buffer
    Args:
        vertices: 3 x nver
        colors: 3 x nver
        triangles: 3 x ntri
        h: height
        w: width
    """
    # initial
    image = np.zeros((h, w, c))

    depth_buffer = np.zeros([h, w]) - 999999.0
    # triangle depth: approximate the depth to the average value of z in each
    # vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (
        vertices[2, triangles[0, :]]
        + vertices[2, triangles[1, :]]
        + vertices[2, triangles[2, :]]
    ) / 3.0
    tri_tex = (
        colors[:, triangles[0, :]]
        + colors[:, triangles[1, :]]
        + colors[:, triangles[2, :]]
    ) / 3.0

    for i in range(triangles.shape[1]):
        tri = triangles[:, i]  # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[0, tri]))), 0)
        umax = min(int(np.floor(np.max(vertices[0, tri]))), w - 1)

        vmin = max(int(np.ceil(np.min(vertices[1, tri]))), 0)
        vmax = min(int(np.floor(np.max(vertices[1, tri]))), h - 1)

        if umax < umin or vmax < vmin:
            continue

        for u in range(umin, umax + 1):
            for v in range(vmin, vmax + 1):
                if tri_depth[i] > depth_buffer[v, u] and isPointInTriangle(
                    [u, v], vertices[:2, tri]
                ):
                    depth_buffer[v, u] = tri_depth[i]
                    image[v, u, :] = tri_tex[:, i]
    return image


def get_depth_image(
    vertices: NDArray,
    triangles: NDArray,
    h: int,
    w: int,
    isShow: bool = False,
) -> NDArray:
    """Get the depth image of the face mesh"""
    z = vertices[:, 2:]
    if isShow:
        z = z / max(z)
    depth_image = render_texture_numpy(vertices.T, z.T, triangles.T, h, w, 1)
    return np.squeeze(depth_image)


def write_obj_with_colors(
    obj_name: str,
    vertices: NDArray,
    triangles: NDArray,
    colors: NDArray,
) -> None:
    """Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    """
    triangles = triangles.copy()
    triangles += 1  # meshlab start with 1

    if obj_name.split(".")[-1] != "obj":
        obj_name = obj_name + ".obj"

    # write obj
    with open(obj_name, "w") as f:

        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = "v {} {} {} {} {} {}\n".format(
                vertices[i, 0],
                vertices[i, 1],
                vertices[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            s = "f {} {} {}\n".format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)


def write_obj_with_texture(
    obj_name: str,
    vertices: NDArray,
    triangles: NDArray,
    texture: NDArray,
    uv_coords: NDArray,
) -> None:
    """Save 3D face model with texture represented by texture map.
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        texture: shape = (256,256,3)
        uv_coords: shape = (nver, 3) max value<=1
    """
    if obj_name.split(".")[-1] != "obj":
        obj_name = obj_name + ".obj"
    mtl_name = obj_name.replace(".obj", ".mtl")
    texture_name = obj_name.replace(".obj", "_texture.png")

    triangles = triangles.copy()
    triangles += 1  # mesh lab start with 1

    # write obj
    with open(obj_name, "w") as f:
        # first line: write mtlib(material library)
        s = "mtllib {}\n".format(os.path.abspath(mtl_name))
        f.write(s)

        # write vertices
        for i in range(vertices.shape[0]):
            s = "v {} {} {}\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            f.write(s)

        # write uv coords
        for i in range(uv_coords.shape[0]):
            s = "vt {} {}\n".format(uv_coords[i, 0], 1 - uv_coords[i, 1])
            f.write(s)

        f.write("usemtl FaceTexture\n")

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[0]):
            # s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[i,0], triangles[i,0], triangles[i,1], triangles[i,1], triangles[i,2], triangles[i,2])
            s = "f {}/{} {}/{} {}/{}\n".format(
                triangles[i, 2],
                triangles[i, 2],
                triangles[i, 1],
                triangles[i, 1],
                triangles[i, 0],
                triangles[i, 0],
            )
            f.write(s)

    # write mtl
    with open(mtl_name, "w") as f:
        f.write("newmtl FaceTexture\n")
        s = "map_Kd {}\n".format(os.path.abspath(texture_name))  # map to image
        f.write(s)

    # write texture as png
    imsave(texture_name, texture)


def write_obj_with_colors_texture(
    obj_name: str,
    vertices: NDArray,
    colors: NDArray,
    triangles: NDArray,
    texture: NDArray,
    uv_coords: NDArray,
) -> None:
    """Save 3D face model with texture.
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        texture: shape = (256,256,3)
        uv_coords: shape = (nver, 3) max value<=1
    """
    if obj_name.split(".")[-1] != "obj":
        obj_name = obj_name + ".obj"
    mtl_name = obj_name.replace(".obj", ".mtl")
    texture_name = obj_name.replace(".obj", "_texture.png")

    triangles = triangles.copy()
    triangles += 1  # mesh lab start with 1

    # write obj
    with open(obj_name, "w") as f:
        # first line: write mtlib(material library)
        s = "mtllib {}\n".format(os.path.abspath(mtl_name))
        f.write(s)

        # write vertices
        for i in range(vertices.shape[0]):
            s = "v {} {} {} {} {} {}\n".format(
                vertices[i, 0],
                vertices[i, 1],
                vertices[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
            f.write(s)

        # write uv coords
        for i in range(uv_coords.shape[0]):
            s = "vt {} {}\n".format(uv_coords[i, 0], 1 - uv_coords[i, 1])
            f.write(s)

        f.write("usemtl FaceTexture\n")

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[0]):
            # s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[i,0], triangles[i,0], triangles[i,1], triangles[i,1], triangles[i,2], triangles[i,2])
            s = "f {}/{} {}/{} {}/{}\n".format(
                triangles[i, 2],
                triangles[i, 2],
                triangles[i, 1],
                triangles[i, 1],
                triangles[i, 0],
                triangles[i, 0],
            )
            f.write(s)

    # write mtl
    with open(mtl_name, "w") as f:
        f.write("newmtl FaceTexture\n")
        s = "map_Kd {}\n".format(os.path.abspath(texture_name))  # map to image
        f.write(s)

    # write texture as png
    imsave(texture_name, texture)


def norm_pos_map(pos_maps: NDArray) -> NDArray:
    """Normalize the position map"""
    pos_maps = pos_maps.reshape(-1, 256 * 256, 3)
    norm_pos_maps = np.zeros_like(pos_maps)
    N, _, _ = pos_maps.shape
    for i in range(N):
        pos_map = pos_maps[i]
        max_v = np.amax(pos_map, axis=0)
        min_v = np.amin(pos_map, axis=0)
        norm_pos_maps[i] = (pos_map - min_v) / (max_v - min_v) * 255
    return norm_pos_maps.reshape(-1, 256, 256, 3)

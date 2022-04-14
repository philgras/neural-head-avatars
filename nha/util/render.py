from pytorch3d.renderer import (
    TexturesVertex,
    DirectionalLights,
    MeshRenderer,
    HardPhongShader,
)

from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    BlendParams
)
import torch


def hard_feature_blend(features, fragments, blend_params) -> torch.Tensor:
    """
    Naive blending of top K faces to return a feature image
      - **D** - choose features of the closest point i.e. K=0
      - **A** - is_background --> pix_to_face == -1

    Args:
        features: (N, H, W, K, D) features for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image. This is used to
              determine the output shape.
        blend_params: BlendParams instance that contains a background_color
        field specifying the color for the background
    Returns:
        RGBA pixel_colors: (N, H, W, 4)
    """
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device

    # Mask for the background.
    is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)

    if torch.is_tensor(blend_params.background_color):
        background_color = blend_params.background_color.to(device)
    else:
        background_color = features.new_tensor(blend_params.background_color)  # (3)

    # Find out how much background_color needs to be expanded to be used for masked_scatter.
    num_background_pixels = is_background.sum()

    # Set background color.
    pixel_features = features[..., 0, :].masked_scatter(
        is_background[..., None],
        background_color[None, :].expand(num_background_pixels, -1),
    )  # (N, H, W, 3)

    # Concat with the alpha channel.
    alpha = (~is_background).float()[..., None]
    return torch.cat([pixel_features, alpha], dim=-1)  # (N, H, W, 4)


def create_intrinsics_matrix(fx, fy, px, py):
    return torch.tensor([[fx, 0, px], [0, fy, py], [0, 0, 1.0]]).float()


def normalize_image_points(u, v, resolution):
    """
    normalizes u, v coordinates from [0 ,image_size] to [-1, 1]
    :param u:
    :param v:
    :param resolution:
    :return:
    """
    u = 2 * (u - resolution[1] / 2.0) / resolution[1]
    v = 2 * (v - resolution[0] / 2.0) / resolution[0]
    return u, v


def create_camera_objects(K, RT, resolution, device):
    """
    Create pytorch3D camera objects from camera parameters
    :param K:
    :param RT:
    :param resolution:
    :return:
    """
    R = RT[:, :, :3]
    T = RT[:, :, 3]
    H, W = resolution
    img_size = torch.tensor([[H, W]] * len(K), dtype=torch.int, device=device)
    f = torch.stack((K[:, 0, 0], K[:, 1, 1]), dim=-1)
    principal_point = torch.cat([K[:, [0], -1], H - K[:, [1], -1]], dim=1)
    cameras = PerspectiveCameras(
        R=R,
        T=T,
        principal_point=principal_point,
        focal_length=f,
        device=device,
        image_size=img_size,
        in_ndc=False,
    )
    return cameras


def batch_project(points, K, RT, image_size, device, normalize=True):
    cameras = create_camera_objects(K, RT, image_size, device)

    proj_points= cameras.transform_points_screen(points)
    if normalize:
        proj_points[:, :, 0], proj_points[:, :, 1] = normalize_image_points(
            proj_points[:, :, 0], proj_points[:, :, 1], image_size
        )
    return proj_points


def render_shaded_mesh(mesh, K, RT, image_size, device, light_colors=(0.4, 0.6, 0.3)):
        cameras = create_camera_objects(K, RT, image_size, device)

        lights = DirectionalLights(
            ambient_color=[[light_colors[0]] * 3],
            diffuse_color=[[light_colors[1]] * 3],
            specular_color=[[light_colors[2]] * 3],
            direction=[[0, 1, -1]],
            device=device,
        )
        # define renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras,
                raster_settings=RasterizationSettings(cull_backfaces=True, image_size=image_size),
            ),
            shader=HardPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(sigma=0, gamma=0),
            ),
        )

        return renderer(mesh).clamp(0.0, 1.0).permute(0, 3, 1, 2)




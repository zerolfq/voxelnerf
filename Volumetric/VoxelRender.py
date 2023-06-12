import numpy as np
import torch
from pytorch3d.structures import Volumes
from rich import print
from packaging import version as pver

need_pytorch3d = False
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d = True
if need_pytorch3d:
    print("IO")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


class VoxelRender:
    def __init__(self, VoxelModel):
        self.Volume = VoxelModel
        # bound = np.array(
        #     [self.Volume.densities().shape[2], self.Volume.densities().shape[3], self.Volume.densities().shape[4]])
        # self.Bound = torch.tensor(bound, dtype=torch.float)
        # self.Bound *= self.Volume.locator.voxel_size

    def Render(self, xyzs):  # xyzs[N,3]
        volumes_densities = self.Volume.densities()
        dim_density = volumes_densities.shape[1]
        volumes_features = self.Volume.features()
        nums = xyzs.shape[0]
        xyzs = xyzs.view(1, nums, 3)

        rays_points_local = self.Volume.world_to_local_coords(xyzs)  # batch,N,3
        min_bound = torch.tensor([[[-1.0, -1.0, -1.0]]], device=device)
        max_bound = torch.tensor([[[1.0, 1.0, 1.0]]], device=device)
        IsInBound = torch.all(rays_points_local >= min_bound, dim=2) & torch.all(rays_points_local <= max_bound, dim=2)
        IsInBound.to(device)
        IsInBound = IsInBound.view(*IsInBound.shape, 1)
        rays_points_local_flat = rays_points_local.view(
            rays_points_local.shape[0], -1, 1, 1, 3
        )

        rays_densities = torch.nn.functional.grid_sample(
            volumes_densities,
            rays_points_local_flat,
            align_corners=True,
            mode='bilinear',
        )

        rays_densities = rays_densities.permute(0, 2, 3, 4, 1).view(
            *rays_points_local.shape[:-1], volumes_densities.shape[1]
        )
        rays_densities = rays_densities * IsInBound

        min_threshold = torch.tensor([[[0.1]]], device=device)
        max_threshold = torch.tensor([[[1.0]]], device=device)
        densities_threshold = torch.all(rays_densities >= min_threshold, dim=2) & torch.all(
            rays_densities <= max_threshold, dim=2)
        densities_threshold = densities_threshold.view(*densities_threshold.shape, 1)
        rays_densities = rays_densities * densities_threshold

        # print('rays_densities', (rays_densities * IsInBound))
        if volumes_features is None:
            dim_feature = 0
            _, rays_features = rays_densities.split([dim_density, dim_feature], dim=-1)
        else:
            rays_features = torch.nn.functional.grid_sample(
                volumes_features,
                rays_points_local_flat,
                align_corners=True,
                mode='bilinear',
            )

            # permute the dimensions & reshape features after sampling
            rays_features = rays_features.permute(0, 2, 3, 4, 1).view(
                *rays_points_local.shape[:-1], volumes_features.shape[1]
            )
        rays_features = rays_features * IsInBound

        rays_features = rays_features.view(rays_features.shape[1], rays_features.shape[-1])
        rays_densities = rays_densities.view(rays_densities.shape[1])
        return rays_densities, rays_features


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


#
#
# if __name__ == '__main__':
#     densities = torch.arange(8 * 8 * 8, dtype=torch.float, device=device)
#     densities = densities.view(1, 1, 8, 8, 8)
#     densities -= 15.0
#     features = torch.ones(8 * 8 * 8 * 3, dtype=torch.float, device=device)
#     features = features.view(1, 3, 8, 8, 8)
#     volumes = Volumes(
#         densities=densities,
#         features=features,
#         voxel_size=1
#     )
#     # print('densities', densities)
#
#     X = torch.arange(4, dtype=torch.int32, device=device).split(4)
#     Y = torch.arange(4, dtype=torch.int32, device=device).split(4)
#     Z = torch.arange(4, dtype=torch.int32, device=device).split(4)
#
#     for xs in X:
#         for ys in Y:
#             for zs in Z:
#                 xx, yy, zz = custom_meshgrid(xs, ys, zs)
#
#                 coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
#                 xyzs = coords + torch.zeros_like(coords, dtype=torch.float, device=device)
#                 # xyzs.device = device
#                 # xyzs.dtype = torch.float
#                 _VoxelRender = VoxelRender(volumes)
#                 rays_densities, rays_features = _VoxelRender.Render(xyzs)
#
#     # mesh2Voxel = Mesh2Voxel(num_view=70, IsShow=False)
#     # mesh2Voxel.train(15, 500)
#     # mesh2Voxel.SetTrainModel(volume_translation=(0, 0, 0), volume_rotator=(0, 90, 0))
#     # rotating_volume_frames = Volumetric.generate_rotating_volume(mesh2Voxel, n_frames=7 * 4,
#     #                                                              volumes=mesh2Voxel.ResultVoxel)

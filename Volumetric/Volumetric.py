import os
from typing import List, Optional, Tuple, Union

from rich import print
from torch.nn import Parameter

need_pytorch3d = False
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d = True
if need_pytorch3d:
    print("IO")

import torch
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from IPython import display

# Data structures and functions for rendering
from pytorch3d.structures import Volumes  # 体素表达
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher
)
from pytorch3d.transforms import so3_exp_map

_Scalar = Union[int, float]
_Vector = Union[torch.Tensor, Tuple[_Scalar, ...], List[_Scalar]]
_ScalarOrVector = Union[_Scalar, _Vector]

_VoxelSize = _ScalarOrVector
_Translation = _Vector
_Rotator = _Vector
_Scale = _Vector

# obtain the utilized device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# In[ ]:


from Volumetric.plot_image_grid import image_grid
from Volumetric.generate_cow_renders import generate_cow_renders
# from plot_image_grid import image_grid
# from generate_cow_renders import generate_cow_renders


def huber(x, y, scaling=0.1):
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling ** 2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss


class VolumeModel(torch.nn.Module):
    log_colors: Parameter

    def __init__(self, renderer, volume_size=[64] * 3, voxel_size=0.1):
        super().__init__()
        # After evaluating torch.sigmoid(self.log_colors), we get
        # densities close to zero.
        self.log_densities = torch.nn.Parameter(-4.0 * torch.ones(1, *volume_size))
        # print(*volume_size)
        # print(self.log_densities.shape)
        # After evaluating torch.sigmoid(self.log_colors), we get
        # a neutral gray color everywhere.
        self.log_colors = torch.nn.Parameter(torch.zeros(3, *volume_size))
        self._voxel_size = voxel_size
        # Store the renderer module as well.
        self._renderer = renderer

    def forward(self, cameras):
        batch_size = cameras.R.shape[0]

        # Convert the log-space values to the densities/colors
        densities = torch.sigmoid(self.log_densities)
        colors = torch.sigmoid(self.log_colors)

        # Instantiate the Volumes object, making sure
        # the densities and colors are correctly
        # expanded batch_size-times.
        volumes = Volumes(
            densities=densities[None].expand(
                batch_size, *self.log_densities.shape),
            features=colors[None].expand(
                batch_size, *self.log_colors.shape),
            voxel_size=self._voxel_size,
        )

        # Given cameras and volumes, run the renderer
        # and return only the first output value
        # (the 2nd output is a representation of the sampled
        # rays which can be omitted for our purpose).
        return self._renderer(cameras=cameras, volumes=volumes)[0]

    @property
    def voxel_size(self):
        return self._voxel_size


class Mesh2Voxel:
    def __init__(self, num_view=40, volumeModel=128, volume_size=128, lr=0.1, IsShow=True, extent_world=3.0):
        self.target_cameras, self.target_images, self.target_silhouettes = generate_cow_renders(num_views=num_view)
        print(f'Generated {len(self.target_images)} images/silhouettes/cameras.')
        render_size = self.target_images.shape[1]
        self.volume_extent_world = extent_world

        self.raysampler = NDCMultinomialRaysampler(
            image_width=render_size,
            image_height=render_size,
            n_pts_per_ray=150,
            min_depth=0.1,
            max_depth=self.volume_extent_world,
        )
        self.raymarcher = EmissionAbsorptionRaymarcher()

        self.renderer = VolumeRenderer(
            raysampler=self.raysampler, raymarcher=self.raymarcher
        )

        self.target_cameras = self.target_cameras.to(device)
        self.target_images = self.target_images.to(device)
        self.target_silhouettes = self.target_silhouettes.to(device)

        self.volumeModel = volumeModel
        self.volume_size = volume_size

        self.volume_model = VolumeModel(
            self.renderer,
            volume_size=[self.volume_size] * 3,
            voxel_size=self.volume_extent_world / self.volume_size,
        ).to(device)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.volume_model.parameters(), lr=self.lr)

        self.ResultVoxel = None
        self.IsShow = IsShow
        self.TransformWorld = torch.zeros(4, 4)

    def train(self, batch_size=10, n_iter=1000):
        if self.LoadModel(f'./data/model', f'voxel.pth'):
            return

        for iteration in range(n_iter):
            # In case we reached the last 75% of iterations,
            # decrease the learning rate of the optimizer 10-fold.
            if iteration == round(n_iter * 0.75):
                print('Decreasing LR 10-fold ...')
                self.optimizer = torch.optim.Adam(
                    self.volume_model.parameters(), lr=self.lr * 0.1
                )

            # Zero the optimizer gradient.
            self.optimizer.zero_grad()

            # Sample random batch indices.
            batch_idx = torch.randperm(len(self.target_cameras))[:batch_size]

            # Sample the minibatch of cameras.
            batch_cameras = FoVPerspectiveCameras(
                R=self.target_cameras.R[batch_idx],
                T=self.target_cameras.T[batch_idx],
                znear=self.target_cameras.znear[batch_idx],
                zfar=self.target_cameras.zfar[batch_idx],
                aspect_ratio=self.target_cameras.aspect_ratio[batch_idx],
                fov=self.target_cameras.fov[batch_idx],
                device=device,
            )

            # Evaluate the volumetric model.
            rendered_images, rendered_silhouettes = self.volume_model(
                batch_cameras
            ).split([3, 1], dim=-1)

            # Compute the silhouette error as the mean huber
            # loss between the predicted masks and the
            # target silhouettes.
            sil_err = huber(
                rendered_silhouettes[..., 0], self.target_silhouettes[batch_idx],
            ).abs().mean()

            # Compute the color error as the mean huber
            # loss between the rendered colors and the
            # target ground truth images.
            color_err = huber(
                rendered_images, self.target_images[batch_idx],
            ).abs().mean()

            # The optimization loss is a simple
            # sum of the color and silhouette errors.
            loss = color_err + sil_err

            # Print the current values of the losses.
            if iteration % 10 == 0:
                print(
                    f'Iteration {iteration:05d}:'
                    + f' color_err = {float(color_err):1.2e}'
                    + f' mask_err = {float(sil_err):1.2e}'
                )

            # Take the optimization step.
            loss.backward()
            self.optimizer.step()

            # Visualize the renders every 40 iterations.
            if iteration % 40 == 0 and self.IsShow:
                # Visualize only a single randomly selected element of the batch.
                im_show_idx = int(torch.randint(low=0, high=batch_size, size=(1,)))
                fig, ax = plt.subplots(2, 2, figsize=(10, 10))
                ax = ax.ravel()
                clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
                ax[0].imshow(clamp_and_detach(rendered_images[im_show_idx]))
                ax[1].imshow(clamp_and_detach(self.target_images[batch_idx[im_show_idx], ..., :3]))
                ax[2].imshow(clamp_and_detach(rendered_silhouettes[im_show_idx, ..., 0]))
                ax[3].imshow(clamp_and_detach(self.target_silhouettes[batch_idx[im_show_idx]]))
                for ax_, title_ in zip(
                        ax,
                        ("rendered image", "target image", "rendered silhouette", "target silhouette")
                ):
                    ax_.grid("off")
                    ax_.axis("off")
                    ax_.set_title(title_)
                fig.canvas.draw()
                fig.show()
                display.clear_output(wait=True)
                display.display(fig)
        self.SaveModel(f'./data/model', f'voxel.pth')

    def SaveModel(self, filepath, filename):
        if os.path.isdir(filepath):
            pass
        else:
            os.makedirs(filepath)

        filename = os.path.join(filepath, filename)

        torch.save({'model': self.volume_model.state_dict()}, filename)

    def LoadModel(self, filepath, filename):
        filename = os.path.join(filepath, filename)
        if not os.path.exists(filepath):
            return False
        state_dict = torch.load(filename)
        self.volume_model.load_state_dict(state_dict['model'])
        return True

    def SetTrainModel(self,
                      volume_translation: _Translation = (0.0, 0.0, 0.0),
                      volume_rotator: _Rotator = (0, 0, 0),
                      volume_scale: _Scale = (1.0, 1.0, 1.0)):
        with torch.no_grad():
            densities = torch.sigmoid(self.volume_model.log_densities)  # 获取模型训练完毕的数据
            colors = torch.sigmoid(self.volume_model.log_colors)
            batch_size = 1

            densities = densities[None].expand(
                batch_size, *self.volume_model.log_densities.shape)
            print(colors.shape)
            features = colors[None].expand(
                batch_size, *self.volume_model.log_colors.shape)
            print('features_min', features.min(), 'features_max', features.max())
            print('densities_min', densities.min(), 'densities_max', densities.max())
            self.ResultVoxel = Volumes(
                densities=densities,
                features=colors[None].expand(
                    batch_size, *self.volume_model.log_colors.shape),
                voxel_size=self.volume_model.voxel_size,
                volume_translation=volume_translation,
                volume_rotator=volume_rotator,
                volume_scale=volume_scale
            )

    def SetWordlTransform(self, _Transform):
        self.TransformWorld = _Transform

    def SetVoxelModel(self, _Voxel):
        self.ResultVoxel = _Voxel


def generate_rotating_volume(mesh2Voxel, n_frames=50, volumes=None):
    logRs = torch.zeros(n_frames, 3, device=device)
    logRs[:, 1] = torch.linspace(0.0, 2.0 * 3.14, n_frames, device=device)
    Rs = so3_exp_map(logRs)
    Ts = torch.zeros(n_frames, 3, device=device)
    Ts[:, 2] = 2.7
    frames = []
    print('Generating rotating volume ...')
    for R, T in zip(tqdm(Rs), Ts):
        # print('R', R, 'T', T)
        camera = FoVPerspectiveCameras(
            R=R[None],
            T=T[None],
            znear=mesh2Voxel.target_cameras.znear[0],
            zfar=mesh2Voxel.target_cameras.zfar[0],
            aspect_ratio=mesh2Voxel.target_cameras.aspect_ratio[0],
            fov=mesh2Voxel.target_cameras.fov[0],
            device=device,
        )
        temp = mesh2Voxel.renderer(cameras=camera, volumes=volumes)
        temp = temp[0]
        temp = temp[..., :3]

        frames.append(temp.clamp(0.0, 1.0))
    rotating_volume_frames = torch.cat(frames)
    image_grid(rotating_volume_frames.clamp(0., 1.).cpu().numpy(), rows=4, cols=7, rgb=True, fill=True)
    plt.show()

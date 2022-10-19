# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
# from training.volumetric_rendering.renderer import ImportanceRenderer
# from training.volumetric_rendering.ray_sampler import RaySampler
# import dnnlib

@persistence.persistent_class
class MonoPlaneGenerator(torch.nn.Module):
    def __init__(self,
                z_dim,                      # Input latent (Z) dimensionality.
                c_dim,                      # Conditioning label (C) dimensionality.
                w_dim,                      # Intermediate latent (W) dimensionality.
                img_resolution,             # Output resolution.
                img_channels,               # Number of output color channels.
                sr_num_fp16_res=0,
                mapping_kwargs={},   # Arguments for MappingNetwork.
                rendering_kwargs={},
                sr_kwargs={},
                **synthesis_kwargs):         # Arguments for SynthesisNetwork.
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        # self.renderer = ImportanceRenderer()
        # self.ray_sampler = RaySampler()
        self.uv_sampler = UvSampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32 * 1,
                                            mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        # self.superresolution = dnnlib.util.construct_class_by_name(
        #     class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution,
        #     sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.superresolution = Superresolution256(channels=32, img_channels=7, img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(
            32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs

        self._last_planes = None

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False,
                    use_cached_backbone=False, **synthesis_kwargs):
        # cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        # ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create monoplane by running StyleGAN backbone
        # N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes
        # Keep the original plane dimension of N x 32 x H x W

        # Create a batch of UV coordinates for SVBRDF map
        # N = ws.shape[0]
        N, C, _, _ = planes.shape
        uv_coords = self.uv_sampler(N, neural_rendering_resolution, ws.device)  # N x M x 2
        _, M, _ = uv_coords.shape

        sampled_features = torch.nn.functional.grid_sample(
            planes, uv_coords.unsqueeze(1).float(), mode='bilinear', padding_mode='zeros',  # N, C, 1, M
            align_corners=False).permute(0, 3, 2, 1).reshape(N, M, C)
        feature_samples = self.decoder(sampled_features)
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        svbrdf_image = feature_image[:, :7]
        sr_image = self.superresolution(
            svbrdf_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
            **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': svbrdf_image}

    # def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False,
    #             **synthesis_kwargs):
    #     # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes.
    #     ws = self.mapping(
    #         z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
    #     planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
    #     planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
    #     return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    # def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False,
    #                     **synthesis_kwargs):
    #     # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
    #     planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
    #     planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
    #     return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None,
                update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(
            z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(
            ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution,
            cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


class MonoPlaneNoSRGenerator(torch.nn.Module):
    def __init__(self,
                z_dim,                      # Input latent (Z) dimensionality.
                c_dim,                      # Conditioning label (C) dimensionality.
                w_dim,                      # Intermediate latent (W) dimensionality.
                img_resolution,             # Output resolution.
                img_channels,               # Number of output color channels.
                sr_num_fp16_res=0,
                mapping_kwargs={},   # Arguments for MappingNetwork.
                rendering_kwargs={},
                sr_kwargs={},
                **synthesis_kwargs):         # Arguments for SynthesisNetwork.
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.uv_sampler = UvSampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32 * 1,
                                            mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = SuperresolutionNoWeight(channels=32, img_channels=7, img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(
            32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 7})
        self.neural_rendering_resolution = 256
        self.rendering_kwargs = rendering_kwargs

        self._last_planes = None

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False,
                    use_cached_backbone=False, **synthesis_kwargs):
        # cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        # ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create monoplane by running StyleGAN backbone
        # N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes
        # Keep the original plane dimension of N x 32 x H x W

        # Create a batch of UV coordinates for SVBRDF map
        # N = ws.shape[0]
        N, C, _, _ = planes.shape
        uv_coords = self.uv_sampler(N, neural_rendering_resolution, ws.device)  # N x M x 2
        _, M, _ = uv_coords.shape

        sampled_features = torch.nn.functional.grid_sample(
            planes, uv_coords.unsqueeze(1).float(), mode='bilinear', padding_mode='zeros',  # N, C, 1, M
            align_corners=False).permute(0, 3, 2, 1).reshape(N, M, C)
        feature_samples = self.decoder(sampled_features)
        H = W = self.neural_rendering_resolution
        # feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        # svbrdf_image = feature_image[:, :7]
        # sr_image = self.superresolution(
        #     svbrdf_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
        #     **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        svbrdf_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        sr_image = self.superresolution(svbrdf_image)

        return {'image': sr_image}

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None,
                update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(
            z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(
            ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution,
            cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer, SynthesisBlock
from training.superresolution import SynthesisBlockNoUp
from torch_utils.ops import upfirdn2d


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )

    def forward(self, sampled_features):
        # Aggregate features
        # sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.contiguous().view(N * M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        out = torch.sigmoid(x[..., :]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF

        return out


class UvSampler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_size, resolution, device='cuda'):
        """
        Create batches of uv coordinates.

        resolution: int

        uv_coords: (N, M, 2), M = resolution ** 2
        """
        N = batch_size

        uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=device),
                                        torch.arange(resolution, dtype=torch.float32, device=device),
                                        indexing='ij')) * (1. / resolution) + (0.5 / resolution)
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        uv = uv.unsqueeze(0).repeat(N, 1, 1)

        return uv


class Superresolution512(torch.nn.Module):
    def __init__(self, channels, img_channels, img_resolution, sr_num_fp16_res, sr_antialias,
                num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,  # IGNORE
                **block_kwargs):
        super().__init__()
        assert img_resolution == 512

        use_fp16 = sr_num_fp16_res > 0
        self.input_resolution = 128
        self.sr_antialias = sr_antialias

        self.block0 = SynthesisBlock(channels, 128, w_dim=512, resolution=256,
                img_channels=img_channels, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), 
                **block_kwargs)
        self.block1 = SynthesisBlock(128, 64, w_dim=512, resolution=512,
                img_channels=img_channels, is_last=True, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), 
                **block_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1, 3, 3, 1]))

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] != self.input_resolution:
            # After ver 1.12.0, can add antialias=self.sr_antialias
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb


class Superresolution256(torch.nn.Module):
    def __init__(self, channels, img_channels, img_resolution, sr_num_fp16_res, sr_antialias,
                num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,# IGNORE
                **block_kwargs):
        super().__init__()
        assert img_resolution == 256
        use_fp16 = sr_num_fp16_res > 0
        self.sr_antialias = sr_antialias
        self.input_resolution = 128
        self.block0 = SynthesisBlockNoUp(channels, 128, w_dim=512, resolution=128,
                img_channels=img_channels, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.block1 = SynthesisBlock(128, 64, w_dim=512, resolution=256,
                img_channels=img_channels, is_last=True, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] < self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb


class SuperresolutionNoWeight(torch.nn.Module):
    def __init__(self, channels, img_channels, img_resolution, sr_num_fp16_res, sr_antialias,
                num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,# IGNORE
                **block_kwargs):
        super().__init__()
        assert img_resolution == 256
        use_fp16 = sr_num_fp16_res > 0
        self.sr_antialias = sr_antialias
        self.input_resolution = 256
        # self.block0 = SynthesisBlockNoUp(channels, 128, w_dim=512, resolution=128,
        #         img_channels=img_channels, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        # self.block1 = SynthesisBlock(128, 64, w_dim=512, resolution=256,
        #         img_channels=img_channels, is_last=True, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))

    def forward(self, rgb):
        rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)

        return rgb

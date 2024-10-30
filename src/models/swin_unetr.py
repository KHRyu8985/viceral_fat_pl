import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetrBasicBlock, UnetOutBlock
from monai.networks.nets.swin_unetr import SwinTransformer
from typing import Sequence
from monai.inferers import sliding_window_inference

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super(SPADE, self).__init__()
        self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)

        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out

class ModifiedUnetrUpBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, norm_name, res_block, label_nc):
        super().__init__()
        self.spa_de = SPADE(out_channels, label_nc=label_nc)  # Adjust label_nc as needed
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip, segmap):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        # Apply SPADE only if segmap is not None
        if segmap is not None:
            x = self.spa_de(x, segmap)

        x = self.conv(x)        
        x = self.relu(x)
        return x

class SwinUNETRv2(nn.Module):
    def __init__(
        self,
        img_size: Sequence[int],
        in_channels: int,
        out_channels: int,
        feature_size: int = 48,
        norm_name: str = "instance",
        use_checkpoint: bool = True,
        spatial_dims: int = 3,
        label_nc: int = 1,  # New parameter for segmap channels
    ) -> None:
        super().__init__()

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample="mergingv2",
            use_v2=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = ModifiedUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            norm_name=norm_name,
            res_block=True,
            label_nc=label_nc,  # Pass label_nc to the block
        )

        self.decoder4 = ModifiedUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            norm_name=norm_name,
            res_block=True,
            label_nc=label_nc,  # Pass label_nc to the block
        )

        self.decoder3 = ModifiedUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            norm_name=norm_name,
            res_block=True,
            label_nc=label_nc,  # Pass label_nc to the block
        )

        self.decoder2 = ModifiedUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            norm_name=norm_name,
            res_block=True,
            label_nc=label_nc,  # Pass label_nc to the block
        )

        self.decoder1 = ModifiedUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            norm_name=norm_name,
            res_block=True,
            label_nc=label_nc,  # Pass label_nc to the block
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(self, x_in, segmap):
        hidden_states_out = self.swinViT(x_in)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        
        # Apply segmap only in decoder5, decoder4, and decoder3
        dec3 = self.decoder5(dec4, hidden_states_out[3], segmap)
        dec2 = self.decoder4(dec3, enc3, segmap)
        dec1 = self.decoder3(dec2, enc2, segmap)
        
        # Do not apply segmap in decoder2 and decoder1
        dec0 = self.decoder2(dec1, enc1, None)
        out = self.decoder1(dec0, enc0, None)
        
        logits = self.out(out)
        return logits
    
if __name__ == "__main__":
    # Define the input parameters
    img_size = (96, 96, 96)  # Example image size
    in_channels = 1  # Example number of input channels
    out_channels = 2  # Example number of output channels
    feature_size = 48  # Example feature size
    label_nc = 3  # Example number of segmap channels

    # Create an instance of the model
    model = SwinUNETRv2(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        norm_name="instance",
        use_checkpoint=True,
        spatial_dims=3,
        label_nc=label_nc
    )

    # Set the model to evaluation mode
    model.eval()

    # Generate dummy input data
    x_in = torch.randn(1, in_channels, *img_size)  # Batch size of 1
    segmap = torch.randn(1, label_nc, *img_size)  # Example segmentation map with multiple channels

    # Define the sliding window parameters
    roi_size = (64, 64, 64)  # Size of the sliding window
    sw_batch_size = 4  # Number of windows to process in parallel
    overlap = 0.25  # Overlap between windows

    # Concatenate x_in and segmap along the channel dimension
    combined_input = torch.cat((x_in, segmap), dim=1)

    # Update the infer_func to handle the combined input
    def infer_func(inputs):
        # Determine the number of channels for x_in from the inputs
        num_channels_x_in = in_channels  # Use the in_channels defined outside
        x_in = inputs[:, :num_channels_x_in, ...]
        segmap = inputs[:, num_channels_x_in:, ...]
        return model(x_in, segmap)

    # Perform sliding window inference with the combined input
    logits = sliding_window_inference(
        inputs=combined_input,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=infer_func,
        overlap=overlap
    )

    # Print the output shape
    print("Sliding window output shape:", logits.shape)
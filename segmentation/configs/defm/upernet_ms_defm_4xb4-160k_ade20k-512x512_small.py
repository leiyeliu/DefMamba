_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="/home/user/code/ly/MSVMamba/classification/defm_small.pth",
        dims=96,
        depths=(2, 2, 6, 2),
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=1.0,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.3,
    ),
    # decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
    # auxiliary_head=dict(in_channels=256, num_classes=150)
)
# train_dataloader = dict(batch_size=4) # as gpus=4


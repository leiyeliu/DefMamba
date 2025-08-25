_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="/home/user/code/ly/MSVMamba/classification/defm_small.pth",
        # copied from classification/configs/vssm/vssm_tiny_224.yaml
        dims=96,
        depths=(2, 2, 6, 2),
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=1.0,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
    ),
)

train_dataloader = dict(batch_size=2) # as gpus=8


import os
from functools import partial

import torch

from .vmamba import VSSM
try:
    from .heat import HeatM
except:
    HeatM = None

# try:
#     from .vim import build_vim
# except Exception as e:
#     build_vim = lambda *args, **kwargs: None


# still on developing...
def build_vssm_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        # print(config.MODEL.VSSM.DEPTHS)
        # print(config.MODEL.VSSM.group)
        model = VSSM(
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_simple_init=config.MODEL.VSSM.SSM_SIMPLE_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            recurrent=config.MODEL.VSSM.RECURRENT,
            sscore_type=config.MODEL.VSSM.SSCORE_TYPE,
            inter_block_ssm=config.MODEL.VSSM.INTER_BLOCK_SSM,
            grid_size=config.MODEL.VSSM.GRID_SIZE,
            multiscale_ratio=config.MODEL.VSSM.MULTISCALE_RATIO,
            multiscale_ksize=config.MODEL.VSSM.MULTISCALE_KSIZE,
            continous_patch=config.MODEL.VSSM.CONTINOUS_PATCH,
            adaptive_merge=config.MODEL.VSSM.ADAPTIVE_MERGE,
            add_conv=config.MODEL.VSSM.ADD_CONV,
            b1_seq=config.MODEL.VSSM.B1_SEQ,
            b1_ratio=config.MODEL.VSSM.B1_RATIO,
            add_se = config.MODEL.VSSM.ADD_SE,
            ms_fusion = config.MODEL.VSSM.MS_FUSION,
            up_sample = config.MODEL.VSSM.UP_SAMPLE,
            convFFN = config.MODEL.VSSM.CONVFFN,
            lpu = config.MODEL.VSSM.LPU,
            sep_norm = config.MODEL.VSSM.SEP_NORM,
            ms_stage=config.MODEL.VSSM.MS_STAGE,
            ms_split = config.MODEL.VSSM.MS_SPLIT,
            ffn_dropout = config.MODEL.VSSM.FFN_DROPOUT,
        )
        return model

    return None


def build_model(config, is_pretrain=False):
    model = build_vssm_model(config, is_pretrain)
    return model





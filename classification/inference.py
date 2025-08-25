from config import get_config
from models import build_model
import argparse
import datetime
import time
import torch
from main import str2bool
import numpy as np
from utils.logger import create_logger

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--save_every', type=str2bool, default=False, help='')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main(config):
    ckpt_path = '/home/li/yh/mamba/vmamba/exclude/nan/nan_model.pth'
    print(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    #del all module. prefix
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    msg = model.load_state_dict(checkpoint, strict=True)
    print(f"resuming model: {msg}")
    nan_output = np.load('../exclude/nan/nan_output.npy')
    nan_sample = np.load('../exclude/nan/nan_sample.npy')
    model.train()
    nan_sample = torch.from_numpy(nan_sample)[120:].cuda(non_blocking=True)
    #to fp16
    #nan_sample = nan_sample.half()
    with torch.cuda.amp.autocast(enabled=True):
        output = model(nan_sample)

    # Check if the output is nan
    if torch.isnan(output).any():
        print("Output contains nan")
    else:
        print("Output does not contain nan")




if __name__ == '__main__':
    args, config = parse_option()
    main(config)
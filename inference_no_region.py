'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule, getImgEmbed, postprocess
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result
from torch.utils.tensorboard import SummaryWriter
from data.dense_dataset import blip_collate_fn,blip_eval_collate_fn,blip_test_collate_fn
from densecap import densecap_resnet50_fpn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval() 
    cudnn.benchmark = False
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 2

    result = []
    iter0 = 0
    
    for image_for_extract, mask_tensor_list, boxes_list, image_id in metric_logger.log_every(data_loader, print_freq, header): 

        try:
            image_for_extract = torch.stack(image_for_extract).to(device)
            for idx in range(0,image_for_extract.shape[0]):#逐张
                captions = model.generate(image_for_extract[idx].unsqueeze(0), torch.stack(mask_tensor_list[idx]).to(device), device, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                        min_length=config['min_length'])
                for region_idx, (caption) in enumerate(captions):       
                    result.append({"image_id": image_id[idx], "caption": caption, "boxes":boxes_list[idx][region_idx] })                 
        except Exception as e:
            print(str(e))
        
   
    print("===Normal Eval Finish===")  
  
    return result



def main(args, config): 
    device = torch.device(args.device)
    config['prompt'] = 'an area of '

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #### Dataset #### 
    print("Creating captioning inferencing dataset")
    # train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', config) 
    train_dataset, val_dataset, test_dataset = create_dataset('dense', config, device)  


    samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[1,2,2],
                                                          is_trains=[True, False, False], collate_fns=[blip_collate_fn,blip_eval_collate_fn,blip_test_collate_fn])         

    ### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])

    model = model.to(device)  
    model.load_state_dict(torch.load('output/knowledge_extract_model_epoch5.pt')) 

    

    print("Finish Creating model")
    model_without_ddp = model
            
    print("Start Inference")

    val_result = evaluate(model_without_ddp, test_loader, device, config)  
    val_result_file = save_result(val_result, args.result_dir, 'full_val_for_5_with_region', remove_duplicate='image_id')            
                    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--config', default='./configs/caption_dense.yaml')
    parser.add_argument('--output_dir', default='output/Caption_dense')   
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)

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
from data.dense_dataset import blip_collate_fn,blip_eval_collate_fn
from densecap import densecap_resnet50_fpn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn


@torch.no_grad()
def evaluate(model, mask_model, data_loader, device, config):
    # evaluate
    model.eval() 
    mask_model.eval()
    cudnn.benchmark = False
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 2

    result = []
    iter0 = 0
    
    for image_for_region, image_for_extract, image_org_size, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        # torch.cuda.empty_cache()
        iter0 += 1
        # if iter0 < 340:
        #     continue
        try:
            image_for_region = [img.to(device) for img in image_for_region]     
            image_for_extract = torch.stack(image_for_extract).to(device)
            _, res, after_mask_model_size = mask_model(image_for_region)  
            res['predict_region'] = postprocess(res['predict_region'], image_org_size, after_mask_model_size, device)
            mask_tensor_list = getImgEmbed(res['predict_region'], image_org_size)
            mask_tensor_list = [mask_tensor.to(device) for mask_tensor in mask_tensor_list]
            
            for idx in range(0,image_for_extract.shape[0]):#逐张
                captions = model.generate(image_for_extract[idx].unsqueeze(0), mask_tensor_list[idx], device, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                        min_length=config['min_length'])
                for region_idx, (caption) in enumerate(captions):       
                    result.append({"image_id": image_id[idx], "caption": caption, "corresponding_region":res['predict_region'][idx][region_idx].cpu().numpy().tolist()})                 
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
                                                          is_trains=[True, False, False], collate_fns=[blip_collate_fn,blip_eval_collate_fn,None])         

    ### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])

    model = model.to(device)  
    model.load_state_dict(torch.load('output/knowledge_extract_model_epoch5.pt')) 

    mask_model = densecap_resnet50_fpn(backbone_pretrained=config['backbone_pretrained'],#True
                                  feat_size=config['feat_size'],#4096
                                  hidden_size=config['hidden_size'],#512
                                  max_len=config['max_len'],#16
                                  emb_size=config['emb_size'],#512
                                  rnn_num_layers=config['rnn_num_layers'],#1
                                  vocab_size=config['vocab_size'],#10629
                                  fusion_type=config['fusion_type'],#init_inject
                                  box_detections_per_img=config['box_detections_per_img'])#50

    mask_model.load_state_dict(torch.load('region_model/model_result/region_detection_model.pt'))
   

    mask_model = mask_model.to(device)
    print("Finish Creating model")
    model_without_ddp = model
            

    print("Start Inference")

             
    val_result = evaluate(model_without_ddp, mask_model, val_loader, device, config)  
    val_result_file = save_result(val_result, args.result_dir, 'full_val_for_5', remove_duplicate='image_id')            
                    



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

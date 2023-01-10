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
from data.dense_dataset import blip_collate_fn
from densecap import densecap_resnet50_fpn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

# device = torch.device('cpu')
def train(model, mask_model, data_loader, optimizer, epoch, device, max_caption_num = 15):#实际应为88
# def train(mask_model, data_loader, epoch, device, max_caption_num = 15):#实际应为88

    # train
    model.train()  
    mask_model.train()

    writer = SummaryWriter(log_dir='./tensorboard_dense/test/'+ time.strftime('%y-%m-%d_%H.%M', time.localtime())) # 确定路径
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    print('Train Caption Epoch: [{}]'.format(epoch))
    # header = 'Train Caption Epoch: [{}]'.format(epoch)
    # print_freq = 50
    i = 0

    for image, image_org_size, targets, caption_actual_num in tqdm.tqdm(data_loader):#image:batch_size*3*384*384 caption
            image = [img.to(device) for img in image]
            for item in targets:
                item['boxes'] = item['boxes'].to(device)
                item['caps'] = item['caps'].to(device)
                item['caps_len'] = item['caps_len'].to(device)
            mask_losses, res, after_mask_model_size = mask_model(image, targets)
            # image = image.to(device)
            if res == None:
                continue
            res['predict_region'] = postprocess(res['predict_region'],image_org_size,after_mask_model_size,device)
            res['matched_gt_boxes'] = postprocess(res['matched_gt_boxes'],image_org_size,after_mask_model_size,device)

            # loss_box_reg, proposal, gt_boxes_list, corr_region_cap_list = mask_model(image, targets)
            mask_tensor_list = getImgEmbed(res['predict_region'],image_org_size)
            mask_tensor_list = [mask_tensor.to(device) for mask_tensor in mask_tensor_list]

            model = model.to(device)

            ctploss = model(image, max_caption_num, caption_actual_num, res, mask_tensor_list)      
        
            optimizer.zero_grad()
            overall_loss = ctploss +  mask_losses['loss_box_reg'] + mask_losses['loss_rpn_box_reg']
            overall_loss.backward()
            # for losses, box_reg_loss in zip(loss_list,mask_losses['loss_box_reg']):
            #     overall_loss = box_reg_loss + losses
            #     overall_loss.backward()
            optimizer.step()    
        
            writer.add_scalar('train_overall_loss',overall_loss.item(),i)
            writer.add_scalar('mask_loss',mask_losses['loss_box_reg'].item(),i)
            writer.add_scalar('generate_ctp_loss',ctploss.item(),i)

            # writer.add_scalar('train_ctp_loss',ctploss_list[-1].item(),i)
            # if len(regloss_list) > 0:
            #     writer.add_scalar('train_reg_loss',regloss_list[-1],i)
            i += 1
            
            metric_logger.update(loss=overall_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            if i >= 2000:
                print("interrupt from 10!")
                break

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluate(model, mask_model, data_loader, device, config):
    # evaluate
    model.eval() 
    mask_model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 1
    decoder_num = 15

    result = []
    iter0 = 0
    print("Eval Start") 
    for image, image_org_size, tensor_list, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
        image = image.to(device)     
        # image = [img.to(device) for img in image]
        mask_losses, res, after_mask_model_size = mask_model(image)  
        res['predict_region'] = postprocess(res['predict_region'],image_org_size, after_mask_model_size, device)
        mask_tensor_list = getImgEmbed(res['predict_region'], image_org_size)
        mask_tensor_list = [mask_tensor.to(device) for mask_tensor in mask_tensor_list]
        
        for idx in range(0,image.shape[0]):#逐张
            captions = model.generate(image[idx].unsqueeze(0), mask_tensor_list[idx], device, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                    min_length=config['min_length'])
            for region_idx, (caption) in enumerate(captions):       
                result.append({"image_id": image_id[idx].item(), "caption": caption, "corresponding_region":res['predict_region'][idx][region_idx].cpu().numpy().tolist()})                 
        iter0 += 1
        if iter0 >= 100:
            break    
    print("eval finish--")  
  
    return result


def main(args, config):
    if args.distributed:
        utils.init_distributed_mode(args)    
    device = torch.device(args.device)
    # device = 'cuda:3'
    config['prompt'] = 'an area of '
    args.distributed = False

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    # train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', config) 
    train_dataset, val_dataset, test_dataset = create_dataset('dense', config, device)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[1,2,2],
                                                          is_trains=[True, False, False], collate_fns=[blip_collate_fn,None,None])         

    ### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])

    model = model.to(device)   
    # mask_model = mask_model.to(device)  


    mask_model = densecap_resnet50_fpn(backbone_pretrained=config['backbone_pretrained'],#True
                                  feat_size=config['feat_size'],#4096
                                  hidden_size=config['hidden_size'],#512
                                  max_len=config['max_len'],#16
                                  emb_size=config['emb_size'],#512
                                  rnn_num_layers=config['rnn_num_layers'],#1
                                  vocab_size=config['vocab_size'],#10629
                                  fusion_type=config['fusion_type'],#init_inject
                                  box_detections_per_img=config['box_detections_per_img'])#50
    # if config['use_pretrain_fasterrcnn']:#true
    mask_model.backbone.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).backbone.state_dict(), strict=False)
    mask_model.rpn.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).rpn.state_dict(), strict=False)

    mask_model = mask_model.to(device)
    print("Finish Creating model- pretrain")
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': (para for name, para in mask_model.named_parameters()
                                        if para.requires_grad and 'box_describer' not in name)},
                                  {'params': (para for para in mask_model.roi_heads.box_describer.parameters()
                                              if para.requires_grad), 'lr':  1e-3}],
                                  lr=config['init_lr'], weight_decay=config['weight_decay'])
            
    best = 0
    best_epoch = 0
    args.evaluate = False

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats = train(model, mask_model, train_loader, optimizer, epoch, device)
            # train_stats = train(mask_model, train_loader, epoch, device) 
             
        
        val_result = evaluate(model_without_ddp, mask_model, val_loader, device, config)  
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id')        
  
        # test_result = evaluate(model_without_ddp, test_loader, device, config)  
        # test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d'%epoch, remove_duplicate='image_id')  

        # if utils.is_main_process():   
        #     coco_val = coco_caption_eval(config['coco_gt_root'],val_result_file,'val')
        #     coco_test = coco_caption_eval(config['coco_gt_root'],test_result_file,'test')
            
        #     if args.evaluate:            
        #         log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
        #                      **{f'test_{k}': v for k, v in coco_test.eval.items()},                       
        #                     }
        #         with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
        #             f.write(json.dumps(log_stats) + "\n")                   
        #     else:             
        #         save_obj = {
        #             'model': model_without_ddp.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'config': config,
        #             'epoch': epoch,
        #         }

        #         if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
        #             best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
        #             best_epoch = epoch                
        #             torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    
        #         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                      **{f'val_{k}': v for k, v in coco_val.eval.items()},
        #                      **{f'test_{k}': v for k, v in coco_test.eval.items()},                       
        #                      'epoch': epoch,
        #                      'best_epoch': best_epoch,
        #                     }
        #         with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        #             f.write(json.dumps(log_stats) + "\n")     
                    
        if args.evaluate: 
            break
        # dist.barrier()     

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default='./configs/caption_coco.yaml')
    # parser.add_argument('--output_dir', default='output/Caption_coco')        
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
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
from utils import cosine_lr_schedule, getImgEmbed, postprocess, packToJsonAndVisualize
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel as OrgBertModel
from data.region_dataset import regiondata_collate_fn
from densecap import densecap_resnet50_fpn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

def train(mask_model, data_loader, optimizer, epoch, device):#实际应为88

    mask_model.train()

    writer = SummaryWriter(log_dir='./tensorboard_region/test/'+ time.strftime('%y-%m-%d_%H.%M', time.localtime())) # 确定路径
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    print('Train Caption Epoch: [{}]'.format(epoch))
    # header = 'Train Caption Epoch: [{}]'.format(epoch)
    # print_freq = 50
    i = 0
    visualize_region_result = []

    for image, image_org_size, targets, tokenizer, img_id in tqdm.tqdm(data_loader):#image:batch_size*3*384*384 caption
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
        ##添加部分训练集的预测结果，供可视化查看效果
        if i % 1000 == 0: #and i != 0:
            packToJsonAndVisualize(visualize_region_result,res['predict_region'],res['matched_gt_boxes'],res['corr_region_cap'],img_id,tokenizer)

        optimizer.zero_grad()
        overall_loss = mask_losses['loss_box_reg'] + mask_losses['loss_rpn_box_reg'] + mask_losses['loss_text_generate']
        torch.backends.cudnn.benchmark = False
        overall_loss.backward()
        optimizer.step()   
        torch.cuda.synchronize() 
    
        writer.add_scalar('train_overall_loss',overall_loss.item(),i)
        writer.add_scalar('mask_loss',mask_losses['loss_box_reg'].item(),i)

        i += 1
        
        metric_logger.update(loss=overall_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])       

    visualize_region_result_fileName = 'visualize_train_result-epoch' + str(epoch) +'.json'
    with open("/local/scratch3/xfang31/BLIP-MultiDecoder/output/Caption_dense/region_train/"+ visualize_region_result_fileName,"w") as f1:
        json.dump(visualize_region_result,f1)
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluate(mask_model, data_loader, device, config):
    # evaluate
    mask_model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10
    decoder_num = 15

    result = []
    iter0 = 0
    print("Eval Start") 
    for image, image_org_size, tensor_list, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
        image = image.to(device)     

        _, res, after_mask_model_size = mask_model(image)  
        res['predict_region'] = postprocess(res['predict_region'], image_org_size, after_mask_model_size, device)

        for imgid, region in zip(image_id,res['predict_region']):
            result.append({'image_id':imgid.item(),'region_proposal':region.cpu().numpy().tolist()})

        iter0 += 1
        if iter0 >= 2000:
            break  
        
    print("===eval finish===")  
  
    return result

def main(args, config):   
    device = torch.device(args.device)
    # device = 'cuda:3'
    config['prompt'] = 'an area of '#caption prompt

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('region', config, device)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[1,2,2],
                                                          is_trains=[True, False, False], collate_fns=[regiondata_collate_fn,None,None])         

    ### Model #### 
    print("Creating model") 

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
 
    
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    optimizer = torch.optim.Adam([{'params': (para for name, para in mask_model.named_parameters()
                                        if para.requires_grad and 'box_describer' not in name)},
                                  {'params': (para for para in mask_model.roi_heads.box_describer.parameters()
                                              if para.requires_grad), 'lr':  1e-3}],
                                  lr=config['init_lr'], weight_decay=config['weight_decay'])
            
    args.evaluate = False
    minloss = 100000.0
    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats = train(mask_model, train_loader, optimizer, epoch, device)
            if minloss > float(train_stats['loss']):
                print("update min loss in epoch "+str(epoch))
                print("min loss is "+train_stats['loss'])
                minloss = float(train_stats['loss'])
                torch.save(mask_model.state_dict(),'region_model/model_result/region_detection_model.pt')
            
        val_result = evaluate(mask_model, val_loader, device, config)  
        val_result_file = save_result(val_result, args.result_dir, 'region_cap_val_epoch%d'%epoch, remove_duplicate='image_id')        
                    
        if args.evaluate: 
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_dense.yaml')
    parser.add_argument('--output_dir', default='./output/Caption_dense')   
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
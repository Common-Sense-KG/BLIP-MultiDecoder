from datasets import load_dataset
from transformers.utils.dummy_vision_objects import ImageGPTFeatureExtractor
import random
from PIL import ImageDraw, ImageFont, Image
from data import create_dataset, create_sampler, create_loader
from transformers import ViTFeatureExtractor
import torch
import argparse
import ruamel.yaml as yaml
import os
from pathlib import Path
import tqdm

def train(feature_extractor, data_loader, epoch, device):
    # train
    print_freq = 50
    i = 0

    for image in tqdm.tqdm(data_loader):#image:batch_size*3*384*384 caption

    # for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)    #2(batch size)*3(channel)*384*384   
        
        model = model.to(device)

        loss_list, ctploss_list, regloss_list = model(image, caption, tensor_list, max_caption_num, caption_actual_num)      
        
        optimizer.zero_grad()
        for loss in loss_list:
            loss.backward()
        optimizer.step()    
        
        writer.add_scalar('train_overall_loss',loss.item(),i)
        writer.add_scalar('train_ctp_loss',ctploss_list[-1].item(),i)
        if len(regloss_list) > 0:
            writer.add_scalar('train_reg_loss',regloss_list[-1],i)
        i += 1
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if i >= 7000:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

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
        
    # yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    device = torch.device('cuda:3')
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
    train_dataset, val_dataset, test_dataset = create_dataset('dense', config)  
    samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[2,2,2],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])  
    # feature_extractor(image, return_tensors='pt') 
    print("Start training")
    # start_time = time.time()    
    for epoch in range(0, 3):   
        train_stats = train(feature_extractor, test_loader, epoch, device) 
        
        val_result = evaluate(model_without_ddp, val_loader, device, config)  
        # val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id')   

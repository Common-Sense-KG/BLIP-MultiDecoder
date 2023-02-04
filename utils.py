import math
def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    

def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):        
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
        
import numpy as np
import io
import os
import time
import sys
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

# from sentence_transformers.util import cos_sim  
# from sentence_transformers import SentenceTransformer as SBert
import torch.nn.functional as F

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)        
        
from annotation.data_preprocess import squareOverlap
def getImgEmbed(predict_region_list,image_org_size):#xx*4张量
    Img_mask = []
    for predict_region_tensor , img_size in zip(predict_region_list,image_org_size):
        pixelW = img_size[0] / 24
        pixelH = img_size[1] / 24
        image_mask_per_image = torch.zeros([predict_region_tensor.shape[0],577])
        for idx in range(predict_region_tensor.shape[0]):
            x = predict_region_tensor[idx][0].item()
            y = predict_region_tensor[idx][1].item()
            w = predict_region_tensor[idx][2].item() - predict_region_tensor[idx][0].item()
            h = predict_region_tensor[idx][3].item() - predict_region_tensor[idx][1].item()
            empty = torch.zeros(1,577)
            for i in range(0,576):
                x0 = i % 24 * pixelW
                y0 = i // 24 * pixelH
                if squareOverlap(x0,y0,pixelW,pixelH,x,y,w,h):
                    empty[0][i] = 1
            # print(i)
            empty[0][576] = 1
            image_mask_per_image[idx] = empty
        Img_mask.append(image_mask_per_image)
    # print(empty)
    return Img_mask

    
# def postprocess(res,org_size,now_size,device):
#     predict_boxes_list = res['predict_region']
#     gt_boxes_list = res['matched_gt_boxes']
#     ratios = [
#         torch.tensor(s_org, dtype=torch.float32, device=device)
#         / torch.tensor(s, dtype=torch.float32, device=device)
#         for s, s_org in zip(now_size, org_size)
#     ]
    
#     # ratio_height, ratio_width = ratios
#     for i,(predict_item,gt_item) in enumerate(zip(predict_boxes_list,gt_boxes_list)):
#         ratio_height, ratio_width = ratios[i].t() 
#         for idx in range(predict_item.shape[0]):
#             xmin, ymin, xmax, ymax = predict_item[idx].t()
#             xmin = xmin * ratio_width
#             xmax = xmax * ratio_width
#             ymin = ymin * ratio_height
#             ymax = ymax * ratio_height
#             predict_item[idx] = torch.stack((xmin.unsqueeze(0), ymin.unsqueeze(0), xmax.unsqueeze(0), ymax.unsqueeze(0)), dim=1).squeeze(0)
#             xmin, ymin, xmax, ymax = gt_item[idx].t()
#             xmin = xmin * ratio_width
#             xmax = xmax * ratio_width
#             ymin = ymin * ratio_height
#             ymax = ymax * ratio_height
#             gt_item[idx] = torch.stack((xmin.unsqueeze(0), ymin.unsqueeze(0), xmax.unsqueeze(0), ymax.unsqueeze(0)), dim=1).squeeze(0)
            
#     res['predict_region'] = predict_boxes_list
#     res['matched_gt_boxes'] = gt_boxes_list
#     return res

#     # return torch.stack((xmin, ymin, xmax, ymax), dim=1)
    
    
def postprocess(tensor,org_size,now_size,device):
    ratios = [
        torch.tensor(s_org, dtype=torch.float32, device=device)
        / torch.tensor(s, dtype=torch.float32, device=device)
        for s, s_org in zip(now_size, org_size)
    ]
    # ratio_height, ratio_width = ratios
    for i,(item) in enumerate(tensor):
        ratio_width, ratio_height = ratios[i].t() 
        for idx in range(item.shape[0]):
            xmin, ymin, xmax, ymax = item[idx].t()
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            item[idx] = torch.stack((xmin.unsqueeze(0), ymin.unsqueeze(0), xmax.unsqueeze(0), ymax.unsqueeze(0)), dim=1).squeeze(0)
            
    return tensor    

def packToJsonAndVisualize(result,predict_region,matched_gt_boxes,text,img_id,tokenizer):
    for i in range(len(predict_region)):
        region = predict_region[i].detach().cpu().numpy().tolist()
        gt_boxes = matched_gt_boxes[i].detach().cpu().numpy().tolist()
        imgID = img_id[i]
        sentence_list = text[i].detach().cpu().numpy().tolist()
        true_sentence_list = []
        for sentence in sentence_list:
            caption = tokenizer.decode(sentence, skip_special_tokens=True)
            true_sentence_list.append(caption)
        result.append({'image_id':imgID,'region_proposal':region,'gt_boxes':gt_boxes,'corr_sentence':true_sentence_list})

def compare_sentence_similarity(caption,captions_list,tokenizer):
    similarity_score = []
    tokenize_result = tokenizer(captions_list, padding=True, return_tensors="pt")
    single_caption = torch.tensor(tokenizer(caption).input_ids).unsqueeze(0)
    for caption_idx in range(tokenize_result.input_ids.shape[0]):
        main_tensor = tokenize_result.input_ids[caption_idx].unsqueeze(0)
        if main_tensor.shape[1] > single_caption.shape[1]:
            sub_main_tensor = torch.cat([single_caption,torch.zeros(1,main_tensor.shape[1] - single_caption.shape[1])],dim=1)
            similarity_score.append(F.cosine_similarity(main_tensor[:,4:].float(),sub_main_tensor[:,4:].float()))#去掉前缀再计算相似度
        else:
            main_tensor = torch.cat([main_tensor,torch.zeros(1,single_caption.shape[1] - main_tensor.shape[1])],dim=1)
            similarity_score.append(F.cosine_similarity(main_tensor[:,4:].float(),single_caption[:,4:].float()))

    return max(similarity_score)

    # single_list = [caption]
    # embedding_single = model.encode(single_list)
    # embedding_all = model.encode(captions_list)
    # cosine_scores = cos_sim(embedding_single,embedding_all)
    # return max(cosine_scores)
    

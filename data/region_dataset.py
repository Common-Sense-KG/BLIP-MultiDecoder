# dataset for dense
import os
import json
import random

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption
import numpy as np
import cv2
import torch
from transformers import BertTokenizer
import torchvision.transforms as trans

# from data.utils import pre_caption_dense
def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

class region_train(Dataset):
    def __init__(self, transform, image_root, ann_root, device, max_words=30, prompt='an area of '):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        
        filename = 'dense_train.json'
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        tf_idf_name = 'dense_train_predicate_tfidf_score.json'
        self.corresponding_tf_idf = json.load(open(os.path.join(ann_root,tf_idf_name),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.tokenizer = init_tokenizer()
        self.device = device
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    

        trans1 = trans.ToTensor()
        
        ann = self.annotation[index]#根据index获取annotation
        output_targets = {'boxes':torch.tensor([]),'caps':torch.tensor([]),'caps_len':torch.tensor([])}
        
        image_path = os.path.join(self.image_root,str(ann['image_id'])+'.jpg')        
        image = Image.open(image_path).convert('RGB')   
        image_tensor = trans1(image)
        width,height = image.size[1], image.size[0]
        # image = self.transform(image)
        image_org_size = [width,height]

        caption = ''
        captions_list = []
        boxes_list = []

        ###region 训练，全部放入
        for i,(phrase) in enumerate(ann['phrase_list']):
            caption = self.prompt + pre_caption(phrase['caption'])
            captions_list.append(caption)
            phrase['boxes'] = [x + 0.01 for x in phrase['boxes']]#防止0对后边回归的参数影响
            boxes_list.append(phrase['boxes'])
            
        tokenize_result = self.tokenizer(captions_list,padding=True, return_tensors="pt")

        output_targets['boxes'] = torch.tensor(boxes_list)
        output_targets['caps'] = tokenize_result.input_ids
        caps_len = []
        for i in range(tokenize_result.attention_mask.shape[0]):
            caps_len.append(torch.count_nonzero(tokenize_result.attention_mask[i]).item())

        output_targets['caps_len'] = torch.tensor(caps_len)

        return image_tensor, image_org_size, output_targets, self.tokenizer, ann['image_id']
    
class region_eval(Dataset):
    def __init__(self, image_root, ann_root):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        filename = 'dense_eval.json'
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = trans.ToTensor()
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,str(ann['image_id'])+'.jpg')     
        image = Image.open(image_path) 
        width,height = image.size[0],image.size[1]
        image_org_size = [width,height]

        image = image.convert('RGB')  
        image = self.transform(image)    
              
        img_id = ann['image_id']
        
        return image, image_org_size, int(img_id)   
        
    
    
class region_test(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        filename = 'dense_test.json'
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(self.image_root+str(ann['image_id'])+'.jpg')#存路径
            self.img2txt[img_id] = []#逐项初始化
            for i, caption in enumerate(ann['phrase_list']):#对每一个caption进行预处理 append id
                self.text.append(pre_caption(caption['caption'],max_words))#是否有不同？ 此处仍为拼接 还是逐个存储
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id#互相认定
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, str(self.annotation[index]['image_id'])+'.jpg')        
        image = Image.open(image_path)

        return image, int(self.annotation[index]['image_id'])

def region_train_data_collate_fn(data):
    image_list, image_org_size, output_targets, image_id_list = [],[],[],[]

    for image_item, size, targets_item, tokenizer, image_id in data:
        image_list.append(image_item)
        image_org_size.append(size)
        output_targets.append(targets_item)
        image_id_list.append(image_id)
        dataTokenizer = tokenizer

    return image_list, image_org_size, output_targets, dataTokenizer, image_id_list

def region_eval_data_collate_fn(data):
    image_list, image_org_size_list, img_id_list = [], [], []

    for image,image_org_size,img_id in data:
        image_list.append(image)
        image_org_size_list.append(image_org_size)
        img_id_list.append(img_id)

    return image_list,image_org_size_list,img_id_list


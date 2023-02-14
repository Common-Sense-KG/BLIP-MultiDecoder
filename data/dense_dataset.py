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

import sys
from sentence_transformers.util import cos_sim  
from sentence_transformers import SentenceTransformer as SBert
from utils import compare_sentence_similarity

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

class dense_train(Dataset):
    def __init__(self, transform, image_root, ann_root, device, max_words=30, prompt='an area of '):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        #url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'augment/dense_train_ConceptNet_augment.json'
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        tf_idf_name = 'dense_train_predicate_tfidf_score.json'
        self.corresponding_tf_idf = json.load(open(os.path.join(ann_root,tf_idf_name),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.tokenizer = init_tokenizer()
        self.device = device
        # self.similarity_compare_model = SBert('paraphrase-multilingual-MiniLM-L12-v2')
        
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

        # transToTensor = trans.ToTensor()
        
        ann = self.annotation[index]#根据index获取annotation
        # match_tf_idf = self.corresponding_tf_idf[index]
        output_targets = {'boxes':torch.tensor([]),'caps':torch.tensor([]),'caps_len':torch.tensor([])}
        
        image_path = os.path.join(self.image_root,str(ann['image_id'])+'.jpg')        
        image = Image.open(image_path).convert('RGB')   
        # image = transToTensor(image)
        width,height = image.size[1],image.size[0]
        image = self.transform(image)
        image_org_size = [width,height]


        ###### org: tfidf + cosine similarity
        # caption = ''
        # captions_list = []
        # boxes_list = []
        # mask_list = []
        # times = 0
        # ##引入tf idf对数据做筛选
        # while len(captions_list) < len(ann['phrase_list']) * 0.6 :
        #     times += 1
        #     if times > 500 and len(captions_list) > 0:
        #         break
        #     for i,(phrase) in enumerate(ann['phrase_list']):
        #         tf_idf_score = match_tf_idf['predicate_tf_idf_score'][i]
        #         if tf_idf_score['tf_idf_score']< 0.4 and random.uniform(tf_idf_score['tf_idf_score'], 1) < 0.55 :
        #             continue
        #         caption = self.prompt + pre_caption(phrase['caption'])
        #         if caption in captions_list:
        #             continue
        #         elif len(captions_list) > 0 and compare_sentence_similarity(caption,captions_list,self.tokenizer) > 0.75:
        #             continue
                
        #         captions_list.append(caption)
        #         boxes_list.append(phrase['boxes'])
        #         mask_list.append(phrase['tensor'])
            
        # tokenize_result = self.tokenizer(captions_list, padding=True, return_tensors="pt")

        # output_targets['boxes'] = torch.tensor(boxes_list)
        # output_targets['image_mask'] = torch.tensor(mask_list)
        # output_targets['caps'] = tokenize_result.input_ids
        # caps_len = []
        # for i in range(tokenize_result.attention_mask.shape[0]):
        #     caps_len.append(torch.count_nonzero(tokenize_result.attention_mask[i]).item())

        # output_targets['caps_len'] = torch.tensor(caps_len)
         
        # return image, image_org_size, output_targets, ann['image_id']

        ###new code
        captions_list = []
        boxes_list = []
        mask_list = []
        for phrase in ann['phrase_list']:
            captions_list.append(self.prompt + pre_caption(phrase['caption']))
            boxes_list.append(phrase['boxes'])
            mask_list.append(phrase['tensor'])
            
        tokenize_result = self.tokenizer(captions_list, padding=True, return_tensors="pt")

        output_targets['boxes'] = torch.tensor(boxes_list)
        output_targets['image_mask'] = torch.tensor(mask_list)
        output_targets['caps'] = tokenize_result.input_ids
        caps_len = []
        for i in range(tokenize_result.attention_mask.shape[0]):
            caps_len.append(torch.count_nonzero(tokenize_result.attention_mask[i]).item())

        output_targets['caps_len'] = torch.tensor(caps_len)
         
        return image, image_org_size, output_targets, ann['image_id']
    
class dense_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, device):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        # urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
        #         'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        # filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        filename = 'dense_eval.json'
        # download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    

        transToTensor = trans.ToTensor()
        
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root,str(ann['image_id'])+'.jpg')      
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  #modify
        image = Image.open(image_path) 
        width,height = image.size[1],image.size[0]
        image_org_size = [width,height]
        # results = model(image) 
        image = image.convert('RGB')  
        image_for_region = transToTensor(image)
        image_for_extract = self.transform(image)    

        img_id = ann['image_id']
        
        return image_for_region, image_for_extract, image_org_size, int(img_id)   
        


class dense_test(Dataset):#contain region inference
    def __init__(self, transform, image_root, ann_root, device):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        # urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
        #         'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        # filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        filename = 'dense_eval.json'
        # download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    

        transToTensor = trans.ToTensor()
        
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root,str(ann['image_id'])+'.jpg')      
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  #modify
        image = Image.open(image_path) 
        width,height = image.size[1],image.size[0]
        image_org_size = [width,height]
        # results = model(image) 
        tensor_list = []
        boxes_list = []
        image = image.convert('RGB')  
        image_for_extract = self.transform(image)    
        for phrase in ann['phrase_list']:
            tensor_list.append(torch.tensor(np.array(phrase['tensor'])))  
            boxes_list.append(phrase['boxes'])  
              
        # max_caption_num = 70
        # truly_length = len(tensor_list)
        img_id = ann['image_id']
        # tensor_list  += [np.zeros((1,577),np.float64) for j in range(max_caption_num - truly_length)]
        
        return image_for_extract, tensor_list, boxes_list, int(img_id)   
        
 
    
# class dense_test(Dataset):
#     def __init__(self, transform, image_root, ann_root, device, max_words=30):  
#         '''
#         image_root (string): Root directory of images (e.g. coco/images/)
#         ann_root (string): directory to store the annotation file
#         split (string): val or test
#         '''
#         # urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
#         #         'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
#         # filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
#         # download_url(urls[split],ann_root)

#         # filename = 'objects.json'
#         # ann_root = '/home/hcui25/Research/BLIP-MultiDecoder/annotation/vg_org'
#         filename = 'dense_test.json'
        
#         self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
#         self.transform = transform
#         self.image_root = image_root
        
#         self.text = []
#         self.image = []
#         self.txt2img = {}
#         self.img2txt = {}
        
#         txt_id = 0
#         for img_id, ann in enumerate(self.annotation):
#             self.image.append(self.image_root+str(ann['image_id'])+'.jpg')#存路径
#             self.img2txt[img_id] = []#逐项初始化
#             for i, caption in enumerate(ann['phrase_list']):#对每一个caption进行预处理 append id
#                 self.text.append(pre_caption(caption['caption'],max_words))#是否有不同？ 此处仍为拼接 还是逐个存储
#                 self.img2txt[img_id].append(txt_id)
#                 self.txt2img[txt_id] = img_id#互相认定
#                 txt_id += 1
                                    
#     def __len__(self):
#         return len(self.annotation)
    
#     def __getitem__(self, index):    
        
#         image_path = os.path.join(self.image_root, str(self.annotation[index]['image_id'])+'.jpg')        
#         image = Image.open(image_path)
#         # image = self.transform(image)  

#         return image, int(self.annotation[index]['image_id'])

def blip_eval_collate_fn(data):
    # image_for_region, image_for_extract, image_org_size, int(img_id)  
    image_for_region_list, image_for_extract_list, image_org_size_list, image_id_list = [],[],[],[]
    for image_for_region, image_for_extract, image_org_size, image_id in data:
        image_for_region_list.append(image_for_region)
        image_for_extract_list.append(image_for_extract)
        image_org_size_list.append(image_org_size)
        image_id_list.append(image_id)

    return image_for_region_list, image_for_extract_list, image_org_size_list, image_id_list

def blip_collate_fn(data):
    image_list, image_org_size, output_targets, image_id_list = [],[],[],[]
    for image_item, size, targets_item, image_id in data:
        image_list.append(image_item)
        image_org_size.append(size)
        output_targets.append(targets_item)
        image_id_list.append(image_id)

    return image_list, image_org_size, output_targets, image_id_list

def blip_test_collate_fn(data):
    image_for_extract_list, tensor_list_overalllist, boxes_list_overalllist, image_id_list = [],[],[],[]
    for image_for_extract, tensor_list, boxes_list, image_id in data:
        tensor_list_overalllist.append(tensor_list)
        image_for_extract_list.append(image_for_extract)
        boxes_list_overalllist.append(boxes_list)
        image_id_list.append(image_id)

    return image_for_extract_list, tensor_list_overalllist, boxes_list_overalllist, image_id_list



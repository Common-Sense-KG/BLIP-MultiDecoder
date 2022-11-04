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

# from data.utils import pre_caption_dense
def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

class dense_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='an area of '):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        #url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'dense_train.json'
        
        #download_url(url,ann_root)#ann_root = annotation
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        tf_idf_name = 'dense_train_predicate_tfidf_score.json'
        self.corresponding_tf_idf = json.load(open(os.path.join(ann_root,tf_idf_name),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.tokenizer = init_tokenizer()
        
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
        
        ann = self.annotation[index]#根据index获取annotation
        match_tf_idf = self.corresponding_tf_idf[index]
        output_targets = {'boxes':torch.tensor([]),'caps':torch.tensor([]),'caps_len':torch.tensor([])}
        
        image_path = os.path.join(self.image_root,str(ann['image_id'])+'.jpg')        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)

        caption = ''
        captions_list = []
        boxes_list = []
        # tensor_list = []
        original_captions_list = []
        max_caption_num = 15
        signal = True
        while signal:
            if len(captions_list) >= len(ann['phrase_list']) * 0.6 :
                break
            for i,(phrase) in enumerate(ann['phrase_list']):
                original_captions_list.append(self.prompt + pre_caption(phrase['caption']))
                # above_predicate = '' if i==0 else match_tf_idf['predicate_tf_idf_score'][i-1]['predicate']
                tf_idf_score = match_tf_idf['predicate_tf_idf_score'][i]
                # additional_similar_values = 0.1 if tf_idf_score['predicate'] == above_predicate else 0
                if tf_idf_score['tf_idf_score']< 0.4 and random.uniform(tf_idf_score['tf_idf_score'], 1) < 0.55 :
                    continue
                caption = self.prompt + pre_caption(phrase['caption'])#最后一个句子后留有SEP
                captions_list.append(caption)#大小不一——进行补全
                phrase['boxes'] = [x + 1 for x in phrase['boxes']]
                boxes_list.append(phrase['boxes'])
                # tensor_list.append(np.array(phrase['tensor']))
                if len(captions_list) >= max_caption_num:
                    signal = False
                    break
        # with open('caption_compare.txt',"a") as f:    #设置文件对象
        #     f.write("before preocessing image {}\n".format(str(ann['image_id']))) 
        #     f.writelines(original_captions_list)
        #     f.write("after\n")  
        #     f.writelines(captions_list)

        # print("before preocessing image {}\n".format(str(ann['image_id']))) 
        # print(original_captions_list)
        # print("after\n")  
        # print(captions_list)

        truly_length = len(captions_list)
        if truly_length < max_caption_num:
            captions_list += [" " for i in range(max_caption_num - truly_length)]
            boxes_list += [[1,1,2,2] for i in range(max_caption_num - truly_length)]
            # tensor_list  += [np.zeros((1,577),np.float64) for j in range(max_caption_num - truly_length)]
        tokenize_result = self.tokenizer(captions_list,padding=True, return_tensors="pt")
        # output_targets['boxes'] = 
        output_targets['boxes'] = torch.tensor(boxes_list)
        output_targets['caps'] = tokenize_result.input_ids
        caps_len = []
        for i in range(tokenize_result.attention_mask.shape[0]):
            caps_len.append(torch.count_nonzero(tokenize_result.attention_mask[i]).item())

        output_targets['caps_len'] = torch.tensor(caps_len)
        #caption = self.prompt+pre_caption(ann['caption'], self.max_words) #此处待更改  pre_caption内部拼接与max_words
        # caption[-1] = '[EOS]'
        # caption = caption.strip('[PAD]')
        
        # return image, captions_list, output_targets, truly_length #,self.img_ids[ann['image_id']] 
        return image, output_targets, truly_length #,self.img_ids[ann['image_id']] 
    
    
class dense_eval(Dataset):
    def __init__(self, transform, image_root, ann_root):  
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
        
        ann = self.annotation[index]
        tensor_list = []
        
        image_path = os.path.join(self.image_root,str(ann['image_id'])+'.jpg')      
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  #modify
        image = Image.open(image_path) 
        # results = model(image) 
        image = image.convert('RGB')  
        image = self.transform(image)    
        for phrase in ann['phrase_list']:
            tensor_list.append(np.array(phrase['tensor']))    
              
        max_caption_num = 70
        truly_length = len(tensor_list)
        img_id = ann['image_id']
        tensor_list  += [np.zeros((1,577),np.float64) for j in range(max_caption_num - truly_length)]
        
        return image, tensor_list, int(img_id)   
        
    
    
class dense_test(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        # urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
        #         'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        # filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        # download_url(urls[split],ann_root)

        # filename = 'objects.json'
        # ann_root = '/home/hcui25/Research/BLIP-MultiDecoder/annotation/vg_org'
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
        # image = self.transform(image)  

        return image, int(self.annotation[index]['image_id'])

def blip_collate_fn(data):
    image_list, output_targets, truly_length = [],[],[]
    for image_item, targets_item, truly_length_item in data:
        image_list.append(image_item)
        # captions_list.append(captions_item)
        output_targets.append(targets_item)
        truly_length.append(truly_length_item)

    return image_list, output_targets, truly_length



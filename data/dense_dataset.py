# dataset for dense
import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption
import numpy as np
# from data.utils import pre_caption_dense

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
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
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
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,str(ann['image_id'])+'.jpg')        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)

        caption = ''
        captions_list = []
        tensor_list = []
        for phrase in ann['phrase_list']:
            caption = self.prompt + pre_caption(phrase['caption'])#最后一个句子后留有SEP
            captions_list.append(caption)#大小不一——进行补全
            tensor_list.append(np.array(phrase['tensor']))

        max_caption_num = 88
        truly_length = len(captions_list)
        captions_list += [" " for i in range(max_caption_num - truly_length)]
        tensor_list  += [np.zeros((1,577),np.float64) for j in range(max_caption_num - truly_length)]
        
        #caption = self.prompt+pre_caption(ann['caption'], self.max_words) #此处待更改  pre_caption内部拼接与max_words
        # caption[-1] = '[EOS]'
        # caption = caption.strip('[PAD]')
        
        return image, captions_list, tensor_list, self.img_ids[ann['image_id']] , truly_length
    
    
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
        image = Image.open(image_path).convert('RGB')   
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
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, int(self.annotation[index]['image_id'])
'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
from transformers import BertModel as OrgBertModel

import torch
from torch import tensor
from torch import nn as nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                 
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)  

        
    def forward(self, image, caption, mode):
        
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device) 
        
        if mode=='image':    
            # return image features
            image_embeds = self.visual_encoder(image)             
            return image_embeds
        
        elif mode=='text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)    
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)      
            
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state
        
        
        
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'an area of',
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config) #多头实现
        # self.text_decoder_list = []
        # for i in range(0,max_length):
        #     self.text_decoder = BertLMHeadModel(config=med_config)    
        #     self.text_decoder_list.append(self.text_decoder)
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-2

        
    def forward(self, image, max_caption_num, ground_truth):
        #逐一item拆解
    
        batch_size = len(image)#首先获取batch size
        loss_dict = {"ctploss":[], "regloss":[],"overallloss":[]} 
        for i in range(0,batch_size):
            image_embeds = self.visual_encoder(image[i].unsqueeze(0)) #1*（576+1）*768  24*24+全局 768为patch的representation的dimension

            crossentropy_loss_list = []

            all_encodings = ground_truth[i]['caps'].to(image[0].device) #tensor 15*13
            all_encodings[:,0] = self.tokenizer.bos_token_id
            decoder_targets = all_encodings.masked_fill(all_encodings == self.tokenizer.pad_token_id, -100)         
            decoder_targets[:,:self.prompt_length] = -100 # why -100

            # idx = 0
            for idx in range(all_encodings.shape[0]):
            # while idx < all_encodings.shape[0]:
                cap_mask = all_encodings[idx].masked_fill(all_encodings[idx] > 0, 1).to(image[0].device)
                decoder_output = self.text_decoder(all_encodings[idx].unsqueeze(0), #single caption token
                                               attention_mask = cap_mask.unsqueeze(0), #cap length
                                               encoder_hidden_states = image_embeds, #image / has grad
                                               encoder_attention_mask = ground_truth[i]['image_mask'][idx].unsqueeze(0).to(image[0].device),           
                                               labels = decoder_targets[idx].unsqueeze(0),#传入该pic对应的caption做crossEntropyloss
                                               return_dict = True, ) 
                # idx += 1


                if idx == 0:
                    crossentropy_loss = decoder_output.loss
                    prediction_res = torch.argmax(decoder_output.logits,dim=2)
                    # prediction_res_list.append(decoder_output.logits)#将预测结果加入到list
                else:
                    crossentropy_loss += decoder_output.loss
                    lg = torch.argmax(decoder_output.logits,dim=2)
                    prediction_res = torch.cat([prediction_res,lg],0)

                crossentropy_loss_list.append(decoder_output.loss)

            ctploss = crossentropy_loss / all_encodings.shape[0]
            loss_thisimg = crossentropy_loss / all_encodings.shape[0]
            loss_dict['ctploss'].append(ctploss)
            main_idx = 0
            # sum_reg_loss = torch.zeros_like(torch.ones([1,embeds.shape[1]])).to(image.device)
            sum_reg_loss = torch.tensor([0.00]).to(image[0].device)
            for main_idx in range(0,prediction_res.shape[0]):
                # print(main_idx)
                mainTensor = prediction_res[main_idx]#caption length 
                # for res_new in prediction_res_list[main_idx+1:]:
                for submain_idx in range(main_idx + 1 , prediction_res.shape[0]):
                    # print("inside")
                    # print(res_new)
                    submainTensor = prediction_res[submain_idx]
                    output = F.cosine_similarity(mainTensor.unsqueeze(0).float(),submainTensor.unsqueeze(0).float())
                    sum_reg_loss += output
            # sum_reg_loss = sum_reg_loss // (sum_reg_loss // crossentropy_loss)
            if prediction_res.shape[0] >= 2:
                loss_thisimg += sum_reg_loss.item() / (prediction_res.shape[0] * (prediction_res.shape[0] - 1) / 2 ) * 10
                loss_dict['regloss'].append(sum_reg_loss.item() / (prediction_res.shape[0] * (prediction_res.shape[0] - 1) / 2 ))
            else:
                loss_thisimg += sum_reg_loss.item() 
                loss_dict['regloss'].append(sum_reg_loss.item())

            loss_dict['overallloss'].append(loss_thisimg)

        return loss_dict
        
    def generate(self, image, tensor_list, device ='cuda', decoder_num=15, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image) # image size为384代表传入image为384*384 输出为577*768

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long)
        model_kwargs = {"encoder_hidden_states": image_embeds}
        # model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":tensor_list}
        
        prompt = ['an area of '] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 
        captions = []   
        
        if sample:
            #nucleus sampling // #can try
            for i in range(0,decoder_num):
                outputs = self.text_decoder.generate(input_ids=input_ids,
                                                    max_length=max_length,
                                                    min_length=min_length,
                                                    do_sample=True,
                                                    top_p=top_p,
                                                    num_return_sequences=1,
                                                    eos_token_id=self.tokenizer.sep_token_id,
                                                    pad_token_id=self.tokenizer.pad_token_id, 
                                                    repetition_penalty=1.1, 
                                                    output_scores = True,                                           
                                                    **model_kwargs)
                for output in outputs:
                    caption = self.tokenizer.decode(output, skip_special_tokens=True)    
                    captions.append(caption[len(self.prompt):])

        else:
            # beam search
            for idx in range(0,tensor_list.shape[0]):
                outputs = self.text_decoder.generate(input_ids=input_ids.to(device),
                                                    max_length=max_length,
                                                    min_length=min_length,
                                                    num_beams=num_beams,
                                                    eos_token_id=self.tokenizer.sep_token_id,
                                                    pad_token_id=self.tokenizer.pad_token_id,     
                                                    repetition_penalty=repetition_penalty,
                                                    output_scores = True,
                                                    encoder_attention_mask=tensor_list[idx].unsqueeze(0).repeat_interleave(num_beams,dim=0),
                                                    **model_kwargs)
                for output in outputs:
                    caption = self.tokenizer.decode(output, skip_special_tokens=True)
                    captions.append(caption[len(self.prompt):]) 

        return captions

    def textGenerate(self, image_embeds, text_list, device ='cuda'):
        #new text generate in region_detection model based on forward
        image_embeds = image_embeds.view(image_embeds.size(0),1,256,-1)
        batch_size = len(text_list)#首先获取batch size
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)
        # loss_dict = {"ctploss":[], "regloss":[],"overallloss":[]} 
        overall_idx = 0
        ce_loss = 0.0
        all_caption = []
        for i in range(0,batch_size):

            crossentropy_loss_list = []
            caption_list = []

            all_encodings = text_list[i]
            all_encodings[:,0] = self.tokenizer.bos_token_id
            decoder_targets = all_encodings.masked_fill(all_encodings == self.tokenizer.pad_token_id, -100)         
            decoder_targets[:,:self.prompt_length] = -100 # why -100

            # idx = 0
            # while idx < all_encodings.shape[0]:
            for idx in range(all_encodings.shape[0]):
                cap_mask = all_encodings[idx].masked_fill(all_encodings[idx] > 0, 1).to(image_embeds.device)
                decoder_output = self.text_decoder(all_encodings[idx].unsqueeze(0), 
                                               attention_mask = cap_mask.unsqueeze(0),
                                               encoder_hidden_states = image_embeds[overall_idx],
                                               encoder_attention_mask = image_atts[overall_idx],           
                                               labels = decoder_targets[idx].unsqueeze(0),#传入该pic对应的所有caption逐一做crossEntropyloss 是否正确？
                                               return_dict = True, ) 
                # idx += 1
                overall_idx += 1
                encoding_caption = torch.argmax(decoder_output.logits,dim=2).squeeze(0)
                text_caption = self.tokenizer.decode(encoding_caption, skip_special_tokens=True)
                caption_list.append({'caption_encodings':encoding_caption,'caption_text':text_caption,'gt_text':self.tokenizer.decode(all_encodings[idx-1], skip_special_tokens=True)})


                if idx == 0:
                    crossentropy_loss = decoder_output.loss
                    prediction_res = torch.argmax(decoder_output.logits,dim=2)
                    # prediction_res_list.append(decoder_output.logits)#将预测结果加入到list
                else:
                    crossentropy_loss += decoder_output.loss
                    lg = torch.argmax(decoder_output.logits,dim=2)
                    prediction_res = torch.cat([prediction_res,lg],0)

                crossentropy_loss_list.append(decoder_output.loss)

            ce_loss += crossentropy_loss / all_encodings.shape[0]
            all_caption.append(caption_list)


        return ce_loss,all_caption
    
        

def blip_decoder(pretrained='',**kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model    
    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model        

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

def init_tokenizer_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer,model

def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
    

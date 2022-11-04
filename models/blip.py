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
                 prompt = 'an area of ',
                 max_length = 5,#理论上是88，但启动太慢，且后半部分多为闲置
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
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        
    def forward(self, image, caption, tensor_list, max_caption_num, caption_actual_num):
        #逐一item拆解
        batch_size = image.shape[0]#首先获取batch size
        img_list = image.chunk(batch_size,dim=0)
        # loss_overall = torch.zeros_like(1)
        loss_list = []
        ctploss_list = []
        regloss_list = []
        
        for i in range(0,batch_size):
            
            # prediction_res_list = []
            image_embeds = self.visual_encoder(img_list[i]) #1*（576+1）*768  24*24+全局 768为patch的representation的dimension
            #image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)#1*577 #image embed需要改，要切一部分
            # all_decoder_targets = []
            all_org_text = []
            all_text = []

            idx = 0
            for content in caption:#先将当前item的 所有caption 转为decoder_targets 与 text
                if idx >= min(caption_actual_num[i].item(),max_caption_num):
                    break#防止越界
                all_org_text.append(content[i])
                idx += 1

            # print(all_org_text)
            if all_org_text == []:
                continue 
            all_encodings = self.tokenizer(all_org_text,padding=True, return_tensors="pt").to(image.device) 
            all_encodings['input_ids'][:,0] = self.tokenizer.bos_token_id
            decoder_targets = all_encodings.input_ids.masked_fill(all_encodings.input_ids == self.tokenizer.pad_token_id, -100)         
            decoder_targets[:,:self.prompt_length] = -100 # why -100
                # # text = self.tokenizer(content[i], padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
                # text = self.tokenizer(content[i], padding=True, return_tensors="pt").to(image.device) 
                # #self.tokenizer.vocab_size为30522
                # text.input_ids[:,0] = self.tokenizer.bos_token_id#最前面一位为[DEC]
        
                # decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
                # decoder_targets[:,:self.prompt_length] = -100

                # all_decoder_targets.append(decoder_targets)#最前面一位为-100 最后面一位为[CLS] 
                # all_text.append(text) #最前面一位为[DEC] 最后面一位为[CLS] 
                # idx += 1

            # lr = nn.Linear(1024,self.tokenizer.vocab_size+2,device=image.device)

            # model = OrgBertModel.from_pretrained('bert-large-cased')
            # model.resize_token_embeddings(len(self.tokenizer)) 
            # model = model.eval()
            # model = model.to(image.device)

            # with torch.no_grad():
            #     # idx = 0
            #     # while idx < caption_actual_num[i].item():
            #     #     all_text[idx].embeds = model(**all_text[idx])[0]
            #     #     idx += 1
            #     embeds = modelnew(**all_encodings)
            #     embeds = lr(embeds[0])
                

            # print(embeds.shape)

            idx = 0
            while idx < min(caption_actual_num[i].item(),max_caption_num):
            # for content_new in caption:
            #     if idx >= caption_actual_num[i].item():
            #         break#防止越界
            # text = self.tokenizer(content, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        
            # text.input_ids[:,0] = self.tokenizer.bos_token_id
        
            # decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
            # decoder_targets[:,:self.prompt_length] = -100
            # now_text_decoder = self.text_decoder_list[i]
                # decoder_output = self.text_decoder(all_text[idx].input_ids, 
                #                                attention_mask = all_text[idx].attention_mask, 
                #                                encoder_hidden_states = image_embeds,
                #                                encoder_attention_mask = image_atts,                  
                #                                labels = embeds,#传入该pic对应的所有caption逐一做crossEntropyloss
                #                                return_dict = True, ) 
                decoder_output = self.text_decoder(all_encodings['input_ids'][idx].unsqueeze(0), 
                                               attention_mask = all_encodings['attention_mask'][idx].unsqueeze(0), 
                                               encoder_hidden_states = image_embeds,
                                               encoder_attention_mask = tensor_list[idx][i].to(image.device),                  
                                               labels = decoder_targets[idx].unsqueeze(0),#传入该pic对应的所有caption逐一做crossEntropyloss 是否正确？
                                               return_dict = True, ) 
                  
                idx += 1

                if idx == 1:
                    crossentropy_loss = decoder_output.loss.to(image.device)
                    prediction_res = torch.argmax(decoder_output.logits,dim=2).to(image.device)
                    #prediction_res_list.append(decoder_output.logits)#将预测结果加入到list
                else:
                    crossentropy_loss += decoder_output.loss
                    lg = torch.argmax(decoder_output.logits,dim=2).to(image.device)
                    prediction_res = torch.cat([prediction_res,lg],0)

                # prediction_res_list.append(decoder_output.logits[:,-1,:])#将最后一个token（即cls加入到list）
            # loss_overall += decoder_output.loss
            #prediction_res seq_num * seq_length
            #原来：直接拿predict结果Logits算相似度
            # loss_thisimg = crossentropy_loss / min(caption_actual_num[i].item(),max_caption_num)
            # main_idx = 0
            # sum_reg_loss = torch.zeros_like(torch.ones([1,embeds.shape[1]])).to(image.device)
            # for main_idx in range(0,prediction_res.shape[0]):
            #     # print(main_idx)
            #     mainTensor = prediction_res[main_idx].to(image.device)#caption length * vocab size
            #     # for res_new in prediction_res_list[main_idx+1:]:
            #     for submain_idx in range(main_idx+1 , prediction_res.shape[0]):
            #         # print("inside")
            #         # print(res_new)
            #         output = F.cosine_similarity(mainTensor,prediction_res[submain_idx])
            #         sum_reg_loss += output#sum_reg_loss 1*caption length
            # loss_thisimg += sum_reg_loss.sum()/embeds.shape[1]
            # loss_list.append(loss_thisimg.to(image.device))

            ctploss_thisimg = crossentropy_loss / min(caption_actual_num[i].item(),max_caption_num)
            ctploss_list.append(ctploss_thisimg.to(image.device))
            loss_thisimg = crossentropy_loss / min(caption_actual_num[i].item(),max_caption_num)
            main_idx = 0
            # sum_reg_loss = torch.zeros_like(torch.ones([1,embeds.shape[1]])).to(image.device)
            sum_reg_loss = torch.tensor([0.00]).to(image.device)
            for main_idx in range(0,prediction_res.shape[0]):
                # print(main_idx)
                mainTensor = prediction_res[main_idx].to(image.device)#caption length 
                # for res_new in prediction_res_list[main_idx+1:]:
                for submain_idx in range(main_idx+1 , prediction_res.shape[0]):
                    # print("inside")
                    # print(res_new)
                    output = F.cosine_similarity(mainTensor.unsqueeze(0).float(),prediction_res[submain_idx].unsqueeze(0).float())
                    sum_reg_loss += output#1
            # sum_reg_loss = sum_reg_loss // (sum_reg_loss // crossentropy_loss)
            if prediction_res.shape[0] >= 2:
                loss_thisimg += sum_reg_loss.item() / (prediction_res.shape[0] * (prediction_res.shape[0] - 1) / 2 ) * 6
                regloss_list.append(sum_reg_loss.item() / (prediction_res.shape[0] * (prediction_res.shape[0] - 1) / 2 ))
            loss_list.append(loss_thisimg.to(image.device))
            # if i == 0:
            #     loss_overall = loss_thisimg
            # else:
            #     loss_overall = torch.cat([loss_overall])

        
        return loss_list,ctploss_list,regloss_list
        
    def generate(self, image, tensor_list, decoder_num=15,sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds}
        
        prompt = ['an area of '] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
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
            #beam search
            for i in range(0,decoder_num):
                outputs = self.text_decoder.generate(input_ids=input_ids,
                                                    max_length=max_length,
                                                    min_length=min_length,
                                                    num_beams=num_beams,
                                                    eos_token_id=self.tokenizer.sep_token_id,
                                                    pad_token_id=self.tokenizer.pad_token_id,     
                                                    repetition_penalty=repetition_penalty,
                                                    output_scores = True,
                                                    encoder_attention_mask=tensor_list[i].repeat_interleave(num_beams,dim=0).to(image.device),
                                                    **model_kwargs)  
                for output in outputs:
                    caption = self.tokenizer.decode(output, skip_special_tokens=True)    
                    captions.append(caption[len(self.prompt):])          
        return captions
    

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
    

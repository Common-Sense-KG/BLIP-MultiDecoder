import json
import random
import math
# import spacy
# from itertools import combinations
import os
from PIL import Image
from torch import tensor
import torch
import numpy as np
from annotation.tf_idf_preprocess2 import tf_idf_process
org_data = 'relational_captions.json'

# def pre_process(titles):
#     """
#     Pre-processes titles by removing stopwords and lemmatizing text.
#     :param titles: list of strings, contains target titles,.
#     :return: preprocessed_title_docs, list containing pre-processed titles.
#     """

#     # Preprocess all the titles
#     title_docs = [nlp(x) for x in titles]
#     preprocessed_title_docs = []
#     lemmatized_tokens = []
#     for title_doc in title_docs:
#         for token in title_doc:
#             if not token.is_stop:
#                 lemmatized_tokens.append(token.lemma_)
#         preprocessed_title_docs.append(" ".join(lemmatized_tokens))
#         del lemmatized_tokens[
#             :
#             ]  # empty the lemmatized tokens list as the code moves onto a new title

#     return preprocessed_title_docs

# def similarity_filter(titles):
#     """
#     Recursively check if titles pass a similarity filter.
#     :param titles: list of strings, contains titles.
#     If the function finds titles that fail the similarity test, the above param will be the function output.
#     :return: this method upon itself unless there are no similar titles; in that case the feed that was passed
#     in is returned.
#     """

#     # Preprocess titles
#     preprocessed_title_docs = pre_process(titles)

#     # Remove similar titles
#     all_summary_pairs = list(combinations(preprocessed_title_docs, 2))
#     similar_titles = []
#     for pair in all_summary_pairs:
#         title1 = nlp(pair[0])
#         title2 = nlp(pair[1])
#         similarity = title1.similarity(title2)
#         if similarity > 0.8:
#             similar_titles.append(pair)

#     titles_to_remove = []
#     for a_title in similar_titles:
#         # Get the index of the first title in the pair
#         index_for_removal = preprocessed_title_docs.index(a_title[0])
#         titles_to_remove.append(index_for_removal)

#     # Get indices of similar titles and remove them
#     similar_title_counts = set(titles_to_remove)
#     similar_titles = [
#         x[1] for x in enumerate(titles) if x[0] in similar_title_counts
#     ]

#     # Exit the recursion if there are no longer any similar titles
#     if len(similar_title_counts) == 0:
#         return titles

#     # Continue the recursion if there are still titles to remove
#     else:
#         # Remove similar titles from the next input
#         for title in similar_titles:
#             idx = titles.index(title)
#             titles.pop(idx)
            
#         return similarity_filter(titles)
def getMaxshape(object_info,subject_info):
    min_x = object_info['x'] if object_info['x'] <= subject_info['x'] else subject_info['x']
    min_y = object_info['y'] if object_info['y'] <= subject_info['y'] else subject_info['y']
    max_x = object_info['x'] + object_info['w'] if object_info['x'] + object_info['w'] >= subject_info['x'] + subject_info['w']  else subject_info['x'] + subject_info['w']
    max_y = object_info['y'] + object_info['h'] if object_info['y'] + object_info['h'] >= subject_info['y'] + subject_info['h']  else subject_info['y'] + subject_info['h']
    return np.array([min_x,min_y,max_x,max_y]).tolist()

def judgeNotTheSame(phrase,now_phrase_list):
    for now_phrase in now_phrase_list:
        if phrase == now_phrase['caption']:
            return False

    return True

def delete_duplicate(data):
    immutable_dict = set([str(item) for item in data])
    data = [eval(i) for i in immutable_dict]
    return data

def max(a,b):
    if a>=b:
        return a
    else:
        return b

def min(a,b):
    if a<=b:
        return a
    else:
        return b

def squareOverlap(x1,y1,w1,h1,x2,y2,w2,h2):
    endx = max(x1+w1,x2+w2)
    startx = min(x1,x2)
    width = w1+w2-(endx-startx)

    endy = max(y1+h1,y2+h2)
    starty = min(y1,y2)
    height = h1+h2 -(endy-starty)

    if width <= 0 or height <=0:
        ratio1 = 0 
    else:
        area = width * height
        area1 = w1 * h1
        if area1 == 0:
            ratio1 = 0
        else:
            ratio1 = area/area1

        area2 = w2 * h2
        if area2 == 0:
            ratio2 = 0
        else:
            ratio2 = area/area2
    
    if ratio1 == 0:
        return 0
    elif ratio1 < 0.2 and ratio2 < 0.2:
        return 0
    else: 
        return 1
    

def getPicSize(path,img_id):
    path = os.path.join(path,str(img_id)+".jpg")
    img = Image.open(path)
    return img.size[0],img.size[1]

def getImgEmbedAccordingtoPos(PicW,PicH,x1,y1,w1,h1,x2,y2,w2,h2):
    pixelW = PicW / 24
    pixelH = PicH / 24
    empty = torch.zeros(1,577)
    for i in range(0,576):
        x = i % 24 * pixelW
        y = i // 24 * pixelH
        if squareOverlap(x,y,pixelW,pixelH,x2,y2,w2,h2) or squareOverlap(x,y,pixelW,pixelH,x1,y1,w1,h1):
            empty[0][i] = 1
            # print(i)
    empty[0][576] = 1
    # print(empty)
    return empty

if __name__ == '__main__':
    # nlp = spacy.load("en_core_web_md")
    now_directory = '/local/scratch/hcui25/data/VisualGenomeData'
    with open(os.path.join(now_directory,org_data),'r') as load_org:
        load_dict=json.load(load_org)

    new_data={}
    new_data_list=[]
    for content in load_dict:
        relationships=content['relationships']
        now_img_id=content['image_id']
        width, height = getPicSize('/local/scratch/hcui25/data/VisualGenomeData/VG_100K_2',now_img_id)
        now_phrase_list=[]
        new_data={}
        if len(relationships) == 0:
            continue
        for relation in relationships:
            phrase = relation['phrase'].lower()
            predicate = relation['predicate'].lower()
            shape = getMaxshape(relation['object'],relation['subject'])
            # phrase = relation['phrase'].lower()
            #tensor = getImgEmbedAccordingtoPos(width,height,relation['object']['x'],relation['object']['y'],relation['object']['w'],relation['object']['h'],relation['subject']['x'],relation['subject']['y'],relation['subject']['w'],relation['subject']['h'])
            # new_phrase_with_tensor={'caption':phrase,'tensor':tensor}
            if judgeNotTheSame(phrase,now_phrase_list):#实现相同caption去重
                tensor = getImgEmbedAccordingtoPos(width,height,relation['object']['x'],relation['object']['y'],relation['object']['w'],relation['object']['h'],relation['subject']['x'],relation['subject']['y'],relation['subject']['w'],relation['subject']['h'])
                new_phrase_with_tensor={'caption':phrase,'tensor':tensor.numpy().tolist(),'predicate':predicate,'boxes':shape}#更精细化
                now_phrase_list.append(new_phrase_with_tensor)

        # similarity_filter(now_phrase_list)
        new_data={'image_id':now_img_id,'phrase_list':now_phrase_list}
        new_data_list.append(new_data)

    random.shuffle(new_data_list)

    dense_data_train=new_data_list[0:math.floor(len(new_data_list)*0.6*0.5)]
    dense_data_eval=new_data_list[math.floor(len(new_data_list)*0.6*0.5):math.floor(len(new_data_list)*0.6*0.5)+math.floor(len(new_data_list)*0.2*0.5)]
    dense_data_test=new_data_list[math.floor(len(new_data_list)*0.6*0.5)+math.floor(len(new_data_list)*0.2*0.5):math.floor(len(new_data_list)*0.5)]
    print("======dense data prepare finish======")

    
    # dense_train_new = []
    # for content in dense_data_train:
    #     image_id=content['image_id']
    #     phrase_list=content['phrase_list']
    #     phrase_length=len(phrase_list)#共有多少个caption
    #     if phrase_length == 0:
    #         continue
    #     width,height = getPicSize('/archive/hot0/fxy/BLIP/data/dense/VG_100K_2',image_id)
    #     for caption in phrase_list:

    #     new_data = {"image":image_id,"caption":phrase_list}
    #     dense_train_new.append(new_data)

    with open("dense_train.json","w") as f1:
        json.dump(dense_data_train,f1)

    print("======dense train data prepare finish======")
    dense_eval_new = []
    dense_eval_gt = {'annotations':[],'images':[]}
    caption_idx = 1
    for content in dense_data_eval:
        image_id=content['image_id']
        phrase_list=content['phrase_list']
        phrase_length=len(phrase_list)#共有多少个caption
        # if phrase_length == 0:
        #     continue
        # phrase_list = delete_duplicate(phrase_list)
        # new_data={"image":image_id,"caption":phrase_list}   
        # dense_eval_new.append(new_data)#去除原eval数据集中caption为空的部分

        dense_eval_gt['images'].append({'id':image_id})
        if phrase_length >= 5:
            ran = random.sample(range(0,phrase_length),5)
            for i in ran:
                dense_eval_gt['annotations'].append({'image_id':image_id,"caption":phrase_list[i]['caption'],"tensor":phrase_list[i]['tensor'],"id":caption_idx,"boxes":phrase_list[i]['boxes']})
                caption_idx += 1
        else:
            num = 0
            while num < 5:
                i = random.randint(0,phrase_length-1)
                dense_eval_gt['annotations'].append({'image_id':image_id,"caption":phrase_list[i]['caption'],"tensor":phrase_list[i]['tensor'],"id":caption_idx,"boxes":phrase_list[i]['boxes']})
                caption_idx += 1
                num += 1


    with open("dense_eval.json","w") as f2:
        json.dump(dense_data_eval,f2)
    
    with open("dense_eval_gt.json","w") as f3:
        json.dump(dense_eval_gt,f3)
    print("======dense eval data stored finish======")

    dense_test_new = []
    dense_test_gt = {'annotations':[],'images':[]}
    caption_idx = 1
    for content in dense_data_test:
        image_id=content['image_id']
        phrase_list=content['phrase_list']
        phrase_length = len(phrase_list)#共有多少个caption
        # if phrase_length == 0:
        #     continue
        # phrase_list = delete_duplicate(phrase_list)
        # new_data={"image":image_id,"caption":phrase_list}   
        # dense_test_new.append(new_data)#去除原eval数据集中caption为空的部分

        dense_test_gt['images'].append({'id':image_id})
        if phrase_length >=5 :
            ran = random.sample(range(0,phrase_length),5)
            for i in ran:
                dense_test_gt['annotations'].append({'image_id':image_id,"caption":phrase_list[i]['caption'],"tensor":phrase_list[i]['tensor'],"id":caption_idx,"boxes":phrase_list[i]['boxes']})
                caption_idx += 1
        else:
            num = 0
            while num < 5:
                i =random.randint(0,phrase_length-1)
                dense_test_gt['annotations'].append({'image_id':image_id,"caption":phrase_list[i]['caption'],"tensor":phrase_list[i]['tensor'],"id":caption_idx,"boxes":phrase_list[i]['boxes']})
                caption_idx += 1
                num += 1


    with open("dense_test.json","w") as f4:
        json.dump(dense_data_test,f4)
    
    with open("dense_test_gt.json","w") as f5:
        json.dump(dense_test_gt,f5)

    print("======dense test data prepare finish======")

    tf_idf_process()
    



# # Set globals
# nlp = spacy.load("en_core_web_md")

# 

# if __name__ == "__main__":
#     with open("dense_train.json",'r') as load_org1:
#         load_dict=json.load(load_org1)
#     max_len = 0
#     dense_train_new = []
#     for content in load_dict:
#         image_id=content['image']
#         caption = content['caption']
#         caption = list(set(caption))
#         # similarity_filter(caption)
#         # print("after",caption)
#         if len(caption)>max_len:
#             max_len = len(caption)
#         new_data = {"image":image_id,"caption":caption}
#         dense_train_new.append(new_data)

#     print("max length for train: ",max_len)
#     with open("dense_train.json","w") as f1:
#         json.dump(dense_train_new,f1)

#     with open("dense_eval.json",'r') as load_org1:
#         load_dict=json.load(load_org1)
#     max_len = 0
#     dense_eval_new = []
#     for content in load_dict:
#         image_id=content['image']
#         caption = content['caption']
#         caption = list(set(caption))
#         # similarity_filter(caption)
#         # print("after",caption)
#         if len(caption)>max_len:
#             max_len = len(caption)
#         new_data = {"image":image_id,"caption":caption}
#         dense_eval_new.append(new_data)
        
#     print("max length for eval: ",max_len)
#     with open("dense_eval.json","w") as f1:
#         json.dump(dense_eval_new,f1)

#     with open("dense_test.json",'r') as load_org1:
#         load_dict=json.load(load_org1)
#     max_len = 0
#     dense_test_new = []
#     for content in load_dict:
#         image_id=content['image']
#         caption = content['caption']
#         caption = list(set(caption))
#         # similarity_filter(caption)
#         # print("after",caption)
#         if len(caption)>max_len:
#             max_len = len(caption)
#         new_data = {"image":image_id,"caption":caption}
#         dense_test_new.append(new_data)
        
#     print("max length for test: ",max_len)
#     with open("dense_test.json","w") as f1:
#         json.dump(dense_test_new,f1)
    
    




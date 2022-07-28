import json
import random
import math

if __name__ == '__main__':
    with open("relational_captions.json",'r') as load_org:
        load_dict=json.load(load_org)

    new_data={}
    new_data_list=[]
    for content in load_dict:
        relationships=content['relationships']
        now_phrase_list=[]
        new_data={}
        for relation in relationships:
            #phrase=relation['phrase'].lower
            now_phrase_list.append(relation['phrase'].lower())
        now_img_id=content['image_id']
        new_data={'image_id':now_img_id,'phrase_list':now_phrase_list}
        new_data_list.append(new_data)

    random.shuffle(new_data_list)

    dense_data_train=new_data_list[0:math.floor(len(new_data_list)*0.6)]
    dense_data_eval=new_data_list[math.floor(len(new_data_list)*0.6):math.floor(len(new_data_list)*0.6)+math.floor(len(new_data_list)*0.2)]
    dense_data_test=new_data_list[math.floor(len(new_data_list)*0.6)+math.floor(len(new_data_list)*0.2):-1]

    with open("dense_train.json","w") as f1:
        json.dump(dense_data_train,f1)

    
    dense_eval_new = []
    dense_eval_gt = {'annotations':[],'images':[]}
    caption_idx = 1
    for content in dense_data_eval:
        image_id=content['image_id']
        phrase_list=content['phrase_list']
        phrase_length=len(phrase_list)#共有多少个caption
        if phrase_length == 0:
            continue
        new_data={"image":image_id,"caption":phrase_list}   
        dense_eval_new.append(new_data)#去除原eval数据集中caption为空的部分

        dense_eval_gt['images'].append({'id':image_id})
        if phrase_length >= 5:
            ran = random.sample(range(0,phrase_length),5)
            for i in ran:
                dense_eval_gt['annotations'].append({'image_id':image_id,"caption":phrase_list[i],"id":caption_idx})
                caption_idx += 1
        else:
            num = 0
            while num < 5:
                i = random.randint(0,phrase_length-1)
                dense_eval_gt['annotations'].append({'image_id':image_id,"caption":phrase_list[i],"id":caption_idx})
                caption_idx += 1
                num += 1


    with open("dense_eval.json","w") as f2:
        json.dump(dense_eval_new,f2)
    
    with open("dense_eval_gt.json","w") as f3:
        json.dump(dense_eval_gt,f3)


    dense_test_new = []
    dense_test_gt = {'annotations':[],'images':[]}
    caption_idx = 1
    for content in dense_data_test:
        image_id=content['image_id']
        phrase_list=content['phrase_list']
        phrase_length=len(phrase_list)#共有多少个caption
        if phrase_length == 0:
            continue
        new_data={"image":image_id,"caption":phrase_list}   
        dense_test_new.append(new_data)#去除原eval数据集中caption为空的部分

        dense_test_gt['images'].append({'id':image_id})
        if phrase_length >=5 :
            ran = random.sample(range(0,phrase_length),5)
            for i in ran:
                dense_test_gt['annotations'].append({'image_id':image_id,"caption":phrase_list[i],"id":caption_idx})
                caption_idx += 1
        else:
            num = 0
            while num < 5:
                i =random.randint(0,phrase_length-1)
                dense_test_gt['annotations'].append({'image_id':image_id,"caption":phrase_list[i],"id":caption_idx})
                caption_idx += 1
                num += 1


    with open("dense_test.json","w") as f4:
        json.dump(dense_test_new,f4)
    
    with open("dense_test_gt.json","w") as f5:
        json.dump(dense_test_gt,f5)





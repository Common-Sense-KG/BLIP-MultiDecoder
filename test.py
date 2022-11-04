import torch
import cv2
from PIL import Image
import os
# from easycv.predictors import TorchYoloXPredictor


# model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
# # image = torch.rand(2,3,384,384)
# # results = model(image)
# image_path = os.path.join('/local/scratch/hcui25/data/VisualGenomeData/VG_100K_2/', '2325799.jpg') 
# image = Image.open(image_path) 
# model.imgsz = [800,600]
# model.conf = 0.1
# model.iou = 0.9
# model.augment = True
# results = model(image) 

# # results.print() 
# print(results) 
# print(results.pandas().xyxy[0])
# results.save() 


# output_ckpt = 'export_blade/epoch_300_pre_notrt.pt.blade'
# detector = TorchYoloXPredictor(output_ckpt,use_trt_efficientnms=False)

# image_path = os.path.join('/local/scratch/hcui25/data/VisualGenomeData/VG_100K_2/', '186.jpg') 
# # image = Image.open(image_path) 
# img = cv2.imread(image_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# output = detector.predict([img])
# print(output)

# # visualize image
# image = img.copy()
# for box, cls_name in zip(output[0]['detection_boxes'], output[0]['detection_class_names']):
#     # box is [x1,y1,x2,y2]
#     box = [int(b) for b in box]
#     image = cv2.rectangle(image, tuple(box[:2]), tuple(box[2:4]), (0,255,0), 2)
#     cv2.putText(image, cls_name, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

# cv2.imwrite('result.jpg',image)
# # model = torch.load('/home/hcui25/Research/BLIP-MultiDecoder/model_final_f05665.pkl')


# # model.eval()
# # result = model(image)

from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import os

image_path = os.path.join('/local/scratch/hcui25/data/VisualGenomeData/VG_100K_2/', '2325799.jpg') 
image = Image.open(image_path) 

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    # let's only keep detections with score > 0.9
    if score > 0.9:
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence
import numpy as np

from torchvision.ops import boxes as box_ops
from torchvision.models.detection import _utils as det_utils
from models.blip import blip_decoder
# from densecap import FastRCNNPredictor

def detect_loss_old(box_regression, regression_targets):   
    # box_regression  #61*4 251+272+220+271
    proposal_num = box_regression.shape[0]
    reg_tar = torch.cat(regression_targets,0)
    box_loss = 0.0
    box_loss += F.smooth_l1_loss(box_regression, reg_tar, reduction='sum')
    box_loss = box_loss / proposal_num

    return box_loss

def detect_loss(box_regression, regression_targets,text_output_list,corr_region_cap_list):
    # box_regression  #61*4 251+272+220+271
    proposal_num = box_regression.shape[0]
    reg_tar = torch.cat(regression_targets,0)
    box_loss = 0.0
    box_loss += F.smooth_l1_loss(box_regression, reg_tar, reduction='sum')
    box_loss = box_loss / proposal_num

    text_loss = 0.0
    start_idx = 0
    for region_caption_relation in corr_region_cap_list:
        cap_num = region_caption_relation.shape[0]
        cap_align_length = region_caption_relation.shape[1]
        generate_text = torch.t(pad_sequence([i.squeeze(0) for i in text_output_list[start_idx:start_idx+cap_num]]))
        start_idx += cap_num
        hinge_loss = nn.HingeEmbeddingLoss(margin=0.2)#image caption的loss
        differ_length = cap_align_length - generate_text.shape[1]
        if differ_length != 0:
            padding_tensor = torch.zeros((cap_num,differ_length)).to(generate_text.device)
            generate_text = torch.cat((generate_text, padding_tensor),dim=1)
        text_loss += hinge_loss(generate_text.float(),region_caption_relation.float())
    
    text_loss = text_loss / (len(text_output_list) * 100)

    return box_loss,text_loss


def caption_loss(caption_predicts, caption_gt, caption_length):
    """
    Computes the loss for caption part.
    Arguments:
        caption_predicts (Tensor)
        caption_gt (Tensor or list[Tensor])
        caption_length (Tensor or list[Tensor])
        caption_loss (Tensor)
    """

    if isinstance(caption_gt, list) and isinstance(caption_length, list):
        # caption_gt = torch.cat(caption_gt, dim=0)  # (batch_size, max_len+1)
        cap_gt_list = []
        for cap_gt_item in caption_gt:
            cap_gt_list += list(torch.chunk(cap_gt_item, cap_gt_item.shape[0], dim = 0))
# all_list = [item.squeeze(0) for item in all_list]
        caption_gt = pad_sequence([cap_gt_item.squeeze(0) for cap_gt_item in cap_gt_list], batch_first=True)
        caption_length = torch.cat(caption_length, dim=0) # (batch_size, )
        assert caption_predicts.shape[0] == caption_gt.shape[0] and caption_predicts.shape[0] == caption_length.shape[0]

    # '<bos>' is not considered
    caption_length = torch.clamp(caption_length-1, min=0)

    predict_pps = pack_padded_sequence(caption_predicts, caption_length, batch_first=True, enforce_sorted=False)

    target_pps = pack_padded_sequence(caption_gt[:, 1:], caption_length, batch_first=True, enforce_sorted=False)

    return F.cross_entropy(predict_pps.data, target_pps.data)


class DenseCapRoIHeads(nn.Module):

    def __init__(self,
                 box_describer,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Whether return features during testing
                 return_features=False,
                 ):

        super(DenseCapRoIHeads, self).__init__()

        self.return_features = return_features
        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor
        # self.box_predictor_new = FastRCNNPredictor(256*16,1)
        self.box_describer = box_describer

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.feature_transform_1 = nn.Linear(16,256)
        self.feature_transform_2 = nn.Linear(256,768)
        self.feature_transform_3 = nn.Linear(768,16)

        self.textModel = blip_decoder(pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth',vit = 'base',vit_grad_ckpt = False, vit_ckpt_layer = 0, prompt = '')#image_size = 32?
        # self.textModel.eval()


    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        """
        Calculate IOU and filter the proposal.
        Arguments:
            proposals (Tensor or list[Tensor]) [proposal+gt_boxes,4]
            gt_boxes (Tensor or list[Tensor])
            gt_labels (Tensor or list[Tensor]) [all 1 tensor with the length of gt_boxes_num]
        """
        matched_idxs = []
        labels = []
        bbx_score_all = [[]] * len(proposals)
        for img_idx, (proposals_in_image, gt_boxes_in_image, gt_labels_in_image) in enumerate(zip(proposals, gt_boxes, gt_labels)):  # 每张图片循环
            bbx_score = []
            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #calculate iou 
                #iou (Tensor[N, M]): the NxM matrix containing the IoU values for every element in boxes1 and boxes2
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                #get truly proposals num
                actual_size = proposals_in_image.shape[0] - gt_boxes_in_image.shape[0]
                for i in range(match_quality_matrix.shape[0]):
                    #Gets the best match for each groundtruth region box
                    bbx_score.append({'corresponding_region_index':i,'gt_box':gt_boxes_in_image[i],'score':match_quality_matrix[i][torch.argmax(match_quality_matrix[i][:actual_size])],'region':proposals_in_image[torch.argmax(match_quality_matrix[i][:actual_size])]})
                
                
                # temp = match_quality_matrix.t()[:actual_size]#取每个proposal框的max
                # for i in range(temp.shape[0]):
                #     if temp[i][torch.argmax(temp[i])] >= 0.7: # rpn网络中只有 > 0.7会被设为正样本
                #         bbx_score.append({'corresponding_region_index':torch.argmax(temp[i]).item(),'gt_box':gt_boxes_in_image[torch.argmax(temp[i])],'score':temp[i][torch.argmax(temp[i])],'region':proposals_in_image[i]})   
                
                #取proposal的最匹配idx,根据iou值判定为前景/中景/后景，在后续进行filter，此处code用不到
                """""
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                
                # matched = matched_idxs_in_image >= 0 
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = torch.tensor(0)

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = torch.tensor(-1)  # -1 is ignored by sampler
                """""

            bbx_score_all[img_idx] = bbx_score

        return bbx_score_all

    def subsample(self, labels):

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def select_training_samples(self, proposals, targets):
        """
        proposals: (List[Tensor[N, 4]])
        targets (List[Dict])
        """
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        for t in targets:
            #首先筛除用来填补的无意义框和caption
            mask = torch.nonzero(torch.gt(t["caps_len"],2))
            t['boxes'] = torch.index_select(t['boxes'],dim=0,index=mask.squeeze(1)).to(device)
            #筛除掉重复的框
            unique_index = torch.tensor(np.unique(t['boxes'].cpu().numpy(),axis=0,return_index=True)[-1]).to(device)
            t['boxes'] = torch.index_select(t['boxes'],dim=0,index=unique_index).to(device)
            t['caps'] = torch.index_select(t['caps'],dim=0,index=unique_index).to(device)
            t['caps_len'] = torch.index_select(t['caps_len'],dim=0,index=unique_index).to(device)


        gt_boxes = [t["boxes"].to(device) for t in targets]
        gt_labels = [torch.ones((t["boxes"].shape[0],), dtype=torch.int64, device=device) for t in
                     targets]  # generate labels LongTensor(1)

        # append ground-truth bboxes to proposals
        # List[2*N,4],一个list是一张图片
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]


        # assign_result 为 每一个region获取最为对应的proposal, 为list
        assign_result = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        num_images = len(proposals)
        matched_gt_boxes_list = []
        proposals_item_list = [] 
        corr_region_cap_list = []
        for index in range(num_images):
            
            sample_items = assign_result[index]
            if len(sample_items) == 0:
                continue

            matched_gt_boxes = []
            proposals_item = []
            region_index = []
            roi_score = []
            corr_region_cap = []

            for item in sample_items:
                matched_gt_boxes.append(item['gt_box'])
                proposals_item.append(item['region'])
                region_index.append(item['corresponding_region_index'])
                corr_region_cap.append(targets[index]['caps'][item['corresponding_region_index']])
                roi_score.append(item['score'])

            all_matched_gt_boxes = torch.stack(matched_gt_boxes,dim=0)
            all_proposals_item  = torch.stack(proposals_item,dim=0)
            all_corr_region_cap = torch.stack(corr_region_cap,dim=0)

            matched_gt_boxes_list.append(all_matched_gt_boxes)
            proposals_item_list.append(all_proposals_item)
            corr_region_cap_list.append(all_corr_region_cap)

        if len(matched_gt_boxes_list)!=0 and len(proposals_item_list)!=0:
            regression_targets = self.box_coder.encode(matched_gt_boxes_list, proposals_item_list)
        else:
            regression_targets = None
        return proposals_item_list, regression_targets, matched_gt_boxes_list, corr_region_cap_list

    def postprocess_outputs(self, box_regression, proposals, image_shapes, logits):
        device = logits.device
        num_classes = logits.shape[-1]
        pred_scores = F.softmax(logits, -1)

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        # all_scores = []
        # all_labels = []
        # all_captions = []
        # all_box_features = []
        # remove_inds_list = []
        # keep_list = []
        for boxes, scores, image_shape in zip(pred_boxes_list,pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)

            all_boxes.append(boxes)


        return all_boxes

    def postprocess_eval_outputs(self, box_regression, proposals, image_shapes, logits):
        device = logits.device
        num_classes = logits.shape[-1]
        pred_scores = F.softmax(logits, -1)

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_captions = []
        all_box_features = []
        remove_inds_list = []
        keep_list = []
        for boxes, scores, image_shape in zip(pred_boxes_list,pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            labels = labels.reshape(-1)

            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)


        return all_boxes

    def postprocess_detections_eval(self, logits, box_regression, caption_predicts, proposals, image_shapes,
                               box_features, return_features):
        device = logits.device
        num_classes = logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_caption_list = caption_predicts.split(boxes_per_image, 0)
        if return_features:
            pred_box_features_list = box_features.split(boxes_per_image, 0)
        else:
            pred_box_features_list = None

        all_boxes = []
        all_scores = []
        all_labels = []
        all_captions = []
        all_box_features = []
        remove_inds_list = []
        keep_list = []
        for boxes, scores, captions, image_shape in zip(pred_boxes_list, pred_scores_list, pred_caption_list,
                                                        image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            remove_inds_list.append(inds)
            boxes, scores, captions, labels = boxes[inds], scores[inds], captions[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, captions, labels = boxes[keep], scores[keep], captions[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            keep_list.append(keep)
            boxes, scores, captions, labels = boxes[keep], scores[keep], captions[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_captions.append(captions)
            all_labels.append(labels)

        if return_features:
            for inds, keep, box_features in zip(remove_inds_list, keep_list, pred_box_features_list):
                all_box_features.append(box_features[inds[keep]//(num_classes-1)])

        return all_boxes, all_scores, all_captions, all_box_features


    def postprocess_detections(self, logits, box_regression, caption_predicts, proposals, image_shapes,
                               box_features, return_features):
        device = logits.device
        num_classes = logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_caption_list = caption_predicts.split(boxes_per_image, 0)
        if return_features:
            pred_box_features_list = box_features.split(boxes_per_image, 0)
        else:
            pred_box_features_list = None

        all_boxes = []
        all_scores = []
        all_labels = []
        all_captions = []
        all_box_features = []
        remove_inds_list = []
        keep_list = []
        for boxes, scores, captions, image_shape in zip(pred_boxes_list, pred_scores_list, pred_caption_list,
                                                        image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            remove_inds_list.append(inds)
            boxes, scores, captions, labels = boxes[inds], scores[inds], captions[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, captions, labels = boxes[keep], scores[keep], captions[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            keep_list.append(keep)
            boxes, scores, captions, labels = boxes[keep], scores[keep], captions[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_captions.append(captions)
            all_labels.append(labels)

        if return_features:
            for inds, keep, box_features in zip(remove_inds_list, keep_list, pred_box_features_list):
                all_box_features.append(box_features[inds[keep]//(num_classes-1)])

        return all_boxes, all_scores, all_captions, all_box_features

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["caps"].dtype == torch.int64, 'target caps must of int64 (torch.long) type'
                assert t["caps_len"].dtype == torch.int64, 'target caps_len must of int64 (torch.long) type'

        if self.training:
            proposals, regression_targets, matched_gt_boxes_list, corr_region_cap_list = \
                self.select_training_samples(proposals, targets)
            if regression_targets == None:
                return None, {}
        else:
            regression_targets = None

        #feature '0':2*256*168*104; '1':2*256*84*52; '2':2*256*42*26; '3':2*256*21*13 'pool':2*256*11*7
        box_features = self.box_roi_pool(features, proposals, image_shapes) # MultiScaleRoIAlign 通过pool池化使得不同size的proposal转换为相同维数的特征向量
        box_features = self.box_head(box_features) #池化pool的过程 23(12+11)*256*7*7 ==> 1*577*768 (1*257*49?)
        
        
        new_feature = box_features.view(box_features.size(0),1,256,-1)
        new_feature = self.feature_transform_2(self.feature_transform_1(new_feature))
        if self.training:
            ce_loss, _ = self.textModel.textGenerate(new_feature, corr_region_cap_list)

        new_feature = self.feature_transform_3(new_feature.squeeze(1))
        
        logits, box_regression = self.box_predictor(new_feature.view(new_feature.size(0),-1)) #最终输入：23*4096
        
        #######################################################
        # using features from box_head to get box_regression
        # if self.training:
        #     new_feature = box_features.view(box_features.size(0),1,256,-1)
        #     new_feature = self.feature_transform_2(self.feature_transform_1(new_feature))
        #     ce_loss, _ = self.textModel.textGenerate(new_feature, corr_region_cap_list)
        #     #newfeature caption_num*1*256*768   
        # logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}

        pred_boxes_list = self.postprocess_outputs(box_regression, proposals, image_shapes, logits)

        if self.training:
            loss_box_reg_list = detect_loss_old(box_regression,regression_targets)
            losses = {
                "loss_box_reg":loss_box_reg_list,
                "loss_text_generate":ce_loss,
            }
            result = {
                "predict_region":pred_boxes_list,
                "matched_gt_boxes" :matched_gt_boxes_list,
                "corr_region_cap":corr_region_cap_list,
            }

        else:
            #TODO: 结合logits对proposal框进行筛选？NMS？
            losses = {}
            result = {
                "predict_region":pred_boxes_list,
            }

        return result, losses

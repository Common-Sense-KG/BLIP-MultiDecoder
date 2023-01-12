import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

from torchvision.ops import boxes as box_ops
from torchvision.models.detection import _utils as det_utils
from models.blip import blip_decoder



def detect_loss(box_regression, regression_targets):
# def detect_loss(proposals, matched_gt_boxes_list, box_regression):
    """
    Computes the loss for detection part.
    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    # labels = torch.cat(labels, dim=0)
    # regression_targets = torch.cat(regression_targets, dim=0)

    # classification_loss = F.cross_entropy(class_logits, labels)

    # # get indices that correspond to the regression targets for
    # # the corresponding ground truth labels, to be used with
    # # advanced indexing
    # sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    # labels_pos = labels[sampled_pos_inds_subset]
    # N, num_classes = class_logits.shape
    # loss_convert = nn.Linear()
    # reg_list = list(torch.chunk(box_regression, box_regression.shape[0], dim = 0))
    # box_loss_list = []
    # idx = 0 
    # for target_item in regression_targets:
    #     box_reg = torch.stack(reg_list[idx:idx+target_item.shape[0]],dim=0).squeeze(1)
    # # regression_targets = torch.cat(regression_targets, dim=0)
    #     box_loss = F.smooth_l1_loss(
    #         box_reg,
    #         target_item,
    #         reduction="sum",)#不reduction
    #     box_loss = box_loss / target_item.shape[0]
    #     box_loss_list.append(box_loss)
    #     idx += target_item.shape[0]
        
    # box_regression  #61*4 251+272+220+271
    proposal_num = box_regression.shape[0]
    reg_tar = torch.cat(regression_targets,0)
    box_loss = 0
    box_loss += F.smooth_l1_loss(box_regression, reg_tar, reduction='sum')
    box_loss = box_loss / proposal_num
    # proposal_num = 0
    # for idx in range(len(proposals)):
    #     box_loss += F.smooth_l1_loss(
    #         proposals[idx],
    #         matched_gt_boxes_list[idx],
    #         reduction="sum",
    #     )
    #     proposal_num += proposals[idx].shape[0]
    
    # box_loss = box_loss / proposal_num

    return box_loss


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
        self.box_describer = box_describer

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

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
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                actual_size = proposals_in_image.shape[0] - gt_boxes_in_image.shape[0]
                for i in range(match_quality_matrix.shape[0]):
                    # if match_quality_matrix[i][torch.argmax(match_quality_matrix[i][:actual_size])] < 0.7: #将gtbox对应所有anchors中iou最大的保留 且避免重复添加
                    bbx_score.append({'corresponding_region_index':i,'gt_box':gt_boxes_in_image[i],'score':match_quality_matrix[i][torch.argmax(match_quality_matrix[i][:actual_size])],'region':proposals_in_image[torch.argmax(match_quality_matrix[i][:actual_size])]})
                
                
                # temp = match_quality_matrix.t()[:actual_size]#取每个proposal框的max
                # for i in range(temp.shape[0]):
                #     if temp[i][torch.argmax(temp[i])] >= 0.7: # rpn网络中只有 > 0.7会被设为正样本
                #         bbx_score.append({'corresponding_region_index':torch.argmax(temp[i]).item(),'gt_box':gt_boxes_in_image[torch.argmax(temp[i])],'score':temp[i][torch.argmax(temp[i])],'region':proposals_in_image[i]})


                #模仿rpn网络加入 将gtbox对应所有anchors中iou最大的保留 
                #         
                # iou (Tensor[N, M]): the NxM matrix containing the IoU values for every element in boxes1 and boxes2

                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)#取matrix的最匹配idx

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                
                # matched = matched_idxs_in_image >= 0 

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = torch.tensor(0)

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = torch.tensor(-1)  # -1 is ignored by sampler

            bbx_score_all[img_idx] = bbx_score

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels, bbx_score_all

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
            mask = torch.nonzero(torch.gt(t["caps_len"],2))
            t['boxes'] = torch.index_select(t['boxes'],dim=0,index=mask.squeeze(1)).to(device)#首先筛除用来填补的无意义框和caption
            unique_index = torch.tensor(np.unique(t['boxes'].cpu().numpy(),axis=0,return_index=True)[-1]).to(device)
            t['boxes'] = torch.index_select(t['boxes'],dim=0,index=unique_index).to(device)#筛除掉重复的框
            t['caps'] = torch.index_select(t['caps'],dim=0,index=unique_index).to(device)
            t['caps_len'] = torch.index_select(t['caps_len'],dim=0,index=unique_index).to(device)


        gt_boxes = [t["boxes"].to(device) for t in targets]
        gt_labels = [torch.ones((t["boxes"].shape[0],), dtype=torch.int64, device=device) for t in
                     targets]  # generate labels LongTensor(1)

        # append ground-truth bboxes to propos
        # List[2*N,4],一个list是一张图片
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        # get matching gt indices for each proposal
        matched_idxs, labels, bbx_score = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        num_images = len(proposals)
        matched_gt_boxes_list = []
        proposals_item_list = [] 
        corr_region_cap_list = []
        region_index_list = [[]] * num_images
        roi_score_list = [[]] * num_images
        for img_id in range(num_images):
            sample_items = bbx_score[img_id]
            if len(sample_items) == 0:
                continue
            # for sample_item in sample_items:
            #     sample_item['corresponding_region_index']
            matched_gt_boxes = []
            proposals_item = []
            region_index = []
            roi_score = []
            corr_region_cap = []
            for item in sample_items:
                matched_gt_boxes.append(item['gt_box'])
                proposals_item.append(item['region'])
                region_index.append(item['corresponding_region_index'])
                corr_region_cap.append(targets[img_id]['caps'][item['corresponding_region_index']])
                roi_score.append(item['score'])
            all_matched_gt_boxes = torch.stack(matched_gt_boxes,dim=0)
            all_proposals_item  = torch.stack(proposals_item,dim=0)
            all_corr_region_cap = torch.stack(corr_region_cap,dim=0)
            matched_gt_boxes_list.append(all_matched_gt_boxes)
            proposals_item_list.append(all_proposals_item)
            corr_region_cap_list.append(all_corr_region_cap)
        if len(matched_gt_boxes_list)!=0 and len(proposals_item_list)!=0:
            regression_targets = self.box_coder.encode(matched_gt_boxes_list, proposals_item_list)#list为空，易出现错误
        else:
            regression_targets = None
        return proposals_item_list, regression_targets, matched_gt_boxes_list, corr_region_cap_list

    def postprocess_train_outputs(self, box_regression, proposals, image_shapes, logits):
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
            labels = None
            matched_idxs = None
            caption_gt = None
            caption_length = None
            regression_targets = None

        # text_info = 
        # textModel = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
        #                    vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
        #                    prompt=config['prompt'])

        # textModel = textModel.to(device)

        box_features = self.box_roi_pool(features, proposals, image_shapes) # MultiScaleRoIAlign 通过pool池化使得不同size的proposal转换为相同维数的特征向量
        box_features = self.box_head(box_features) #池化pool的过程
        logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}

        pred_boxes_list = self.postprocess_train_outputs(box_regression, proposals, image_shapes, logits)

        
        if self.training:
            loss_box_reg_list = detect_loss(box_regression,regression_targets)
            
            losses = {
                "loss_box_reg":loss_box_reg_list,
            }
            result = {
                "predict_region":pred_boxes_list,
                "matched_gt_boxes" :matched_gt_boxes_list,
                "corr_region_cap":corr_region_cap_list,
            }

        else:
            losses = {}
            result = {
                "predict_region":pred_boxes_list,
            }

            # return loss_box_reg, proposals, matched_gt_boxes_list, corr_region_cap_list

            # boxes, scores, caption_predicts, feats = self.postprocess_detections_eval(logits, box_regression,
            #                                                                      proposals, image_shapes, box_features,
            #                                                                      self.return_features)
            # num_images = len(boxes)
            # for i in range(num_images):
            #     result.append(
            #         {
            #             "boxes": boxes[i],
            #             "caps": caption_predicts[i],
            #             "scores": scores[i],
            #         }
            #     )
            #     if self.return_features:
            #         result[-1]['feats'] = feats[i]

        return result, losses

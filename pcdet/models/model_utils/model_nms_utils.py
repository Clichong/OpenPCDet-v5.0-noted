import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    """
    利用nms挑选出anchor的索引以及对应的置信度
    """
    # 首先根据置信度阈值过滤掉部分box
    src_box_scores = box_scores
    if score_thresh is not None:    # 这里的阈值设定为0.1
        scores_mask = (box_scores >= score_thresh)  # 阈值过滤
        box_scores = box_scores[scores_mask]    # (145)
        box_preds = box_preds[scores_mask]      # (145, 7)

    selected = []
    if box_scores.shape[0] > 0:    # 筛选后还存在object
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))    # 返回排序后的置信度大小以及其索引
        boxes_for_nms = box_preds[indices]  # 根据置信度大小排序
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]  # 根据返回索引找出box索引值

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)  # 原始大于置信度的索引
        # selected表示的box_scores的选择索引，经过这次索引，selected表示的是src_box_scores被选择的索引
        selected = original_idxs[selected]      # 在索引中挑选索引

    # 返回的是被选择的索引和索引分数
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes

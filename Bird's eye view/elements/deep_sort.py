import torch
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from elements.assets import draw_boxes

class DEEPSORT():
    def __init__(self, deepsort_config):
        cfg = get_config()
        cfg.merge_from_file(deepsort_config)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

        print('DeepSort model loaded!')
    
    
    def detection_to_deepsort(self, objects, im0):
        xywh_bboxs = []
        confs = []

        # Adapt detections to deep sort input format
        for obj in objects:
            if obj['label'] == 'player':
                xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
                conf = obj['score']
                # to deep sort format
                x_c, y_c, bbox_w, bbox_h = self.xyxy_to_xywh(*xyxy)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                xywh_bboxs.append(xywh_obj)
                confs.append([conf])

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        # pass detections to deepsort
        outputs = self.deepsort.update(xywhs, confss, im0)

        # draw boxes for visualization
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            draw_boxes(im0, bbox_xyxy, identities)
    

    def xyxy_to_xywh(self, *xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0], xyxy[2]])
        bbox_top = min([xyxy[1], xyxy[3]])
        bbox_w = abs(xyxy[0] - xyxy[2])
        bbox_h = abs(xyxy[1] - xyxy[3])
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h
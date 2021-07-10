import torch
import cv2
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = {0: 'player', 1: 'ball'}

class YOLO():
    def __init__(self,model_path, conf_thres, iou_thres):
        self.yolo_model = attempt_load(weights=model_path, map_location=device)
        print("Yolo model loaded!")
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self,frame):
        """
            Input :
                    BGR image
                 
            Output:
            yolo return list of dict in format:
                {   label   :  str
                    bbox    :  [(xmin,ymin),(xmax,ymax)]
                    score   :  float
                    cls     :  int
                }
        """
        img = cv2.resize(frame, (640,384))

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float()/255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.yolo_model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=None)
        items = []
        
        if pred[0] is not None and len(pred):
            for p in pred[0]:
                if int(p[5]) in list(classes.keys()): 
                    score = np.round(p[4].cpu().detach().numpy(),2)
                    label = classes[int(p[5])]
                    xmin = int(p[0] * frame.shape[1] /640)
                    ymin = int(p[1] * frame.shape[0] /384)
                    xmax = int(p[2] * frame.shape[1] /640)
                    ymax = int(p[3] * frame.shape[0] /384)

                    item = {'label': label,
                            'bbox' : [(xmin,ymin),(xmax,ymax)],
                            'score': score,
                            'cls' : int(p[5])}

                    items.append(item)

        return(items)

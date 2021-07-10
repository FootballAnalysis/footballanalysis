from elements.yolo import YOLO
from elements.deep_sort import DEEPSORT
from elements.perspective_transform import Perspective_Transform
from elements.assets import transform_matrix, detect_color
from arguments import Arguments
from yolov5.utils.plots import plot_one_box

import torch
import os
import cv2
import numpy as np
import sys



def main(opt):
    # Load models
    detector = YOLO(opt.yolov5_model, opt.conf_thresh, opt.iou_thresh)
    deep_sort = DEEPSORT(opt.deepsort_config)
    perspective_transform = Perspective_Transform()

    # Video capture
    cap = cv2.VideoCapture(opt.source)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Save output
    if opt.save:
        output_name = opt.source.split('/')[-1]
        output_name = output_name.split('.')[0] + '_output.' + output_name.split('.')[-1]

        output_path = os.path.join(os.getcwd(), 'inference/output')
        os.makedirs(output_path, exist_ok=True)
        output_name = os.path.join(os.getcwd(), 'inference/output', output_name)

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        out = cv2.VideoWriter(output_name,  
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                opt.outputfps, (int(w), int(h)))


    frame_num = 0

    # Black Image (Soccer Field)
    bg_ratio = int(np.ceil(w/(3*115)))
    gt_img = cv2.imread('./inference/black.jpg')
    gt_img = cv2.resize(gt_img,(115*bg_ratio, 74*bg_ratio))
    gt_h, gt_w, _ = gt_img.shape


    while(cap.isOpened()):
        
        ret, frame = cap.read()
        bg_img = gt_img.copy()

        if ret:
            main_frame = frame.copy()
            yoloOutput = detector.detect(frame)

            # Output: Homography Matrix and Warped image 
            if frame_num % 5 ==0: # Calculate the homography matrix every 5 frames
                M, warped_image = perspective_transform.homography_matrix(main_frame)

            if yoloOutput:

                # Tracking
                deep_sort.detection_to_deepsort(yoloOutput, frame)

                # The homography matrix is applied to the center of the lower side of the bbox.
                for i, obj in enumerate(yoloOutput):
                    xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
                    x_center = (xyxy[0] + xyxy[2])/2 
                    y_center = xyxy[3]
                    
                    if obj['label'] == 'player':
                        coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                        try:
                            color = detect_color(main_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
                            cv2.circle(bg_img, coords, bg_ratio + 1, color, -1)
                        except:
                          pass
                    elif obj['label'] == 'ball':
                        coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                        cv2.circle(bg_img, coords, bg_ratio + 1, (102, 0, 102), -1)
                        plot_one_box(xyxy, frame, (102, 0, 102), label="ball")
            else:
                deep_sort.deepsort.increment_ages()

            frame[frame.shape[0]-bg_img.shape[0]:, frame.shape[1]-bg_img.shape[1]:] = bg_img  
            
            if opt.view:
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & ord('q') == 0xFF:
                    break

            # Saving the output
            if opt.save:
                out.write(frame)

            frame_num += 1
        else:
            break

        sys.stdout.write(
            "\r[Input Video : %s] [%d/%d Frames Processed]"
            % (
                opt.source,
                frame_num,
                frame_count,
            )
        )

    if opt.save:
        print(f'\n\nOutput video has been saved in {output_path}!')
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    opt = Arguments().parse()
    with torch.no_grad():
        main(opt)


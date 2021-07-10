import cv2
from sklearn.cluster import KMeans
import numpy as np


pallete = {'b': (0, 0, 128),
        'g': (0, 128, 0),
        'r': (255, 0, 0),
        'c': (0, 192, 192),
        'm': (192, 0, 192),
        'y': (192, 192, 0),
        'k': (0, 0, 0),
        'w': (255, 255, 255)}

color_for_labels = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in color_for_labels]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def transform_matrix(matrix, p, vid_shape, gt_shape):
    p = (p[0]*1280/vid_shape[1], p[1]*720/vid_shape[0])
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))

    p_after = (int(px*gt_shape[1]/115) , int(py*gt_shape[0]/74))

    return p_after


# Color Detection with K-means
def detect_color(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[1]*img.shape[0],3))

    kmeans = KMeans(n_clusters=2)
    s = kmeans.fit(img)

    labels = kmeans.labels_
    centroid = kmeans.cluster_centers_
    labels = list(labels)
    percent=[]
    
    for i in range(len(centroid)):
        j=labels.count(i)
        j=j/(len(labels))
        percent.append(j)

    detected_color = centroid[np.argmin(percent)]
    
    list_of_colors = list(pallete.values())
    assigned_color = closest_color(list_of_colors, detected_color)[0]
    assigned_color = (int(assigned_color[2]), int(assigned_color[1]), int(assigned_color[0]))

    if assigned_color == (0, 0, 0):
        assigned_color = (128, 128, 128)

    return assigned_color


# Find the closest color to the detected one based on the predefined palette
def closest_color(list_of_colors, color):
    colors = np.array(list_of_colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_shortest = np.where(distances==np.amin(distances))
    shortest_distance = colors[index_of_shortest]

    return shortest_distance 
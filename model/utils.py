import cv2
from numpy import ndarray
import numpy as np

class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
               5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic_light', 10: 'fire hydrant',
               11: 'stop_sign', 12: 'parking_meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog',
               17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
               23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
               28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports_ball',
               33: 'kite', 34: 'baseball_bat', 35: 'baseball_glove', 36: 'skateboard',
               37: 'surfboard', 38: 'tennis_racket', 39: 'bottle', 40: 'wine_glass', 41: 'cup',
               42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
               47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
               52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
               58: 'potted_plant', 59: 'bed', 60: 'dining_table', 61: 'toilet', 62: 'tv',
               63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell_phone',
               68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
               73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
               78: 'hair drier', 79: 'toothbrush'}

colors = [(67, 161, 255), (19, 222, 24), (186, 55, 2), (167, 146, 11), (190, 76, 98),
          (130, 172, 179), (115, 209, 128), (204, 79, 135), (136, 126, 185), (209, 213, 45),
          (44, 52, 10), (101, 158, 121), (179, 124, 12), (25, 33, 189), (45, 115, 11),
          (73, 197, 184), (62, 225, 221), (32, 46, 52), (20, 165, 16), (54, 15, 57),
          (12, 150, 9), (10, 46, 99), (94, 89, 46), (48, 37, 106), (42, 10, 96),
          (7, 164, 128), (98, 213, 120), (40, 5, 219), (54, 25, 150), (251, 74, 172),
          (0, 236, 196), (21, 104, 190), (226, 74, 232), (120, 67, 25), (191, 106, 197),
          (8, 15, 134), (21, 2, 1), (142, 63, 109), (133, 148, 146), (187, 77, 253),
          (155, 22, 122), (218, 130, 77), (164, 102, 79), (43, 152, 125), (185, 124, 151),
          (95, 159, 238), (128, 89, 85), (228, 6, 60), (6, 41, 210), (11, 1, 133),
          (30, 96, 58), (230, 136, 109), (126, 45, 174), (164, 63, 165), (32, 111, 29),
          (232, 40, 70), (55, 31, 198), (148, 211, 129), (10, 186, 211), (181, 201, 94),
          (55, 35, 92), (129, 140, 233), (70, 250, 116), (61, 209, 152), (216, 21, 138),
          (100, 0, 176), (3, 42, 70), (151, 13, 44), (216, 102, 88), (125, 216, 93),
          (171, 236, 47), (253, 127, 103), (205, 137, 244), (193, 137, 224), (36, 152, 214),
          (17, 50, 238), (154, 165, 67), (114, 129, 60), (119, 24, 48), (73, 8, 110)]


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None) -> ndarray:
    """
    Draws the object masks on the image.

    :param image: The input image.
    :param boxes: The bounding boxes.
    :param scores: The scores of the detections.
    :param class_ids: The class IDs of the detections.
    :param mask_alpha: The opacity of the mask overlay. Default is 0.3.
    :param mask_maps: The mask predictions. Default is None.
    :return: The image with object masks overlaid on it.
    """
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    mask_img = draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)

    # Draw bounding boxes and labels of detections
    labels = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id][::-1]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

        label = class_names[class_id]
        labels.append(label)
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return mask_img


def draw_masks(image, boxes, class_ids, mask_alpha=0.3, mask_maps=None) -> ndarray:
    """
    Draws the object masks on the image.

    :param image: The input image.
    :param boxes: The bounding boxes.
    :param class_ids: The class IDs of the detections.
    :param mask_alpha: The opacity of the mask overlay. Default is 0.3.
    :param mask_maps: The mask predictions. Default is None.
    :return: The image with object masks overlaid on it.
    """
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        color = colors[class_id][::-1]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill mask image
        if mask_maps is None:
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        else:
            crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def sigmoid(x) -> ndarray:
    """
    Sigmoid function.

    :param x: The input array.
    :return: The output array after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    """
    Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2).

    :param x: The input bounding boxes in (x, y, w, h) format.
    :return: The converted bounding boxes in (x1, y1, x2, y2) format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression on the bounding boxes.

    :param boxes: The bounding boxes.
    :param scores: The scores of the detections.
    :param iou_threshold: The IoU threshold for suppression.
    :return: The list of indices of the kept bounding boxes after non-maximum suppression.
    """
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    """
    Computes the IoU (Intersection over Union) between a box and a list of boxes.

    :param box: The first box in (x1, y1, x2, y2) format.
    :param boxes: The list of boxes in (x1, y1, x2, y2) format.
    :return: The IoU values between the box and the list of boxes.
    """
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

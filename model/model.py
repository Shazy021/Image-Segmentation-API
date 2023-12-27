import math
import cv2
import numpy as np
from numpy import ndarray
import onnxruntime
from model.utils import nms, sigmoid, xywh2xyxy, draw_detections, class_names


class YOLOSeg:
    """
    A class that performs object segmentation using the YOLO model.
    """

    def __init__(self, model_path: str,
                 conf_thres: float = 0.4,
                 iou_thres: float = 0.5, ):
        """
        Initializes the YOLOSeg object.

        :param model_path: The path to the YOLO model.
        :param conf_thres: The confidence threshold for filtering out object detections.
            Default is 0.4.
        :param iou_thres: The intersection over union threshold for non-maximum suppression.
            Default is 0.5.
        """
        self.model_path = model_path
        self.conf_thresh = conf_thres
        self.iou_thres = iou_thres
        self.img = None
        self.session = None
        self.input_names = None
        self.input_shape = None
        self.input_height = None
        self.input_width = None
        self.output_names = None
        self.boxes = None
        self.scores = None
        self.class_ids = None
        self.mask_maps = None
        self.img_height = None
        self.img_width = None

    def model_run(self, img: ndarray) -> dict:
        """
        Runs the YOLO model and segment objects in the input image.

        :param img: The input image for segmentation.
        :return: A dictionary with the following keys:
                    - 'img': ndarray. The input image with object masks overlaid on it.
                    - 'resp': dict. A dictionary containing the count of each detected
                              class in the image.
        """
        self.img = img
        self.initialize_model(self.model_path)

        self.segment_objects(self.img)
        combined_img = self.draw_masks(self.img)

        res_dict = {}

        for key in self.class_ids:
            if class_names[key] in res_dict:
                res_dict[class_names[key]] += 1
            else:
                res_dict[class_names[key]] = 1

        return {
            'img': combined_img,
            'resp': res_dict
        }

    def initialize_model(self, model_path: str) -> None:
        """
         Initializes the YOLO model using the given model path.

        :param model_path: The path to the YOLO model.
        """
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=[
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
        )

        self.get_input_details()
        self.get_output()

    def get_input_details(self) -> None:
        """
        Gets the input details of the YOLO model.
        """
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output(self) -> None:
        """
        Gets the output details of the YOLO model.
        """
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def segment_objects(self, image: ndarray) -> tuple:
        """
        Segments objects in the input image using the YOLO model.

        :param image: The input image.
        :return: A tuple containing the bounding boxes,
                 scores, class IDs, and mask predictions of the objects
        """
        input_tensor = self.prepare_input(image)

        outputs = self.model_inference(input_tensor)

        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input(self, image: ndarray) -> ndarray:
        """
        Prepares the input image for inference.

        :param image: The input image.
        :return: The prepared input tensor for the model.
        """
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.resize(image, (self.input_width, self.input_height))

        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def model_inference(self, input_tensor: ndarray) -> ndarray:
        """
        Performs inference using the YOLO model.

        :param input_tensor: The input tensor for the model.
        :return: The output predictions from the model
        """
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    def process_box_output(self, box_output: ndarray) -> tuple:
        """
        Processes the output of the YOLO model for object detection.

        :param box_output: The output predictions for bounding box detection.
        :return: A tuple containing the filtered bounding boxes,
                 scores, class IDs, and mask predictions.
        """
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - 36

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:num_classes], axis=1)
        predictions = predictions[scores > self.conf_thresh, :]
        scores = scores[scores > self.conf_thresh]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_thres)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def extract_boxes(self, box_predictions: ndarray) -> ndarray:
        """
        Extracts the bounding boxes from the predictions.

        :param box_predictions:
        :return:
        """
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def process_mask_output(self, mask_predictions: ndarray, mask_output: ndarray):
        """
        Processes the output of the YOLO model for mask prediction.

        :param mask_predictions: The output predictions for mask.
        :param mask_output: The shape of the mask predictions.
        :return: The processed mask predictions.
        """
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                         (self.img_height, self.img_width),
                                         (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i, (scale_box, box) in enumerate(zip(scale_boxes, self.boxes)):
            scale_x1 = int(math.floor(scale_box[0]))
            scale_y1 = int(math.floor(scale_box[1]))
            scale_x2 = int(math.ceil(scale_box[2]))
            scale_y2 = int(math.ceil(scale_box[3]))

            x1 = int(math.floor(box[0]))
            y1 = int(math.floor(box[1]))
            x2 = int(math.ceil(box[2]))
            y2 = int(math.ceil(box[3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                                   (x2 - x1, y2 - y1),
                                   interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def draw_masks(self, image: ndarray, mask_alpha: float = 0.5) -> ndarray:
        """
        Draws the object masks on the image.

        :param image: The input image.
        :param mask_alpha: The alpha value for opacity of the mask overlay. Default is 0.5.
        :return: The image with object masks overlaid on it.
        """
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha, mask_maps=self.mask_maps)

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape) -> ndarray:
        """
        Rescales the bounding boxes to the original image dimensions

        :param boxes: The bounding boxes.
        :param input_shape: The input shape of the YOLO model.
        :param image_shape: The original image shape.
        :return: The rescaled bounding boxes.
        """
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes

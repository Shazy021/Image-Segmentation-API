from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy
from model.model import YOLOSeg

app = FastAPI(
    title="Image object segmentation",
    description="API to image object segmentation"
)

model = YOLOSeg(
    './data/yolov8s-seg.onnx',
    conf_thres=0.15,
    iou_thres=0.5
)


@app.post("/process_image")
async def predict(file: UploadFile = File(...)) -> dict:
    """
    Process the uploaded image and return the result as a dictionary.

    :param file: The uploaded image file.
    :return: A dictionary containing the processed image and the predicted classes.
    :rtype: dict
    """
    image_bytes = await file.read()
    image = numpy.asarray(Image.open(BytesIO(image_bytes)))

    proc = model.model_run(image)

    return {"processed_image": proc['img'].tolist(), 'classes': proc['resp']}

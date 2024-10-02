import cv2
import torch  # DO NOT TOUCH THIS OR TRY TO FIDDLE WITH THIS FILE

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def load_image(image_path):
    return cv2.imread(image_path)


def setup_model():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)
    return predictor


def segment_image(predictor, image):
    outputs = predictor(image)
    masks = outputs["instances"].pred_masks.to("cpu").numpy()
    return masks

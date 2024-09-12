import os

import detectron2.data.transforms as T
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

SIZE = 128
AUG = T.FixedSizeCrop((SIZE, SIZE), pad_value=0)
inv_label_map = {
    0: "lymphocyte",
    1: "monocyte",
}


def transform(image):
    image = AUG.get_transform(image).apply_image(image)
    return image


class BatchPredictor(DefaultPredictor):
    """Run d2 on a list of images."""

    def __call__(self, images):

        input_images = []
        for image in images:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]
            new_image = transform(image)
            new_image = torch.as_tensor(new_image.astype("float32").transpose(2, 0, 1))

            input_images.append({"image": new_image, "height": height, "width": width})

        with torch.no_grad():
            preds = self.model(input_images)
        return preds


class Detectron2DetectionPredictor:
    def __init__(self, output_dir, threshold, nms_threshold, weight_root):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
            )
        )


        cfg.DATASETS.TRAIN = ("detection_dataset2",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 1

        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 24, 32]]

        cfg.SOLVER.IMS_PER_BATCH = 10
        cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
        cfg.SOLVER.MAX_ITER = 2000  # 2000 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
        cfg.SOLVER.STEPS = (10, 100, 250)
        cfg.SOLVER.WARMUP_ITERS = 0
        cfg.SOLVER.GAMMA = 0.5
        ###



        cfg.OUTPUT_DIR = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        cfg.MODEL.WEIGHTS = weight_root
        

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
        cfg.MODEL.RPN.NMS_THRESH = nms_threshold

        self._predictor = BatchPredictor(cfg)

    def predict_on_batch(self, x_batch):
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = self._predictor(x_batch)
        predictions = []
        for output in outputs:
            predictions.append([])
            pred_boxes = output["instances"].get("pred_boxes")

            scores = output["instances"].get("scores")
            classes = output["instances"].get("pred_classes")
            centers = pred_boxes.get_centers()
            for idx, center in enumerate(centers):
                x, y = center.cpu().detach().numpy()
                confidence = scores[idx].cpu().detach().numpy()
                label = inv_label_map[int(classes[idx].cpu().detach())]
                prediction_record = {
                    "x": int(x),
                    "y": int(y),
                    "label": str(label),
                    "confidence": float(confidence),
                }
                predictions[-1].append(prediction_record)
        return predictions

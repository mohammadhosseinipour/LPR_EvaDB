# coding=utf-8

import torch
import pandas as pd
import numpy as np

from evadb.functions.abstract.pytorch_abstract_function import PytorchAbstractClassifierFunction
from evadb.functions.yolo_object_detector import Yolo
from evadb.functions.decorators.decorators import setup, forward
from evadb.functions.decorators.io_descriptors.data_types import PyTorchTensor, PandasDataframe
from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.gpu_compatible import GPUCompatible
from evadb.utils.generic_utils import try_to_import_ultralytics

class YoloObjectDetection(AbstractFunction, GPUCompatible):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score
    """

    @property
    def name(self) -> str:
        return "yolo"

    @setup(cacheable=True, function_type="object_detection", batchable=True)
    def setup(self, model_loc: str = "/Users/mohammadhp/Desktop/Projects/EvaDB/LPR_EvaDB/best_LP.pt", threshold=0.70):
        try_to_import_ultralytics()
        from ultralytics import YOLO

        self.threshold = threshold
        self.model = YOLO(model_loc)
        self.device = "cpu"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["data"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None, 3)],
            )
        ],
        output_signatures=[
        PandasDataframe(
            columns=["labels", "bboxes", "scores", "cropped_images"],
            column_types=[
                NdArrayType.STR,
                NdArrayType.FLOAT32,
                NdArrayType.FLOAT32,
                NdArrayType.FLOAT32  # Assuming you're storing the cropped image as a FLOAT32 ndarray
            ],
            column_shapes=[(None,), (None,), (None,), (None, None, 3)],
        )
    ],
    )
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed
        Returns:
            tuple containing predicted_classes (List[List[str]]),
            predicted_boxes (List[List[BoundingBox]]),
            predicted_scores (List[List[float]])
        """
        outcome = []
        # Fix me: this should be taken care by decorators
        frames = np.ravel(frames.to_numpy())
        list_of_numpy_images = [its for its in frames]
        predictions = self.model.predict(
            list_of_numpy_images, device=self.device, conf=self.threshold, verbose=False
        )
        for idx, pred in enumerate(predictions):
            single_result = pred.boxes
            original_frame = list_of_numpy_images[idx]
            # Crop the frame
            try:
                x1, y1, x2, y2 = map(int, single_result.xyxy[0])
                cropped_frame = original_frame[y1:y2, x1:x2]
            except:
                cropped_frame=[(None, None, 3)]
            pred_class = [self.model.names[i] for i in single_result.cls.tolist()]
            pred_score = single_result.conf.tolist()
            pred_score = [round(conf, 2) for conf in single_result.conf.tolist()]
            pred_boxes = single_result.xyxy.tolist()
            sorted_list = list(map(lambda i: i < self.threshold, pred_score))
            t = sorted_list.index(True) if True in sorted_list else len(sorted_list)
            outcome.append(
            {
                "labels": pred_class[:t],
                "bboxes": pred_boxes[:t],
                "scores": pred_score[:t],
                "cropped_images": cropped_frame
            },
            )
        return pd.DataFrame(
        outcome,
        columns=[
            "labels",
            "bboxes",
            "scores",
            "cropped_images"
        ],
    )
    def to_device(self, device: str):
        self.device = device
        return self

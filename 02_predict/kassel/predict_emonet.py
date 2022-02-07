import time
import torch

from hcai_dataset_utils.pytorch_dataset_wrapper import PyTorchDatasetWrapper
from hcai_models.expansion.preprocessing.blaze_face_detector import BlazeFaceDetector
from hcai_models.expansion.preprocessing.bounding_box_image_crop import (
    BoundingBoxImageCrop,
)
from hcai_models.expansion.preprocessing.s3fd_detector import S3FDDetector

from hcai_models.models.image.hcai_emonet.emonet import EmoNet8
import tensorflow_datasets as tfds
import os
import hcai_datasets
import tensorflow as tf

from hcai_models.utils.metric_utils import ACC, SAGR, CCC, PCC, RMSE
from hcai_models.utils.torch_utils import MetricMapping

from matplotlib import pyplot as plt

start = time.time()
from matplotlib import pyplot as plt

# Load Data
ds, ds_info = tfds.load(
    "hcai_nova_dynamic",
    split="dynamic_split",
    with_info=True,
    as_supervised=False,
    data_dir=".",
    shuffle_files=False,
    read_config=tfds.ReadConfig(shuffle_seed=1337),
    builder_kwargs={
        # Database Config
        "db_config_path": "../configs/nova/nova_db.cfg",

        # Dataset Config
        "dataset": "kassel_therapie_korpus",
        "nova_data_dir": os.path.join(r"/Volumes/Corpora_T5/"),
        "sessions": ["OPD_101_No"],
        "roles": ["patient", "therapist"],
        "schemes": [],
        "annotator": "system",
        "data_streams": ["video_resized"],

        # Sample Config
        "frame_size": 0.04,
        "left_context": 0,
        "right_context": 0,
        "flatten_samples": False,

        # Additional Config
        "lazy_loading": False,
        "clear_cache": False
    }
)

# Preprocessing
def pp(batch):
    image = batch["video_resized"]

    #bb = BlazeFaceDetector()(image)
    bb = S3FDDetector()(image)
    image = BoundingBoxImageCrop(output_dim=(256, 256))(image, bb)
    image = tf.experimental.numpy.moveaxis(image, -1, -3)
    image = image / 255
    batch["video_resized"] = image
    return batch

ds = ds.map(pp)
data_it = ds.as_numpy_iterator()
data_list = list(data_it)
data_list.sort(key=lambda x: int(x["frame"].decode("utf-8").split("_")[0]))

# Build model
model = EmoNet8(weights="affectnet")
model.build_model()

# Predict
a = model.predict(data_list['video_resized'])

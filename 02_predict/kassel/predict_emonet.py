import pandas as pd
from hcai_models.expansion.preprocessing.bounding_box_image_crop import (
    BoundingBoxImageCrop,
)
from hcai_models.expansion.preprocessing.s3fd_detector import S3FDDetector
from hcai_models.expansion.preprocessing.blaze_face_detector import BlazeFaceDetector
from hcai_models.models.image.hcai_sfd.sfd import S3FD
from hcai_models.models.image.hcai_emonet.emonet import EmoNet8
import tensorflow_datasets as tfds
import os
import tensorflow as tf
import hcai_datasets
from utils.config_utils import read_configs, builder_kwargs_from_config_paths
import cv2
from matplotlib import pyplot as plt
import decord
from decord import VideoReader
from decord import cpu, gpu
from tqdm import tqdm
import numpy as np
from pathlib import Path
from utils.config_utils import read_configs
import pandas

# Configs
cfg_path = '../../.configs/dynamic_datasets/kassel_emonet.cfg'
cfg = read_configs([cfg_path])
cfg_ds = cfg['DATASET']

# Paths
dataset_root = Path(cfg_ds.get('nova_data_dir')) / cfg_ds.get('dataset')
outpath = Path('./emonet_predictions')

# Files
file_list = []
for vid in dataset_root.rglob('*.video_resized.mp4'):
    file_list.append(vid)

# Preprocessing modules
face_detector = BlazeFaceDetector()
face_detector._device = 'gpu'
bb_cropper = BoundingBoxImageCrop(output_dim=(256, 256))

# Emonet model
emonet = EmoNet8(weights="affectnet")
emonet.build_model()
emonet._model.eval()

# Predict
for file_path in file_list:
    csv_out = outpath / ( str(file_path.parent.stem) + '-' + str(file_path.stem) + '.csv')
    if csv_out.is_file():
        continue
    data = {
        'frame': [],
        'bb': [],
        'expression': [],
        'valence': [],
        'arousal': [],
        'heatmap': []
    }
    with open(file_path, 'rb') as f:
        print(f'Predicting {file_path}')
        vr = VideoReader(f, ctx=cpu(0))
        for i in tqdm(range(len(vr))):
            img = vr[i].asnumpy()

            # Detect faces
            img_pp, (xshift, yshift), orig_size = face_detector.blaze_face.preprocessing(img)
            bb = face_detector.blaze_face.predict(np.expand_dims(img_pp, 0))
            detected_faces = face_detector.blaze_face.postprocessing(bb, shifts=(xshift, yshift), orig_size=orig_size)

            # Predict Emotions
            if detected_faces.any():
                img_emo = bb_cropper._internal_call(img, np.squeeze(detected_faces[0,:4]))
                img_emo = img_emo.astype(dtype=np.float32)
                img_emo = img_emo/255
                img_emo = np.moveaxis(img_emo, -1, -3)
                out = emonet.predict(img_emo)
                data['frame'].append(i)
                data['bb'].append(detected_faces[0])
                data['expression'].append(np.squeeze(out['expression'].cpu().numpy()))
                data['valence'].append(np.squeeze(out['valence'].cpu().numpy()))
                data['arousal'].append(np.squeeze(out['arousal'].cpu().numpy()))
                data['heatmap'].append(np.squeeze(out['heatmap'].cpu().numpy()))

    outpath.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(data)
    df.to_csv(csv_out)


from hcai_models.expansion.preprocessing.bounding_box_image_crop import (
    BoundingBoxImageCrop,
)

from hcai_models.expansion.preprocessing.s3fd_detector import S3FDDetector
from hcai_models.expansion.preprocessing.blaze_face_detector import BlazeFaceDetector
from hcai_models.models.image.hcai_emonet.emonet import EmoNet8
from matplotlib import pyplot as plt
from decord import VideoReader
from decord import cpu
from tqdm import tqdm
import numpy as np
from pathlib import Path
from utils.config_utils import read_configs
import csv
from scipy.special import softmax

# Helper
def get_preds_fromhm(hm, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.
    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]
    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape
    hm_reshape = hm.reshape(B, C, H * W)
    idx = np.argmax(hm_reshape, axis=-1)
    scores = np.take_along_axis(
        hm_reshape, np.expand_dims(idx, axis=-1), axis=-1
    ).squeeze(-1)
    preds, preds_orig = _get_preds_fromhm(hm, idx, center, scale)

    return preds, preds_orig, scores


def transform_np(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.
    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.
    Arguments:
        point {numpy.array} -- the input 2D point
        center {numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution
    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = np.ascontiguousarray(np.linalg.pinv(t))

    new_point = np.dot(t, _pt)[0:2]

    return new_point.astype(np.int32)


def _get_preds_fromhm(hm, idx, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps and the
    coresponding locations of the maximums. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.
    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]
    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape
    idx += 1
    preds = idx.repeat(2).reshape(B, C, 2).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
    preds[:, :, 1] = np.floor((preds[:, :, 1] - 1) / H) + 1

    for i in range(B):
        for j in range(C):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array(
                    [
                        hm_[pY, pX + 1] - hm_[pY, pX - 1],
                        hm_[pY + 1, pX] - hm_[pY - 1, pX],
                    ]
                )
                preds[i, j] += np.sign(diff) * 0.25

    preds -= 0.5

    preds_orig = np.zeros_like(preds)
    if center is not None and scale is not None:
        for i in range(B):
            for j in range(C):
                preds_orig[i, j] = transform_np(preds[i, j], center, scale, H, True)

    return preds, preds_orig


# Debug
def _plot_current_row(current_row):

    # Boudning box
    x1, y1 = np.array(current_row["face_upper_left"].split(";"), dtype=int)
    x2, y2 = np.array(current_row["face_lower_right"].split(";"), dtype=int)

    plt.plot((x1, x2), (y1, y1), marker="o", linewidth=1)
    plt.plot((x1, x1), (y1, y2), marker="o")
    plt.plot((x1, x2), (y2, y2), marker="o")
    plt.plot((x2, x2), (y1, y2), marker="o")

    # Landmarks
    landmarks = np.array(current_row["landmarks"].split(";"), dtype=int)
    for i in range(0, len(landmarks), 2):
        x = landmarks[i]
        y = landmarks[i + 1]
        plt.scatter(x=x, y=y, s=0.5)

    # Image
    plt.imshow(current_row["img"])
    plt.show()


# Configs
cfg_path = "../../.configs/dynamic_datasets/kassel_emonet.cfg"
cfg = read_configs([cfg_path])
cfg_ds = cfg["DATASET"]

# Paths
dataset_root = Path(cfg_ds.get("nova_data_dir")) / cfg_ds.get("dataset")
outpath = Path("./emonet_predictions")

# Files
file_list = []
for vid in dataset_root.rglob("*.video_resized.mp4"):
    file_list.append(vid)

# Preprocessing modules
face_detector = BlazeFaceDetector()
face_detector._device = "gpu"
bb_cropper = BoundingBoxImageCrop(output_dim=(256, 256))

# Emonet model
emonet = EmoNet8(weights="affectnet")
emonet.build_model()
emonet._model.eval()

# Predict
for file_path in file_list:
    csv_out = outpath / (
        str(file_path.parent.stem) + "-" + str(file_path.stem) + ".csv"
    )

    # Skip existing
    if csv_out.is_file():
        continue

    with open(csv_out, "w", newline="", encoding="utf-8") as csv_file:
        # CSV file
        outpath.mkdir(exist_ok=True)
        header = [
            "frame",
            "face_upper_left",
            "face_lower_right",
            "facial_landmarks",
            "expression",
            "valence",
            "arousal",
        ]
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        current_row = {}

        print(f"Predicting {file_path}")
        vr = VideoReader(str(file_path), ctx=cpu(0))
        for i in tqdm(range(len(vr))):

            img = vr[i].asnumpy()

            # Detect faces
            img_pp,  (xshift, yshift), orig_size = face_detector.detector.preprocessing(img)
            detected_faces = face_detector.detector.predict(np.expand_dims(img_pp, 0))
            detected_faces = face_detector.detector.postprocessing(detected_faces, (xshift, yshift), orig_size)

            # Predict Emotions
            if detected_faces.any():

                # only take face with highest confidence (i.e. index 0
                face_bb = np.squeeze(detected_faces[0, :4])
                img_emo = bb_cropper._internal_call(img, face_bb)
                img_emo = img_emo.astype(dtype=np.float32)
                img_emo = img_emo / 255
                img_emo = np.moveaxis(img_emo, -1, -3)
                out = emonet.predict(img_emo)

                scale, bb_center = bb_cropper.get_scale_center(face_bb)
                lndmarks_rel, lndmarks_img, scores = get_preds_fromhm(
                    hm=out["heatmap"].cpu().numpy(), center=bb_center, scale=scale
                )
                # DEBUG
                # current_row["img"] = img
                #!DEBUG
                face_bb_str = [str(int(x)) for x in face_bb]
                current_row["face_upper_left"] = ";".join(
                    (face_bb_str[0], face_bb_str[1])
                )
                current_row["face_lower_right"] = ";".join(
                    (face_bb_str[2], face_bb_str[3])
                )
                current_row["landmarks"] = ";".join(
                    np.squeeze(lndmarks_img).flatten().astype(np.int).astype(np.str)
                )
                current_row["expression"] = ";".join(
                    [
                        "{:0.3f}".format(x)
                        for x in softmax(np.squeeze(out["expression"].cpu().numpy()))
                    ]
                )
                current_row["valence"] = np.squeeze(out["valence"].cpu().numpy())
                current_row["arousal"] = np.squeeze(out["arousal"].cpu().numpy())

            csv_writer.writerow(
                [
                    i,
                    current_row.get("face_upper_left"),
                    current_row.get("face_lower_right"),
                    current_row.get("landmarks"),
                    current_row.get("expression"),
                    current_row.get("valence"),
                    current_row.get("arousal"),
                ]
            )

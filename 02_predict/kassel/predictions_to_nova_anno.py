import pandas as pd
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as et
import csv
from utils.config_utils import read_configs

# Configs
cfg_path = "../../.configs/dynamic_datasets/kassel_emonet.cfg"
cfg = read_configs([cfg_path])
cfg_ds = cfg["DATASET"]

# Paths
dataset_root = Path(cfg_ds.get("nova_data_dir")) / cfg_ds.get("dataset")
prediction_path = dataset_root / Path("scripts/emonet/predictions")

# Emotion Map
emotions_cat_items = [
    ("neutral", "0", "#FF000000"),
    ("happy", "1", "#FF000000"),
    ("sad", "2", "#FF000000"),
    ("suprise", "3", "#FF000000"),
    ("fear", "4", "#FF000000"),
    ("disgust", "5", "#FF000000"),
    ("anger", "6", "#FF000000"),
    ("contempt", "7", "#FF000000"),
    # ('none', '8', '#FF000000'),
    # ('uncertain', '9', '#FF000000'),
    # ('non-face', '10', '#FF000000')
]


def to_nova(
    data: pd.DataFrame,
    annotator: str = "SYSTEM",
    role: str = "role",
    scheme_name: str = "scheme_name",
    scheme_type: str = "DISCRETE",
    # Continuous
    sr: int = 25,
    min_val: int = -1,
    max_val: str = 1,
    min_color: str = "#FFFFFFFF",
    max_color: str = "#FF4F81BD",
    color: str = "#FF008000",
    out_dir: str = ".",
    # Point
    num: int = 0,
    # Discrete
    items: list = None,  # [('<label_name>',  '<label_id>', '<label_color>')...]
):
    # Header
    header_template = """
    <?xml version="1.0"?>
    <annotation ssi-v="3">
    <info ftype="ASCII" size="0"/>
    <meta annotator="" role=""/> 
    <scheme />
    </annotation>"""

    header_dom = et.ElementTree(et.fromstring(header_template.strip()))

    # Setting attributes
    info = header_dom.find("info")
    info.set("size", str(len(data)))

    meta = header_dom.find("meta")
    meta.set("annotator", annotator)
    meta.set("role", role)

    scheme = header_dom.find("scheme")
    scheme.set("name", scheme_name)
    scheme.set("type", scheme_type)

    if scheme_type == "CONTINUOUS":
        scheme.set("sr", str(sr))
        scheme.set("min_val", str(min_val))
        scheme.set("max_val", str(max_val))
        scheme.set("min_color", min_color)
        scheme.set("max_color", max_color)

    elif scheme_type == "DISCRETE":
        scheme.set("color", "#FFFFFFFF")
        for n, i, c in items:
            child = et.SubElement(scheme, "item")
            child.set("name", n)
            child.set("id", i)
            child.set("color", c)

    elif scheme_type == "FREE":
        # TODO#
        raise NotImplementedError
    elif scheme_type == "POINT":
        scheme.set("sr", str(sr))
        scheme.set("color", color)
        scheme.set("num", str(num))
    else:
        print(f"Unsupported scheme {scheme_name}")
        raise ValueError()

    # Writing output
    if not Path(out_dir).is_dir():
        raise IOError

    fn = f"{role}.{scheme_name}.{annotator}.annotation"
    out_path_header = Path(out_dir) / fn
    out_path_data = out_path_header.with_suffix(".annotation~")

    et.indent(header_dom, space="\t", level=0)
    header_dom.write(out_path_header, encoding="utf-8", xml_declaration=True)

    if scheme_type == "DISCRETE":
        data[["start", "end", "id", "conf"]].to_csv(
            out_path_data,
            encoding="UTF8",
            sep=";",
            header=None,
            index=False,
            float_format="%.2f",
        )
    elif scheme_type == "CONTINUOUS":
        data[["score", "conf"]].to_csv(
            out_path_data,
            encoding="UTF8",
            sep=";",
            header=None,
            index=False,
            float_format="%.2f",
        )

    elif scheme_type == "POINT":
        with out_path_data.open("w", encoding="utf-8") as csvfile:
            spamwriter = csv.writer(
                csvfile,
                delimiter=";",
            )
            for index, row in data.iterrows():
                new_row = []
                for col in row:
                    new_row.append(str(col).replace(",", ":").replace(" ", ""))
                spamwriter.writerow([index] + new_row + ["1"])


def expand_bb_points(upper_left, lower_right):

    x1, y1 = [int(x) for x in upper_left.split(";")]
    x2, y2 = [int(x) for x in lower_right.split(";")]

    return (x1, y1), (x1, y2), (x2, y1), (x2, y2)


for csv_file in prediction_path.glob("*.csv"):
    print(f"Converting {csv_file}")
    df = pd.read_csv(csv_file)

    # Cateogorical Expression
    df_expression = df["expression"]
    df_expression = df_expression.str.split(
        ";", len(emotions_cat_items), expand=True
    ).astype(float)

    # Apply temporal smoothing as done in the emonet code.
    # Note: In the emonet code this is done before the softmax application, but we do not have this information anymore.
    tw = [0.1, 0.1, 0.15, 0.25, 0.4]
    df_expression = df_expression.rolling(len(tw)).apply(lambda x: np.sum(tw * x))
    df_expression = df_expression.fillna(-1)
    df_expression = pd.concat(
        [df_expression.idxmax(axis=1).astype(int), df_expression.max(axis=1)], axis=1
    )
    df_expression.columns = ["id", "conf"]
    # Setting garbage labels
    df_expression.loc[df_expression["conf"] == -1] = -1
    df_expression = df_expression.reset_index()

    # Convert to sparse data
    df_expression = df_expression.groupby(
        [(df_expression["id"] != df_expression["id"].shift()).cumsum()]
    ).agg({"index": ["min", "max"], "id": "mean", "conf": "mean"})
    df_expression["start"] = df_expression["index"]["min"].divide(25)
    df_expression["end"] = df_expression["index"]["max"].divide(25)
    df_expression["id"] = df_expression["id"].astype(int)


    # Valence / Arousal
    df_valence = pd.DataFrame()
    # if prediction does not exist we add confidence zero
    df_valence["score"] = df["valence"].fillna(0)
    df_valence["conf"] = df["valence"].notnull().astype(int)

    df_arousal = pd.DataFrame()
    # if prediction does not exist we add confidence zero
    df_arousal["score"] = df["arousal"].fillna(0)
    df_arousal["conf"] = df["arousal"].notnull().astype(int)

    # Preprocess bounding boxes
    df_bb = pd.concat([df["face_upper_left"], df["face_lower_right"]], axis=1).fillna(
        "-1;-1"
    )
    df_bb = pd.DataFrame(
        df_bb.apply(
            lambda x: [
                (i + 1,) + p + (1,)
                for i, p in enumerate(
                    expand_bb_points(x["face_upper_left"], x["face_lower_right"])
                )
            ],
            axis=1,
        ).to_list()
    )
    df_bb = df_bb.rename(index={x: f"Frame {x+1}" for x in df_bb.index})

    # Preprocess Landmarks
    n_landmarks = 68
    df_lm = df["facial_landmarks"].fillna(";".join(["-1"] * n_landmarks * 2))
    df_lm = df_lm.str.split(";", n_landmarks * 2, expand=True).astype(int)

    # Looooots of reshaping
    x = df_lm[list(df_lm.columns[0::2])].values
    y = df_lm[list(df_lm.columns[1::2])].values
    i = np.tile(np.arange(n_landmarks), len(df_lm)).reshape(x.shape)
    c = np.tile(np.ones(n_landmarks), len(df_lm)).reshape(x.shape)
    np_lm = np.stack((i, x, y, c)).swapaxes(0, -1).swapaxes(0, 1)
    df_lm = pd.DataFrame(map(lambda x: tuple(x), np_lm))
    df_lm = df_lm.applymap(lambda x: tuple(x.astype(int)))

    # Saving annotations in nova format
    session, role = csv_file.stem.split(".")[0].split("-")

    to_nova(
        df_expression,
        annotator="emonet",
        role=role,
        scheme_name="emotion_categorical",
        items=emotions_cat_items,
        out_dir=".",
    )

    exit()
    to_nova(
        df_bb,
        annotator="emonet",
        role=role,
        scheme_name="bounding_box",
        scheme_type="POINT",
        sr=25,
        num=4,
        out_dir=".",
    )
    to_nova(
        df_lm,
        annotator="emonet",
        role=role,
        scheme_name="facial_landmarks",
        scheme_type="POINT",
        sr=25,
        num=68,
        out_dir=".",
    )
    to_nova(
        df_valence,
        annotator="emonet",
        role=role,
        scheme_name="valence25",
        scheme_type="CONTINUOUS",
        sr=25,
        out_dir=dataset_root / session,
    )
    to_nova(
        df_arousal,
        annotator="emonet",
        role=role,
        scheme_name="arousal25",
        scheme_type="CONTINUOUS",
        sr=25,
        out_dir=dataset_root / session,
    )

    print(f"... done.")

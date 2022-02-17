import pandas as pd
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as et
from utils.config_utils import read_configs

# Configs
cfg_path = "../../.configs/dynamic_datasets/kassel_emonet.cfg"
cfg = read_configs([cfg_path])
cfg_ds = cfg["DATASET"]

# Paths
dataset_root = Path(cfg_ds.get("nova_data_dir")) / cfg_ds.get("dataset")
prediction_path = dataset_root / Path('scripts/emonet/predictions')

def to_nova(
    data: pd.DataFrame,
    annotator: str = "SYSTEM",
    role: str = "role",
    scheme_name: str = "scheme_name",
    scheme_type: str = "DISCRETE",
    sr: int = 25,
    min_val: int = -1,
    max_val: str = 1,
    min_color: str = "#FFFFFFFF",
    max_color: str = "#FF4F81BD",
    out_dir: str = ".",
):
    # Header
    header_template = """
    <?xml version="1.0"?>
    <annotation ssi-v="3">
    <info ftype="ASCII" size="0"/>
    <meta annotator="" role=""/> 
    <scheme name="valence" type="CONTINUOUS" sr="25" min="-1" max="1" mincolor="#FFFFFFFF" maxcolor="#FF4F81BD"/>
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
        # TODO
        raise NotImplementedError
    elif scheme_type == "FREE":
        # TODO#
        raise NotImplementedError
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

for csv in prediction_path.glob("*.csv"):
    print(f'Converting {csv}')
    df = pd.read_csv(csv)

    df_valence = pd.DataFrame()
    # if prediction does not exist we add confidence zero
    df_valence["score"] = df["valence"].fillna(0)
    df_valence["conf"] = df["valence"].notnull().astype(int)

    df_arousal = pd.DataFrame()
    # if prediction does not exist we add confidence zero
    df_arousal["score"] = df["arousal"].fillna(0)
    df_arousal["conf"] = df["arousal"].notnull().astype(int)

    # Fix bounding boxes
    # df['bb'] = df['bb'].replace('[ ', '[').replace(' ]', ']').replace('  ', '').replace(' ', ',')
    # arousal

    # expression
    df_expression = df["expression"]
    df_expression = df_expression.fillna("-1;-1")
    df_expression = df_expression.astype(str).apply(
        lambda x: (
            np.argmax([float(y) for y in x.split(";")]),
            np.amax([float(y) for y in x.split(";")]),
        )
    )  # if it works, it works!
    df_expression = pd.DataFrame(df_expression.tolist(), index=df.index)

    # potentially apply temporal smoothing?
    # also group labels together and add duration
    # TODO

    # saving annotations in nova format
    session, role = csv.stem.split('.')[0].split('-')

    to_nova(df_valence, annotator='emonet', role=role, scheme_name='valence25', scheme_type='CONTINUOUS', sr=25, out_dir=dataset_root / session)
    to_nova(df_arousal, annotator='emonet', role=role, scheme_name='arousal25', scheme_type='CONTINUOUS', sr=25, out_dir=dataset_root / session)

    print(f'... done.')

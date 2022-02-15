from pathlib import Path

import pandas as pd
import numpy as np
import ast
from utils.config_utils import read_configs

# Configs
cfg_path = '../../.configs/dynamic_datasets/kassel_emonet.cfg'
cfg = read_configs([cfg_path])
cfg_ds = cfg['DATASET']

# Paths
dataset_root = Path(cfg_ds.get('nova_data_dir')) / cfg_ds.get('dataset')
outpath = Path('./emonet_predictions')

for csv in outpath.glob('*.csv'):
    df = pd.read_csv(csv)

    df_valence = pd.DataFrame()
    # if prediction does not exist we add confidence zero
    df_valence['valence'] = df['valence'].fillna(0)
    df_valence['conf'] = df['valence'].notnull().astype(int)

    df_arousal = pd.DataFrame()
    # if prediction does not exist we add confidence zero
    df_arousal['arousal'] = df['arousal'].fillna(0)
    df_arousal['conf'] = df['arousal'].notnull().astype(int)
    breakpoint()


    # Fix bounding boxes
    #df['bb'] = df['bb'].replace('[ ', '[').replace(' ]', ']').replace('  ', '').replace(' ', ',')
    # arousal
    df_arousal = df['expression']
    df_arousal['conf'] = df['expression'].notnull().astype(int)


    # expression
    df_expression = df['expression']
    df_expression = df_expression.fillna('-1;-1')
    df_expression = df_expression.astype(str).apply(lambda x : (np.argmax([float(y) for y in x.split(';')]), np.amax([float(y) for y in x.split(';')]))) #if it works, it works!
    df_expression = pd.DataFrame(df_expression.tolist(), index=df.index)

    # potentially apply temporal smoothing?
    # also group labels together and add duration
    #TODO


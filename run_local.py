import os

import pandas as pd

data_path = os.path.join(
    "local", "btc-all.csv"
)

df = pd.read_csv(data_path)


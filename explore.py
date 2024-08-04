import os
import pandas as pd

from space_time_modeling.fe import ClassificationFE

# statics 
label_column = "signal"
control_column = "scraped_timestamp"
target_column = "price" 

df = pd.read_csv("btc-all-fe.csv")

fe = ClassificationFE(
    control_column = control_column,
    target_column = target_column,
    label = label_column,
)


info = fe.mutual_info(
    df = df
).head(15)['feature'].to_list()

print(info)
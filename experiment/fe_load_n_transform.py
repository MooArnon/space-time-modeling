import pandas as pd

from space_time_modeling.utilities import load_instance

df = pd.read_csv("tests/df_test.csv")

instance = load_instance(
    "fe_15lag_3-9-12-15-30rolling_percent-change_3-9-12-15-30rsi/fe_15lag_3-9-12-15-30rolling_percent-change_3-9-12-15-30rsi_20240203_122758.pkl"
)
instance.set_label = None

print(type(instance))
print(instance.name)
print(instance.fe_name_list)
print(type(instance.fe_name_list))
print(instance.n_lag)
print(instance.n_lag)

df = instance.transform_df(df)
print(df.columns)
print(df.head(5))
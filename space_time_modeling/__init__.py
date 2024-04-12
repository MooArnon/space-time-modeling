"""
# Overview 
This module was constructed to analyze and create a data model, 
machine learning. It consists of 3 main models.
1. `eda` : Create plots and eda elements for target data
2. `fe` : Create an instance of feature engineering at a target data
3. `modeling` : Construct the machine learning model.

# eda
Construct the EDA element on target data.
```python
from space_time_modeling.eda import eda

plots = [
    "data_date_trend", 
    "pair_plot", 
    "acf_plot", 
    "pacf_plot", 
    "rolling_statistics",
    "correlation_plot",
]

plot_attribute = {
    "control_column": "Date",
    "target_column": "Open",
}

eda(
    df = r"result/test.csv",
    store_at = "result/eda_fed",
    plot = plots,
    plot_attribute = plot_attribute
)
```

# fe

```python
import os

from space_time_modeling.fe import fe_engine

df_path = os.path.join("tests", "df_test.csv")
control_column = "Date"
target_column = "Open"
label = "signal"

# Initiate engine
fe = fe_engine(
    df = df_path,
    control_column = control_column,
    target_column = target_column,
    label = label,
    engine = "classification",
)

# label data
df = fe.add_label(
    df = fe.df, 
    target_column = "Open",
)
```

# modeling

```python
import os

from space_time_modeling.modeling import modeling_engine

df_path = os.path.join("result", "preprocessed.csv")
label_column = "signal"
feature_column = [
    'signal', 'lag_1_day', 'lag_2_day','mean_3_day',
    'std_3_day', 'percentage_change', 'rsi_3',
    'rsi_9', 'rsi_12', 'rsi_15', 'rsi_30'
]

modeling = modeling_engine(
    "classification",
    df = os.path.join("tests", "preprocessed.csv"),
    label_column = label_column,
    feature_column = feature_column,
    result_path = os.path.join("result"),
)

modeling.modeling()
```

---

# Project structure
```
project structure

├── README.md
├── __main.py
├── __main.py
├── eda
├    ├── README.md
├    ├── __init__.py
├    ├── __main.py
├── fe
├    ├── README.md
├    ├── __init__.py
├    ├── __main.py
├── modeling
├    ├── README.md
├    ├── __init__.py
├    ├── __main.py
└──────────────────────────
```
"""

__version__ = 0.1
__author__ = "Arnon Phongsiang" 
__email__ = "arnon.phongsiang@gmail.com"
__github__ = "https://github.com/MooArnon"
__medium__ = "https://medium.com/@oomarnon"
__linkedin__ = "https://www.linkedin.com/in/arnon-pongsiang-320796214/"

##############################################################################

# space-time-modeling-new
This module serves the machine learning modeling purpose.  
It includes everything that is needed to construct and analyze the model.  
There are 2 ways to use this package, clone to local and use `pip install`.

# How to install
To install this package...


---

# EDA
## TimeSeriesEDA engine
Engine by default, cause we are mainly working on time series data.
It takes a `.csv` file consisting of feature data, or a label if you have one.
The main feature must be value and control columns, price and timestamp.
This module will create the plotting element and export it to your working directory.
### Available plot
- `data_date_trend`  
    The trend of the data is followed by assigned `control_column` and `target_column` parameters.
- `pair_plot`  
    Plot the correlation between each features
- `acf_plot`
    Autocorrelation Function plot
- `pacf_plot`
    Partial Autocorrelation Function
- `rolling_statistics`
    The rolling stats
- `correlation_plot`
    Correlation between feature
```python
from space_time_modeling.eda import eda

# Assign attribute
plots = ["data_date_trend", "pair_plot", "acf_plot"]
plot_attribute = {
    "control_column": "Date",
    "target_column": "Open",
}

# Run EDA
eda(
    df = r"result/test.csv",
    store_at = "result/eda_fed",
    plot = plots,
    plot_attribute = plot_attribute
)
```

---

# fe
## classification engine
The feature engineering module. It consumes `.csv` file and creates the feature followed by the name of the plot at `fe_name_list`. It can also create the label for a signal for the classification task.
### Available fe
- `lag_df`  
    The past lag value of a target column.
- `rolling_df`  
    Rolling statistics, now we have mean and std.
- `percent_change_df`  
    Percentage of value changing, t0 and t-1
- `rsi_df`  
    RSI over the value.
```python
from space_time_modeling.fe import fe_engine

# Get engine
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

# Get transform
df = fe.transform_df(
    fe_name_list=[
        "lag_df",
        "rolling_df",
        "percent_change_df",
        "rsi_df",
    ]
)

# Export
df.to_csv("path/to/preprocessed_df.csv")
```

# modeling
## classification engine
The engine to model the algorithm for classification.
The `classification` word means only for the ordinary stats algorithms like `XGBoost`, `catboost`, `Random forest` etc.
This will create all model and export it.
```python
from space_time_modeling.modeling import modeling_engine

#  all columns
label_column = "signal"
feature_column = [
    'signal', 'lag_1_day', 'lag_2_day',
    'lag_13_day', 'lag_14_day', 'lag_15_day', 'mean_3_day',
    'std_15_day', 'mean_30_day',  'percentage_change', 
    'rsi_3', 'rsi_9', 'rsi_12', 'rsi_15', 'rsi_30'
]

# Initiate modeling
modeling = modeling_engine(
    "classification",
    df = os.path.join("tests", "preprocessed.csv"),
    label_column = label_column,
    feature_column = feature_column,
    result_path = os.path.join("result"),
)

# Modeling
modeling.modeling()
```

---

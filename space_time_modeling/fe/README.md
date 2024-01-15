# Feature engineering
Create a `pandas.DataFrame`, transformed by specified feature engineering functions.

# Classification engine
Stored an essential classification feature transformations
1. `lag_df`: Past value of `target_column`
2. `rolling_df`: Rolling statistics for `target_column`, included mean and std
3. `percent_change_df`: Compared change percentage from previous `target_column` record
4. `rsi_df`: RSI of `target_column`

## Example
```python
# Import
from from space_time_modeling.fe import engine

# Class's attribute
df_path = os.path.join("path", "to", "df.csv")
control_column = "Date"
target_column = "Open"
label = "signal"

# Initiate engine
fe = engine(
    df = df_path,
    control_column = control_column,
    target_column = target_column,
    label = label,
    engine = "classification",
)

# Add label
df = fe.add_label(
    df = fe.df, 
    target_column = "Open",
)

# Transform data frame with specified fe functions
df = fe.transform_df(
    [
        "lag_df",
        "rolling_df",
        "percent_change_df",
        "rsi_df",
    ]
)
```

# EDA
This sub-package will create the eda of the input data frame. We, now, have 2 engines of eda algorithm.

# TimeSeriesEDA
This engine will plot the figure and save it in the specified directory.

# Available plot
- `data_date_trend` Trend over data
- `pair_plot`: Correlation between column
- `acf_plot`: Autocorrelation of the target column
- `pacf_plot`: Pair autocorrelation of the target column
- `rolling_statistics`: A rolling stats versus target column

# How to use
```python
import os

from space_time_modeling.eda import eda

# Initialize attribute
plots = [
    "data_date_trend", 
    "pair_plot", 
    "acf_plot", 
    "pacf_plot", 
    "rolling_statistics"
]

plot_attribute = {
    "control_column": "Date",
    "target_column": "Open"
}

# Get plot
eda(
        df = os.path.join("path", "to", "data"),
        store_at = os.path.join("path", "to", "result"),
        plot = plots,
        plot_attribute = plot_attribute
    )

```

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"tests/BTC-USD.csv")

sns.pairplot(df)

plt.show()

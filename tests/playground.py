import jax
jax.config.update("jax_enable_x64", True)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

result = pd.read_csv("results/result_2025-05-19_18-34-32.csv")

sns.kdeplot(result.time, fill=True)  # `fill=True` fills the area under the curve
plt.title('Density Plot')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()


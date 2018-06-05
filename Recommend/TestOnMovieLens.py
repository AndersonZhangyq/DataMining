import pandas as pd
import numpy as np

data = pd.read_csv("ratings.csv")

data = data.as_matrix()[::, 0:3]

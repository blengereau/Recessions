import pandas as pd
import numpy as np


# Function to apply transformations based on code
def apply_transformation(series, code):
    if code == 1:
        return series  # No transformation
    elif code == 2:
        return series.diff()  # First difference ∆xt
    elif code == 3:
        return series.diff().diff()  # Second difference ∆²xt
    elif code == 4:
        return np.log(series)  # log(xt)
    elif code == 5:
        return np.log(series).diff()  # ∆ log(xt)
    elif code == 6:
        return np.log(series).diff().diff()  # ∆² log(xt)
    elif code == 7:
        return series / series.shift(1) - 1  # ∆(xt/xt-1 - 1)

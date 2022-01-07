# It's useful to transform your data before filling them
# in a machine learning model (ex. standardization)

# We'll se some useful transforms:

# Power transform:
#   - Raise all your data points to a given power gamma

import numpy as np
from scipy.optimize import brent
import pandas as pd
import matplotlib.pyplot as plt


def power_transform(a: np.ndarray,
                    gamma: float) -> np.ndarray:
    return np.power(a, gamma)

# Log transform
#   - Take the log of all your data points
#   - It does not accept negative ou null values
#   - If your data have a negative minimum, your can
#   subtract it from all elements
#   - It is common to add 1 to all elements, to ensure
#   there are no 0s

def log_transform(a: np.ndarray) -> np.ndarray:
    b = a
    minimum = np.min(b)
    if minimum < 0:
        b -= minimum
    if np.min(b) == 0:
        b += 1
    return np.log(b)

# Box-Cox transform
#   - Generalizes both previous transforms
#   - People try to make the data "more normal",
#   what cannot be done since time series data can have
#   trends.
#   - What we really want is the series to be stationary
#   - Test for best lambda: https://arxiv.org/pdf/1401.3812.pdf
#   - scipy package uses the box-cox negative log-likelihood
#   function


def boxcox_transform(a: np.ndarray,
                     lamb: float = None) -> np.ndarray:
    if lamb is None:

        def _boxcox_negative_log_likelihood(lamb: float) -> float:
            llf1 = (lamb - 1) * np.sum(np.log(a))
            if lamb == 0:
                variance = np.sum(np.power(a - np.mean(a), 2)) / n
            else:
                b = np.power(a, lamb) / lamb
                variance = np.sum(np.power(b - np.mean(b), 2)) / n
            llf2 = (n / 2.) * np.log(variance)
            return - (llf1 - llf2)

        n = a.size
        lamb = brent(_boxcox_negative_log_likelihood,
                     brack=(-2.0, 2.0))
        print(f"Best Lambda = {lamb}")

    if lamb == 0:
        return np.log(a)
    else:
        return (np.power(a, lamb) - 1) / lamb


df = pd.read_csv(r"C:\Users\roger\git\time-series-analysis\files\airline_passengers.csv",
                 index_col="Month",
                 parse_dates=True)

df["Sqrt"] = power_transform(df["Passengers"], 0.5)
df["Log"] = log_transform(df["Passengers"])
df["BoxCox"] = boxcox_transform(df["Passengers"])

df["Passengers"].plot(kind="hist", bins=20)
plt.show()
df["Sqrt"].plot(kind="hist", bins=20)
plt.show()
df["Log"].plot(kind="hist", bins=20)
plt.show()
df["BoxCox"].plot(kind="hist", bins=20)
plt.show()

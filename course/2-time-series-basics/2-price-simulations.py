import numpy as np
import matplotlib.pyplot as plt

# We will se how to generate random price
# series simulations for evaluating future models.

# Number of time steps
T = 1000
# Initial price
p0 = 10
# Drift
mu = 1e-3

# Last log price
last_p = np.log(p0)

log_returns = np.zeros(T)
prices = np.zeros(T)
for t in range(T):
    # Normal random return (noise)
    r = 0.01 * np.random.randn()

    # Compute new log price
    p = last_p + mu + r

    # Store and return the price
    log_returns[t] = r + mu
    prices[t] = np.exp(p)

    # Update last price
    last_p = p

plt.figure(figsize=(16, 9))
plt.plot(prices)
plt.show()

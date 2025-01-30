<h1 align="center"><em>fsfa</em>: feature-based slow feature analysis</h1>

<p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" height="20"/></a>
 	  <a href="https://twitter.com/compTimeSeries"><img src="https://img.shields.io/twitter/url/https/twitter.com/compTimeSeries.svg?style=social&label=Follow%20%40compTimeSeries" height="20"/></a>
</p>


<p align="center"><img src="img/outline.png" height="400"/></p>

## About

[_Feature-based slow feature analysis (f-SFA)_](https://github.com/KieranOwens/fsfa) is an approach to parameter inference from a non-stationary unknown process ([PINUP](https://arxiv.org/abs/2407.08987)) comprising sliding-window computation of multivariate time series of canonical time-series features followed by dimension reduction using slow feature analysis.

Given a time series measured from a non-stationary process for which variation in dynamical properties is governed by some time-varying parameter (TVP), f-SFA is an unsupervised method for inferring the TVP directly from the time series.

This package provides a python implementation licensed under an MIT license.


### Acknowledgement :

If you use this software, please read and cite this open-access article:

- &#x1F4D7; Owens et al. [Parameter inference from a non-stationary unknown process using statistical feature-based slow feature analysis](url), _arXiv_ (In Preparation).

For details on the parameter inference from a non-stationary unknown process (PINUP) problem see:

- &#x1F4D7; Owens et al. [Parameter inference from a non-stationary unknown process](https://doi.org/10.1063/5.0228236), _Chaos: An Interdisciplinary Journal of Nonlinear Science_ (2024).

## Installation

Using `pip` for [`fsfa`](https://pypi.org/project/fsfa/):

```
pip install fsfa
```

## Usage

```python
import fsfa

# dataset for analysis
X = np.random.random((10000, 2)) # (or more interesting data)

# option 1: compute a time series of time-series features
df = fsfa.feature_embedding(X, window=1000, stride=100)

# option 2: f-SFA
fsfa1 = fsfa.fSFA(X, window=1000, stride=100, n_components=1)

```

## Example: non-stationary logistic map

Here we use the chaotic logistic map to simulate a non-stationary process:

```python
import fsfa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# logistic map function
def logistic_map(x, r):
    return r*x*(1-x)

# time varying parameter (tvp)
T = 10**5
time = np.linspace(0, 1, T, endpoint=0)
tvp = np.sin(2*np.pi*5*time) +\
      np.sin(2*np.pi*11*time) +\
      np.sin(2*np.pi*13*time)

# non-stationary time series with initial condition x=0.6
X = np.zeros((T, 1), 'd')
X[0] = 0.6
for i in range(1,T):
    X[i] = logistic_map(X[i-1],3.6+0.13*tvp[i])
```

Plot the time-varying parameter and non-stationary logistic map process:

```python
# plot tvp and X
fig = plt.figure(layout="constrained", figsize=(8, 4))
mosaic = """
    A
    B
    """
ax = fig.subplot_mosaic(mosaic)

ax["A"].plot(range(T), tvp, linewidth=1)
ax["A"].set_ylabel('r')

ax["B"].scatter(range(T), X, s=0.01, c='black')
ax["B"].set_ylabel('x')
ax["B"].set_xlabel('Timestep')

for loc in "AB":
    ax[loc].spines['top'].set_visible(False)
    ax[loc].spines['right'].set_visible(False)

```

<p align="center"><img src="img/logistic.png" height="300"/></p>

Use f-SFA to infer a time varying parameter directly from the time-series data:
```python
# single-component f-SFA
window, stride = 1000, 100
fsfa1 = fsfa.fSFA(X, window=window, stride=stride, n_components=1)
```

Compute mean values of the time-varying parameter over the same windows as f-SFA and use linear regression to align the f-SFA and TVP for plotting:

```python
# time points at which to evaluate window features
time_points = np.arange(0, T - window + 1, stride)
window_indices = np.array([np.arange(t, t + window) for t in time_points])
tvp_means = [np.mean(tvp[indices]) for indices in window_indices]

# linear projection to align the time series for plotting
model = LinearRegression().fit(X=fsfa1, y=tvp_means)
projection = model.predict(fsfa1)

# plot the TVP and the f-SFA component
fig, ax = plt.subplots(figsize=(8,2))
ax.plot(tvp_means, c='black', lw=2, label='r')
ax.plot(projection, label='f-SFA', lw=2, linestyle='--', c='#3fadaf')
ax.set_xlabel('Window')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# compute Pearson R^2 correlation
R2 = round(np.corrcoef(tvp_means, fsfa1.flatten())[0,1]**2, 3)
ax.text(800, -2.3, r'$R^2=$' + f'{R2}')
```

<p align="center"><img src="img/fsfa.png" height="200"/></p>

## Usage notes

- The default time-series featureset used in f-SFA is _catch24_, but there are other keyword options ('mean', 'meanvar', 'quatiles', 'stft', 'catch24'). Alternatively, a list of time-series statistic functions can be passed as a list.
- `fsfa` uses the `sklearn-sfa` package to perform SFA. Keyword arguments to the underlying SFA function can be passed to fSFA using a dictionary. Important SFA arguments can be found in the `sklearn-sfa` documentation, including fill_mode, svd_solver, and robustness_cutoff.


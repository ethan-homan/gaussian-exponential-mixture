# Gaussian-Exponential Mixture

## Modification details

This is a forked version of the Gaussian Exponential Mixture Library written by from Ethan Holman.  Main changes are:

* Support for a Gaussian/Exponential mixture model where the exponential offset parameter is the same at the Gaussion Mu parameter.  This forms a simplified model for the Noise output of an EMCCD as detailed on the paper 'High Frame Rate Imaging Based Photometry" by Harpsoe et Al. Some changes to fix the numerical instability
* Fixing some Numerical instability.  High values in the distribution cause NaN inthe distribution likeilhoods.  High values are automatically assign to the exponential tail in this instance  

## Introduction

Like the name suggests, this package can be used to quickly fit a mixture of an
exponential distribution and a gaussian distribution to some data. This works with
numpy arrays, so you can easily add this to a Jupyter notebook style analysis.

## Motivation

This is a pattern that I have seen in data where there are boundaries to data
and a clear edge distribution forms on the practical lower bound of the global
distribution while a more symmetric population forms somewhere well clear of an edge.

The main motivation for this was for modeling the distribution of a very specific
kind of metric of the form "proportion of elements in a set of groups X that have property Y"
where X is of high cardinality (more than 100 groups) and Y is noisy, but will have two distinct
populations.

## Installing

This requires python 3.6 +

```shell script
git clone https://github.com/ethanwh/gaussian-exponential-mixture.git
cd gaussian-exponential-mixture
pip install .
```

## Usage
```python
import numpy
from gaussian_exponential_mixture import GaussianExponentialMixture

beta, mu, sigma = 1, 10, 1

exponential_data = numpy.random.exponential(scale=beta, size=500)
gaussian_data = numpy.random.normal(loc=mu, scale=sigma, size=500)
mixed_data = numpy.append(exponential_data, gaussian_data)
mix = GaussianExponentialMixture(mixed_data)
mix.fit()

print(mix.parameters)
```

```
beta: 1.02511 | mu: 9.97145 | sigma: 1.04869 | proportion: 0.50000
```

To see the results next to the data you can plot the fit distribution.

```python
from matplotlib import pyplot as plt
x = numpy.arange(20, step=0.2)
plt.plot(x, mix.pdf(x))
plt.hist(mixed_data, density=True, bins=50)
plt.show()
```
![image](https://user-images.githubusercontent.com/19494792/67649025-ca927e00-f90d-11e9-8658-068148e893a6.png)

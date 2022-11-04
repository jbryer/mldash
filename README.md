
# `mldash`: Machine Learning Dashboard <img src="man/figures/mldash.png" align="right" width="120" align="right" />

<!-- badges: start -->

[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
<!-- badges: end -->

**Contact: [Jason Bryer, Ph.D.](mailto:jason.bryer@cuny.edu)**  
**Website: <https://jbryer.github.io/mldash/>**

The goal of `mldash` is to provide a framework for evaluating the
performance of many predictive models across many datasets. The package
includes common predictive modeling procedures and datasets. Details on
how to contribute additional datasets and models is outlined below. Both
datasets and models are defined in the Debian Control File (dcf) format.
This provides a convenient format for storing both metadata about the
datasets and models but also R code snippets for retrieving data,
training models, and getting predictions. The `run_models` function
handles executing each model for each dataset (appropriate to the
predictive model type, i.e. classification or regression), splitting
data into training and validation sets, and calculating the desired
performance metrics utilizing the
[`yardstick`](https://yardstick.tidymodels.org) package.

## Installation

You can install the development version of `mldash` using the `remotes`
package like so:

``` r
remotes::install_github('jbryer/mldash')
```

The `mldash` package makes use of predictive models implemented in R,
Python, and Java. As a result, there are numerous system requirements
necessary to run *all* the models. We have included instructions in the
[`installation`
vignette](https://jbryer.github.io/mldash/articles/installation.html):

``` r
vignette('installation', package = 'mldash')
```

## Running Predictive Models

To begin, we read in the datasets using the `read_ml_datasets()`
function. There are two parameters:

-   `dir` is the directory containing the metadata files. The default is
    to look in the package’s installation directory.
-   `cache_dir` is the directory where datasets can be stored locally.

This lists the datasets currenlty included in the package.

``` r
ml_datasets <- mldash::read_ml_datasets(dir = 'inst/datasets',
                                        cache_dir = 'inst/datasets')
# head(ml_datasets, n = 4)
```

Similarly, the `read_ml_models` will read in the models. The `dir`
parameter defines where to look for model files.

``` r
ml_models <- mldash::read_ml_models(dir = 'inst/models')
#> Warning in mldash::read_ml_models(dir = "inst/models"): The following packages
#> are not installed but required by the models: FCNN4R, mxnet
# head(ml_models, n = 4)
```

Once the datasets and models have been loaded, the `run_models` will
train and evaluate each model for each dataset as appropriate for the
model type.

``` r
ml_results <- mldash::run_models(datasets = ml_datasets, 
                                 models = ml_models, 
                                 seed = 2112)
```

The `metrics` parameter to `run_models()` takes a list of metrics from
the [`yardstick`](https://yardstick.tidymodels.org/index.html) package
(Kuhn & Vaughan, 2021). The full list of metrics are available here:
<https://yardstick.tidymodels.org/articles/metric-types.html>

## Available Datasets

There are 27 included in the `mldash` package. You can view the packages
in the [`datasets`
vignette](https://jbryer.github.io/mldash/articles/datasets.html).

``` r
vignette('datasets', package = 'mldash')
```

## Available Models

Each model is defined in a Debian Control File (DCF) format the details
of which are described below. Below is the list of models included in
the `mldash` package. Note that models that begin with `tm_` are models
implemented with the [`tidymodels`](https://www.tidymodels.org) R
package; models that begin with `weka_` are models implemented with the
the [`RWeka`](https://cran.r-project.org/web/packages/RWeka/index.html)
which is a wrapper to the [Weka](https://www.cs.waikato.ac.nz/ml/weka/)
collection of machine learning algorithms.

There are 413 included in the `mldash` package. You can view the models
in the [`models`
vignette](https://jbryer.github.io/mldash/articles/models.html).

``` r
vignette('models', package = 'mldash')
```

## Code of Conduct

Please note that the mldash project is released with a [Contributor Code
of
Conduct](https://contributor-covenant.org/version/2/0/CODE_OF_CONDUCT.html).
By contributing to this project, you agree to abide by its terms.

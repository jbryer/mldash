
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

**WARNING** This is very much an alpha project as I explore this
approach to evaluating predictive models. Use at your own risk.

## Installation

You can install the development version of `mldash` using the `remotes`
package like so:

``` r
remotes::install_github('jbryer/mldash')
```

### Python

Many of the models will require Python which is executed using the
`reticulate` package. I, personally, have found the installation and
configuration of Python to be frustrating, especially on a Mac M1.
However, as of this writing, the following works (on my system). First,
install these packages from Github to ensure the latest version.

``` r
remotes::install_github(sprintf("rstudio/%s", c("reticulate", "tensorflow", "keras", "torch")))
```

If you have previously installed Miniconda, it is helpful to start from
a clean slate.

``` r
reticulate::miniconda_uninstall()
```

We can then install Miniconda using the following command:

``` r
reticulate::install_miniconda()
```

Once installed, we can create a conda environment:

``` r
reticulate::conda_create("mldash")
```

And then make it active (note sure if it is necessary to do this for all
three packages, but it doesn’t hurt):

``` r
reticulate::use_condaenv("mldash")
#> Warning in reticulate::use_condaenv("mldash"): multiple Conda environments found; the first-listed will be chosen.
#>     name                                                         python
#> 2 mldash /Users/jbryer/Library/r-miniconda-arm64/envs/mldash/bin/python
#> 6 mldash                /Users/jbryer/miniforge3/envs/mldash/bin/python
#> Warning: The request to `use_python("/Users/jbryer/Library/r-miniconda-arm64/
#> envs/mldash/bin/python")` will be ignored because the environment variable
#> RETICULATE_PYTHON is set to "~/miniforge3/envs/mldash/bin/python"
tensorflow::use_condaenv("mldash")
#> Warning in tensorflow::use_condaenv("mldash"): multiple Conda environments found; the first-listed will be chosen.
#>     name                                                         python
#> 2 mldash /Users/jbryer/Library/r-miniconda-arm64/envs/mldash/bin/python
#> 6 mldash                /Users/jbryer/miniforge3/envs/mldash/bin/python

#> Warning in tensorflow::use_condaenv("mldash"): The request to `use_python("/Users/jbryer/Library/r-miniconda-arm64/envs/mldash/bin/python")` will be ignored because the environment variable RETICULATE_PYTHON is set to "~/miniforge3/envs/mldash/bin/python"
keras::use_condaenv("mldash")
#> Warning in keras::use_condaenv("mldash"): multiple Conda environments found; the first-listed will be chosen.
#>     name                                                         python
#> 2 mldash /Users/jbryer/Library/r-miniconda-arm64/envs/mldash/bin/python
#> 6 mldash                /Users/jbryer/miniforge3/envs/mldash/bin/python

#> Warning in keras::use_condaenv("mldash"): The request to `use_python("/Users/jbryer/Library/r-miniconda-arm64/envs/mldash/bin/python")` will be ignored because the environment variable RETICULATE_PYTHON is set to "~/miniforge3/envs/mldash/bin/python"
```

Although there are utility functions to install `keras`, `tensorflow`,
and `torch` from their respective packages, I found them to not always
work as expected. The `conda_install` function will ensure the Python
packages are installed into the correct environment. Note that as of
this writing, `pytorch` still does not have a Mac M1 native version so
some predictive models will not work on that platform.

``` r
reticulate::conda_install("mldash", 
                          c("jupyterlab", "pandas", "statsmodels",
                            "scipy", "scikit-learn", "matplotlib",
                            "seaborn", "numpy", "pytorch", "tensorflow"))
```

Lastly, ensure that `reticulate` uses the correct Python by setting the
`RETICULATE_PYTHON` environment variable (this can also be put in your
`.Renviron` file to be used across sessions, though I avoid doing that
so I can use different Python paths for different projects).

``` r
Sys.setenv("RETICULATE_PYTHON" = "~/miniforge3/envs/mldash/bin/python")
```

## Running Predictive Models

To begin, we read in the datasets using the `read_ml_datasets()`
function. There are two parameters:

-   `dir` is the directory containing the metadata files. The default is
    to look in the package’s installation directory.
-   `cache_dir` is the directory where datasets can be stored locally.

This lists the datasets currenlty included in the package (more to come
soon).

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
ml_results <- mldash::run_models(datasets = ml_datasets, models = ml_models, seed = 1234)
```

The `metrics` parameter to `run_models()` takes a list of metrics from
the [`yardstick`](https://yardstick.tidymodels.org/index.html) package
(Kuhn & Vaughan, 2021). The full list of metris is available here:
<https://yardstick.tidymodels.org/articles/metric-types.html>

## Available Datasets

-   [abalone](inst/datasets/1) - Predicting the age of abalone from
    physical measurements.
-   [acute_inflammation](inst/datasets/2) - The data was created by a
    medical expert as a data set to test the expert system, which will
    perform the presumptive diagnosis of two diseases of the urinary
    system.
-   [adult](inst/datasets/3) - Predict whether income exceeds \$50K/yr
    based on census data. Also known as “Census Income” dataset.
-   [air](inst/datasets/4) - Contains the responses of a gas multisensor
    device deployed on the field in an Italian city. Hourly responses
    averages are recorded along with gas concentrations references from
    a certified analyzer.
-   [ames](inst/datasets/5) - Ames Housing Data.
-   [appliances_energy](inst/datasets/6) - Experimental data used to
    create regression models of appliances energy use in a low energy
    building.
-   [audit](inst/datasets/7) - Exhaustive one year non-confidential data
    in the year 2015 to 2016 of firms is collected from the Auditor
    Office of India to build a predictor for classifying suspicious
    firms.
-   [bike_sharing_day](inst/datasets/8) - Predication of daily bike
    rental count based on the environmental and seasonal settings
-   [breast_cancer](inst/datasets/9) - Predict malignant or benign for
    in breast cancer patients
-   [cervical_cancer](inst/datasets/10) - The dataset contains 19
    attributes regarding ca cervix behavior risk with class label is
    ca_cervix with 1 and 0 as values which means the respondent with and
    without ca cervix, respectively. predictor for classifying
    suspicious firms.
-   [cmc](inst/datasets/11) - The problem is to predict the current
    contraceptive method choice (no use, long-term methods, or
    short-term methods) of a woman based on her demographic and
    socio-economic characteristics.
-   [credit_card_app](inst/datasets/12) - This data concerns credit card
    applications; good mix of attributes.
-   [energy](inst/datasets/13) - Experimental data used to create
    regression models of appliances energy use in a low energy building.
-   [hs_graduate_earnings](inst/datasets/14) - Predicting high school
    graduates median earnings based on their occupational industries
-   [mars_weather](inst/datasets/15) - Mars Weather
-   [microsoft_stock_price](inst/datasets/16) - Microsoft stock price
    from 2001 to the beginning of 2021
-   [mtcars](inst/datasets/17) - Motor Trend Car Road Tests
-   [natural_gas_prices](inst/datasets/18) - Time series of major
    Natural Gas Prices including US Henry Hub. Data comes from U.S.
    Energy Information Administration EIA.
-   [PedalMe](inst/datasets/19) - A dataset about the number of weekly
    bicycle package deliveries by Pedal Me in London during 2020 and
    2021.
-   [psych_copay](inst/datasets/20) - Copay modes for established
    patients in US zip codes
-   [sales](inst/datasets/21) - This is a transnational data set which
    contains all the transactions for a UK-based online retail.
-   [seattle_weather](inst/datasets/22) - Seattle Weather
-   [sp500](inst/datasets/23) - Standard and Poor’s (S&P) 500 Index Data
    including Dividend, Earnings and P/E Ratio.
-   [tesla_stock_price](inst/datasets/24) - Standard and Poor’s (S&P)
    500 Index Data including Dividend, Earnings and P/E Ratio.
-   [titanic](inst/datasets/25) - The original Titanic dataset,
    describing the survival status of individual passengers on the
    Titanic.
-   [traffic](inst/datasets/26) - Hourly Minneapolis-St Paul, MN traffic
    volume for westbound I-94. Includes weather and holiday features
    from 2012-2018.
-   [wine](inst/datasets/27) - The analysis determined the quantities of
    13 constituents found in each of the three types of wines.

## Available Models

Each model is defined in a Debian Control File (DCF) format the details
of which are described below. Below is the list of models included in
the `mldash` package. Note that models that begin with `tm_` are models
implemented with the [`tidymodels`](https://www.tidymodels.org) R
package; models that begin with `weka_` are models implemented with the
the [`RWeka`](https://cran.r-project.org/web/packages/RWeka/index.html)
which is a wrapper to the [Weka](https://www.cs.waikato.ac.nz/ml/weka/)
collection of machine learning algorithms.

-   [Boosted Classification
    Trees](inst/models/caret_ada_classification.dcf) - Boosted
    Classification Trees from the caret package.
-   [Bagged AdaBoost](inst/models/caret_AdaBag_classification.dcf) -
    Bagged AdaBoost from the caret package.
-   [AdaBoost Classification
    Trees](inst/models/caret_adaboost_classification.dcf) - AdaBoost
    Classification Trees from the caret package.
-   [AdaBoost.M1](inst/models/caret_AdaBoost.M1_classification.dcf) -
    AdaBoost.M1 from the caret package.
-   [Adaptive Mixture Discriminant
    Analysis](inst/models/caret_amdai_classification.dcf) - Adaptive
    Mixture Discriminant Analysis from the caret package.
-   [Adaptive-Network-Based Fuzzy Inference
    System](inst/models/caret_ANFIS_regression.dcf) -
    Adaptive-Network-Based Fuzzy Inference System from the caret
    package.
-   [Model Averaged Neural
    Network](inst/models/caret_avNNet_classification.dcf) - Model
    Averaged Neural Network from the caret package.
-   [Model Averaged Neural
    Network](inst/models/caret_avNNet_regression.dcf) - Model Averaged
    Neural Network from the caret package.
-   [Naive Bayes Classifier with Attribute
    Weighting](inst/models/caret_awnb_classification.dcf) - Naive Bayes
    Classifier with Attribute Weighting from the caret package.
-   [Tree Augmented Naive Bayes Classifier with Attribute
    Weighting](inst/models/caret_awtan_classification.dcf) - Tree
    Augmented Naive Bayes Classifier with Attribute Weighting from the
    caret package.
-   [Bagged Model](inst/models/caret_bag_classification.dcf) - Bagged
    Model from the caret package.
-   [Bagged Model](inst/models/caret_bag_regression.dcf) - Bagged Model
    from the caret package.
-   [Bagged MARS](inst/models/caret_bagEarth_classification.dcf) -
    Bagged MARS from the caret package.
-   [Bagged MARS](inst/models/caret_bagEarth_regression.dcf) - Bagged
    MARS from the caret package.
-   [Bagged MARS using gCV
    Pruning](inst/models/caret_bagEarthGCV_classification.dcf) - Bagged
    MARS using gCV Pruning from the caret package.
-   [Bagged MARS using gCV
    Pruning](inst/models/caret_bagEarthGCV_regression.dcf) - Bagged MARS
    using gCV Pruning from the caret package.
-   [Bagged Flexible Discriminant
    Analysis](inst/models/caret_bagFDA_classification.dcf) - Bagged
    Flexible Discriminant Analysis from the caret package.
-   [Bagged FDA using gCV
    Pruning](inst/models/caret_bagFDAGCV_classification.dcf) - Bagged
    FDA using gCV Pruning from the caret package.
-   [Generalized Additive Model using
    Splines](inst/models/caret_bam_classification.dcf) - Generalized
    Additive Model using Splines from the caret package.
-   [Generalized Additive Model using
    Splines](inst/models/caret_bam_regression.dcf) - Generalized
    Additive Model using Splines from the caret package.
-   [Bayesian Additive Regression
    Trees](inst/models/caret_bartMachine_classification.dcf) - Bayesian
    Additive Regression Trees from the caret package.
-   [Bayesian Additive Regression
    Trees](inst/models/caret_bartMachine_regression.dcf) - Bayesian
    Additive Regression Trees from the caret package.
-   [Bayesian Generalized Linear
    Model](inst/models/caret_bayesglm_classification.dcf) - Bayesian
    Generalized Linear Model from the caret package.
-   [Bayesian Generalized Linear
    Model](inst/models/caret_bayesglm_regression.dcf) - Bayesian
    Generalized Linear Model from the caret package.
-   [Binary Discriminant
    Analysis](inst/models/caret_binda_classification.dcf) - Binary
    Discriminant Analysis from the caret package.
-   [Boosted Tree](inst/models/caret_blackboost_classification.dcf) -
    Boosted Tree from the caret package.
-   [Boosted Tree](inst/models/caret_blackboost_regression.dcf) -
    Boosted Tree from the caret package.
-   [The Bayesian lasso](inst/models/caret_blasso_regression.dcf) - The
    Bayesian lasso from the caret package.
-   [Bayesian Ridge Regression (Model
    Averaged)](inst/models/caret_blassoAveraged_regression.dcf) -
    Bayesian Ridge Regression (Model Averaged) from the caret package.
-   [Bayesian Ridge
    Regression](inst/models/caret_bridge_regression.dcf) - Bayesian
    Ridge Regression from the caret package.
-   [Bayesian Regularized Neural
    Networks](inst/models/caret_brnn_regression.dcf) - Bayesian
    Regularized Neural Networks from the caret package.
-   [Boosted Linear Model](inst/models/caret_BstLm_classification.dcf) -
    Boosted Linear Model from the caret package.
-   [Boosted Linear Model](inst/models/caret_BstLm_regression.dcf) -
    Boosted Linear Model from the caret package.
-   [Boosted Smoothing
    Spline](inst/models/caret_bstSm_classification.dcf) - Boosted
    Smoothing Spline from the caret package.
-   [Boosted Smoothing Spline](inst/models/caret_bstSm_regression.dcf) -
    Boosted Smoothing Spline from the caret package.
-   [Boosted Tree](inst/models/caret_bstTree_classification.dcf) -
    Boosted Tree from the caret package.
-   [Boosted Tree](inst/models/caret_bstTree_regression.dcf) - Boosted
    Tree from the caret package.
-   [C5.0](inst/models/caret_C5.0_classification.dcf) - C5.0 from the
    caret package.
-   [Cost-Sensitive
    C5.0](inst/models/caret_C5.0Cost_classification.dcf) -
    Cost-Sensitive C5.0 from the caret package.
-   [Single C5.0
    Ruleset](inst/models/caret_C5.0Rules_classification.dcf) - Single
    C5.0 Ruleset from the caret package.
-   [Single C5.0 Tree](inst/models/caret_C5.0Tree_classification.dcf) -
    Single C5.0 Tree from the caret package.
-   [Conditional Inference Random
    Forest](inst/models/caret_cforest_classification.dcf) - Conditional
    Inference Random Forest from the caret package.
-   [Conditional Inference Random
    Forest](inst/models/caret_cforest_regression.dcf) - Conditional
    Inference Random Forest from the caret package.
-   [CHi-squared Automated Interaction
    Detection](inst/models/caret_chaid_classification.dcf) - CHi-squared
    Automated Interaction Detection from the caret package.
-   [SIMCA](inst/models/caret_CSimca_classification.dcf) - SIMCA from
    the caret package.
-   [Conditional Inference
    Tree](inst/models/caret_ctree_classification.dcf) - Conditional
    Inference Tree from the caret package.
-   [Conditional Inference
    Tree](inst/models/caret_ctree_regression.dcf) - Conditional
    Inference Tree from the caret package.
-   [Conditional Inference
    Tree](inst/models/caret_ctree2_classification.dcf) - Conditional
    Inference Tree from the caret package.
-   [Conditional Inference
    Tree](inst/models/caret_ctree2_regression.dcf) - Conditional
    Inference Tree from the caret package.
-   [Cubist](inst/models/caret_cubist_regression.dcf) - Cubist from the
    caret package.
-   [Diagonal Discriminant
    Analysis](inst/models/caret_dda_classification.dcf) - Diagonal
    Discriminant Analysis from the caret package.
-   [DeepBoost](inst/models/caret_deepboost_classification.dcf) -
    DeepBoost from the caret package.
-   [Dynamic Evolving Neural-Fuzzy Inference
    System](inst/models/caret_DENFIS_regression.dcf) - Dynamic Evolving
    Neural-Fuzzy Inference System from the caret package.
-   [Stacked AutoEncoder Deep Neural
    Network](inst/models/caret_dnn_classification.dcf) - Stacked
    AutoEncoder Deep Neural Network from the caret package.
-   [Stacked AutoEncoder Deep Neural
    Network](inst/models/caret_dnn_regression.dcf) - Stacked AutoEncoder
    Deep Neural Network from the caret package.
-   [Linear Distance Weighted
    Discrimination](inst/models/caret_dwdLinear_classification.dcf) -
    Linear Distance Weighted Discrimination from the caret package.
-   [Distance Weighted Discrimination with Polynomial
    Kernel](inst/models/caret_dwdPoly_classification.dcf) - Distance
    Weighted Discrimination with Polynomial Kernel from the caret
    package.
-   [Distance Weighted Discrimination with Radial Basis Function
    Kernel](inst/models/caret_dwdRadial_classification.dcf) - Distance
    Weighted Discrimination with Radial Basis Function Kernel from the
    caret package.
-   [Multivariate Adaptive Regression
    Spline](inst/models/caret_earth_classification.dcf) - Multivariate
    Adaptive Regression Spline from the caret package.
-   [Multivariate Adaptive Regression
    Spline](inst/models/caret_earth_regression.dcf) - Multivariate
    Adaptive Regression Spline from the caret package.
-   [Extreme Learning
    Machine](inst/models/caret_elm_classification.dcf) - Extreme
    Learning Machine from the caret package.
-   [Extreme Learning Machine](inst/models/caret_elm_regression.dcf) -
    Extreme Learning Machine from the caret package.
-   [Elasticnet](inst/models/caret_enet_regression.dcf) - Elasticnet
    from the caret package.
-   [Tree Models from Genetic
    Algorithms](inst/models/caret_evtree_classification.dcf) - Tree
    Models from Genetic Algorithms from the caret package.
-   [Tree Models from Genetic
    Algorithms](inst/models/caret_evtree_regression.dcf) - Tree Models
    from Genetic Algorithms from the caret package.
-   [Random Forest by
    Randomization](inst/models/caret_extraTrees_classification.dcf) -
    Random Forest by Randomization from the caret package.
-   [Random Forest by
    Randomization](inst/models/caret_extraTrees_regression.dcf) - Random
    Forest by Randomization from the caret package.
-   [Flexible Discriminant
    Analysis](inst/models/caret_fda_classification.dcf) - Flexible
    Discriminant Analysis from the caret package.
-   [Fuzzy Rules Using Genetic Cooperative-Competitive Learning and
    Pittsburgh](inst/models/caret_FH.GBML_classification.dcf) - Fuzzy
    Rules Using Genetic Cooperative-Competitive Learning and Pittsburgh
    from the caret package.
-   [Fuzzy Inference Rules by Descent
    Method](inst/models/caret_FIR.DM_regression.dcf) - Fuzzy Inference
    Rules by Descent Method from the caret package.
-   [Ridge Regression with Variable
    Selection](inst/models/caret_foba_regression.dcf) - Ridge Regression
    with Variable Selection from the caret package.
-   [Fuzzy Rules Using Chi’s
    Method](inst/models/caret_FRBCS.CHI_classification.dcf) - Fuzzy
    Rules Using Chi’s Method from the caret package.
-   [Fuzzy Rules with Weight
    Factor](inst/models/caret_FRBCS.W_classification.dcf) - Fuzzy Rules
    with Weight Factor from the caret package.
-   [Simplified TSK Fuzzy
    Rules](inst/models/caret_FS.HGD_regression.dcf) - Simplified TSK
    Fuzzy Rules from the caret package.
-   [Generalized Additive Model using
    Splines](inst/models/caret_gam_classification.dcf) - Generalized
    Additive Model using Splines from the caret package.
-   [Generalized Additive Model using
    Splines](inst/models/caret_gam_regression.dcf) - Generalized
    Additive Model using Splines from the caret package.
-   [Boosted Generalized Additive
    Model](inst/models/caret_gamboost_classification.dcf) - Boosted
    Generalized Additive Model from the caret package.
-   [Boosted Generalized Additive
    Model](inst/models/caret_gamboost_regression.dcf) - Boosted
    Generalized Additive Model from the caret package.
-   [Generalized Additive Model using
    LOESS](inst/models/caret_gamLoess_classification.dcf) - Generalized
    Additive Model using LOESS from the caret package.
-   [Generalized Additive Model using
    LOESS](inst/models/caret_gamLoess_regression.dcf) - Generalized
    Additive Model using LOESS from the caret package.
-   [Generalized Additive Model using
    Splines](inst/models/caret_gamSpline_classification.dcf) -
    Generalized Additive Model using Splines from the caret package.
-   [Generalized Additive Model using
    Splines](inst/models/caret_gamSpline_regression.dcf) - Generalized
    Additive Model using Splines from the caret package.
-   [Gaussian
    Process](inst/models/caret_gaussprLinear_classification.dcf) -
    Gaussian Process from the caret package.
-   [Gaussian Process](inst/models/caret_gaussprLinear_regression.dcf) -
    Gaussian Process from the caret package.
-   [Gaussian Process with Polynomial
    Kernel](inst/models/caret_gaussprPoly_classification.dcf) - Gaussian
    Process with Polynomial Kernel from the caret package.
-   [Gaussian Process with Polynomial
    Kernel](inst/models/caret_gaussprPoly_regression.dcf) - Gaussian
    Process with Polynomial Kernel from the caret package.
-   [Gaussian Process with Radial Basis Function
    Kernel](inst/models/caret_gaussprRadial_classification.dcf) -
    Gaussian Process with Radial Basis Function Kernel from the caret
    package.
-   [Gaussian Process with Radial Basis Function
    Kernel](inst/models/caret_gaussprRadial_regression.dcf) - Gaussian
    Process with Radial Basis Function Kernel from the caret package.
-   [Stochastic Gradient
    Boosting](inst/models/caret_gbm_classification.dcf) - Stochastic
    Gradient Boosting from the caret package.
-   [Gradient Boosting
    Machines](inst/models/caret_gbm_h2o_classification.dcf) - Gradient
    Boosting Machines from the caret package.
-   [Gradient Boosting
    Machines](inst/models/caret_gbm_h2o_regression.dcf) - Gradient
    Boosting Machines from the caret package.
-   [Stochastic Gradient
    Boosting](inst/models/caret_gbm_regression.dcf) - Stochastic
    Gradient Boosting from the caret package.
-   [Multivariate Adaptive Regression
    Splines](inst/models/caret_gcvEarth_classification.dcf) -
    Multivariate Adaptive Regression Splines from the caret package.
-   [Multivariate Adaptive Regression
    Splines](inst/models/caret_gcvEarth_regression.dcf) - Multivariate
    Adaptive Regression Splines from the caret package.
-   [Fuzzy Rules via
    MOGUL](inst/models/caret_GFS.FR.MOGUL_regression.dcf) - Fuzzy Rules
    via MOGUL from the caret package.
-   [Genetic Lateral Tuning and Rule Selection of Linguistic Fuzzy
    Systems](inst/models/caret_GFS.LT.RS_regression.dcf) - Genetic
    Lateral Tuning and Rule Selection of Linguistic Fuzzy Systems from
    the caret package.
-   [Fuzzy Rules via
    Thrift](inst/models/caret_GFS.THRIFT_regression.dcf) - Fuzzy Rules
    via Thrift from the caret package.
-   [Generalized Linear
    Model](inst/models/caret_glm_classification.dcf) - Generalized
    Linear Model from the caret package.
-   [Generalized Linear Model](inst/models/caret_glm_regression.dcf) -
    Generalized Linear Model from the caret package.
-   [Negative Binomial Generalized Linear
    Model](inst/models/caret_glm.nb_regression.dcf) - Negative Binomial
    Generalized Linear Model from the caret package.
-   [Boosted Generalized Linear
    Model](inst/models/caret_glmboost_classification.dcf) - Boosted
    Generalized Linear Model from the caret package.
-   [Boosted Generalized Linear
    Model](inst/models/caret_glmboost_regression.dcf) - Boosted
    Generalized Linear Model from the caret package.
-   [glmnet](inst/models/caret_glmnet_classification.dcf) - glmnet from
    the caret package.
-   [glmnet](inst/models/caret_glmnet_h2o_classification.dcf) - glmnet
    from the caret package.
-   [glmnet](inst/models/caret_glmnet_h2o_regression.dcf) - glmnet from
    the caret package.
-   [glmnet](inst/models/caret_glmnet_regression.dcf) - glmnet from the
    caret package.
-   [Generalized Linear Model with Stepwise Feature
    Selection](inst/models/caret_glmStepAIC_classification.dcf) -
    Generalized Linear Model with Stepwise Feature Selection from the
    caret package.
-   [Generalized Linear Model with Stepwise Feature
    Selection](inst/models/caret_glmStepAIC_regression.dcf) -
    Generalized Linear Model with Stepwise Feature Selection from the
    caret package.
-   [Generalized Partial Least
    Squares](inst/models/caret_gpls_classification.dcf) - Generalized
    Partial Least Squares from the caret package.
-   [Heteroscedastic Discriminant
    Analysis](inst/models/caret_hda_classification.dcf) -
    Heteroscedastic Discriminant Analysis from the caret package.
-   [High Dimensional Discriminant
    Analysis](inst/models/caret_hdda_classification.dcf) - High
    Dimensional Discriminant Analysis from the caret package.
-   [High-Dimensional Regularized Discriminant
    Analysis](inst/models/caret_hdrda_classification.dcf) -
    High-Dimensional Regularized Discriminant Analysis from the caret
    package.
-   [Hybrid Neural Fuzzy Inference
    System](inst/models/caret_HYFIS_regression.dcf) - Hybrid Neural
    Fuzzy Inference System from the caret package.
-   [Independent Component
    Regression](inst/models/caret_icr_regression.dcf) - Independent
    Component Regression from the caret package.
-   [C4.5-like Trees](inst/models/caret_J48_classification.dcf) -
    C4.5-like Trees from the caret package.
-   [Rule-Based Classifier](inst/models/caret_JRip_classification.dcf) -
    Rule-Based Classifier from the caret package.
-   [Partial Least
    Squares](inst/models/caret_kernelpls_classification.dcf) - Partial
    Least Squares from the caret package.
-   [Partial Least
    Squares](inst/models/caret_kernelpls_regression.dcf) - Partial Least
    Squares from the caret package.
-   [k-Nearest Neighbors](inst/models/caret_kknn_classification.dcf) -
    k-Nearest Neighbors from the caret package.
-   [k-Nearest Neighbors](inst/models/caret_kknn_regression.dcf) -
    k-Nearest Neighbors from the caret package.
-   [k-Nearest Neighbors](inst/models/caret_knn_classification.dcf) -
    k-Nearest Neighbors from the caret package.
-   [k-Nearest Neighbors](inst/models/caret_knn_regression.dcf) -
    k-Nearest Neighbors from the caret package.
-   [Polynomial Kernel Regularized Least
    Squares](inst/models/caret_krlsPoly_regression.dcf) - Polynomial
    Kernel Regularized Least Squares from the caret package.
-   [Radial Basis Function Kernel Regularized Least
    Squares](inst/models/caret_krlsRadial_regression.dcf) - Radial Basis
    Function Kernel Regularized Least Squares from the caret package.
-   [Least Angle Regression](inst/models/caret_lars_regression.dcf) -
    Least Angle Regression from the caret package.
-   [Least Angle Regression](inst/models/caret_lars2_regression.dcf) -
    Least Angle Regression from the caret package.
-   [The lasso](inst/models/caret_lasso_regression.dcf) - The lasso from
    the caret package.
-   [Linear Discriminant
    Analysis](inst/models/caret_lda_classification.dcf) - Linear
    Discriminant Analysis from the caret package.
-   [Linear Discriminant
    Analysis](inst/models/caret_lda2_classification.dcf) - Linear
    Discriminant Analysis from the caret package.
-   [Linear Regression with Backwards
    Selection](inst/models/caret_leapBackward_regression.dcf) - Linear
    Regression with Backwards Selection from the caret package.
-   [Linear Regression with Forward
    Selection](inst/models/caret_leapForward_regression.dcf) - Linear
    Regression with Forward Selection from the caret package.
-   [Linear Regression with Stepwise
    Selection](inst/models/caret_leapSeq_regression.dcf) - Linear
    Regression with Stepwise Selection from the caret package.
-   [Robust Linear Discriminant
    Analysis](inst/models/caret_Linda_classification.dcf) - Robust
    Linear Discriminant Analysis from the caret package.
-   [Linear Regression](inst/models/caret_lm_regression.dcf) - Linear
    Regression from the caret package.
-   [Linear Regression with Stepwise
    Selection](inst/models/caret_lmStepAIC_regression.dcf) - Linear
    Regression with Stepwise Selection from the caret package.
-   [Logistic Model Trees](inst/models/caret_LMT_classification.dcf) -
    Logistic Model Trees from the caret package.
-   [Localized Linear Discriminant
    Analysis](inst/models/caret_loclda_classification.dcf) - Localized
    Linear Discriminant Analysis from the caret package.
-   [Bagged Logic
    Regression](inst/models/caret_logicBag_classification.dcf) - Bagged
    Logic Regression from the caret package.
-   [Bagged Logic
    Regression](inst/models/caret_logicBag_regression.dcf) - Bagged
    Logic Regression from the caret package.
-   [Boosted Logistic
    Regression](inst/models/caret_LogitBoost_classification.dcf) -
    Boosted Logistic Regression from the caret package.
-   [Logic Regression](inst/models/caret_logreg_classification.dcf) -
    Logic Regression from the caret package.
-   [Logic Regression](inst/models/caret_logreg_regression.dcf) - Logic
    Regression from the caret package.
-   [Least Squares Support Vector
    Machine](inst/models/caret_lssvmLinear_classification.dcf) - Least
    Squares Support Vector Machine from the caret package.
-   [Least Squares Support Vector Machine with Polynomial
    Kernel](inst/models/caret_lssvmPoly_classification.dcf) - Least
    Squares Support Vector Machine with Polynomial Kernel from the caret
    package.
-   [Least Squares Support Vector Machine with Radial Basis Function
    Kernel](inst/models/caret_lssvmRadial_classification.dcf) - Least
    Squares Support Vector Machine with Radial Basis Function Kernel
    from the caret package.
-   [Learning Vector
    Quantization](inst/models/caret_lvq_classification.dcf) - Learning
    Vector Quantization from the caret package.
-   [Model Tree](inst/models/caret_M5_regression.dcf) - Model Tree from
    the caret package.
-   [Model Rules](inst/models/caret_M5Rules_regression.dcf) - Model
    Rules from the caret package.
-   [Model Averaged Naive Bayes
    Classifier](inst/models/caret_manb_classification.dcf) - Model
    Averaged Naive Bayes Classifier from the caret package.
-   [Mixture Discriminant
    Analysis](inst/models/caret_mda_classification.dcf) - Mixture
    Discriminant Analysis from the caret package.
-   [Maximum Uncertainty Linear Discriminant
    Analysis](inst/models/caret_Mlda_classification.dcf) - Maximum
    Uncertainty Linear Discriminant Analysis from the caret package.
-   [Multi-Layer Perceptron](inst/models/caret_mlp_classification.dcf) -
    Multi-Layer Perceptron from the caret package.
-   [Multi-Layer Perceptron](inst/models/caret_mlp_regression.dcf) -
    Multi-Layer Perceptron from the caret package.
-   [Multilayer Perceptron Network with Weight
    Decay](inst/models/caret_mlpKerasDecay_classification.dcf) -
    Multilayer Perceptron Network with Weight Decay from the caret
    package.
-   [Multilayer Perceptron Network with Weight
    Decay](inst/models/caret_mlpKerasDecay_regression.dcf) - Multilayer
    Perceptron Network with Weight Decay from the caret package.
-   [Multilayer Perceptron Network with Weight
    Decay](inst/models/caret_mlpKerasDecayCost_classification.dcf) -
    Multilayer Perceptron Network with Weight Decay from the caret
    package.
-   [Multilayer Perceptron Network with
    Dropout](inst/models/caret_mlpKerasDropout_classification.dcf) -
    Multilayer Perceptron Network with Dropout from the caret package.
-   [Multilayer Perceptron Network with
    Dropout](inst/models/caret_mlpKerasDropout_regression.dcf) -
    Multilayer Perceptron Network with Dropout from the caret package.
-   [Multilayer Perceptron Network with
    Dropout](inst/models/caret_mlpKerasDropoutCost_classification.dcf) -
    Multilayer Perceptron Network with Dropout from the caret package.
-   [Multi-Layer Perceptron, with multiple
    layers](inst/models/caret_mlpML_classification.dcf) - Multi-Layer
    Perceptron, with multiple layers from the caret package.
-   [Multi-Layer Perceptron, with multiple
    layers](inst/models/caret_mlpML_regression.dcf) - Multi-Layer
    Perceptron, with multiple layers from the caret package.
-   [Multilayer Perceptron Network by Stochastic Gradient
    Descent](inst/models/caret_mlpSGD_classification.dcf) - Multilayer
    Perceptron Network by Stochastic Gradient Descent from the caret
    package.
-   [Multilayer Perceptron Network by Stochastic Gradient
    Descent](inst/models/caret_mlpSGD_regression.dcf) - Multilayer
    Perceptron Network by Stochastic Gradient Descent from the caret
    package.
-   [Multi-Layer
    Perceptron](inst/models/caret_mlpWeightDecay_classification.dcf) -
    Multi-Layer Perceptron from the caret package.
-   [Multi-Layer
    Perceptron](inst/models/caret_mlpWeightDecay_regression.dcf) -
    Multi-Layer Perceptron from the caret package.
-   [Multi-Layer Perceptron, multiple
    layers](inst/models/caret_mlpWeightDecayML_classification.dcf) -
    Multi-Layer Perceptron, multiple layers from the caret package.
-   [Multi-Layer Perceptron, multiple
    layers](inst/models/caret_mlpWeightDecayML_regression.dcf) -
    Multi-Layer Perceptron, multiple layers from the caret package.
-   [Monotone Multi-Layer Perceptron Neural
    Network](inst/models/caret_monmlp_classification.dcf) - Monotone
    Multi-Layer Perceptron Neural Network from the caret package.
-   [Monotone Multi-Layer Perceptron Neural
    Network](inst/models/caret_monmlp_regression.dcf) - Monotone
    Multi-Layer Perceptron Neural Network from the caret package.
-   [Multi-Step Adaptive
    MCP-Net](inst/models/caret_msaenet_classification.dcf) - Multi-Step
    Adaptive MCP-Net from the caret package.
-   [Multi-Step Adaptive
    MCP-Net](inst/models/caret_msaenet_regression.dcf) - Multi-Step
    Adaptive MCP-Net from the caret package.
-   [Penalized Multinomial
    Regression](inst/models/caret_multinom_classification.dcf) -
    Penalized Multinomial Regression from the caret package.
-   [Neural Network](inst/models/caret_mxnet_classification.dcf) -
    Neural Network from the caret package.
-   [Neural Network](inst/models/caret_mxnet_regression.dcf) - Neural
    Network from the caret package.
-   [Neural Network](inst/models/caret_mxnetAdam_classification.dcf) -
    Neural Network from the caret package.
-   [Neural Network](inst/models/caret_mxnetAdam_regression.dcf) -
    Neural Network from the caret package.
-   [Naive Bayes](inst/models/caret_naive_bayes_classification.dcf) -
    Naive Bayes from the caret package.
-   [Naive Bayes](inst/models/caret_nb_classification.dcf) - Naive Bayes
    from the caret package.
-   [Naive Bayes
    Classifier](inst/models/caret_nbDiscrete_classification.dcf) - Naive
    Bayes Classifier from the caret package.
-   [Semi-Naive Structure Learner
    Wrapper](inst/models/caret_nbSearch_classification.dcf) - Semi-Naive
    Structure Learner Wrapper from the caret package.
-   [Neural Network](inst/models/caret_neuralnet_regression.dcf) -
    Neural Network from the caret package.
-   [Neural Network](inst/models/caret_nnet_classification.dcf) - Neural
    Network from the caret package.
-   [Neural Network](inst/models/caret_nnet_regression.dcf) - Neural
    Network from the caret package.
-   [Non-Negative Least
    Squares](inst/models/caret_nnls_regression.dcf) - Non-Negative Least
    Squares from the caret package.
-   [Tree-Based
    Ensembles](inst/models/caret_nodeHarvest_classification.dcf) -
    Tree-Based Ensembles from the caret package.
-   [Tree-Based
    Ensembles](inst/models/caret_nodeHarvest_regression.dcf) -
    Tree-Based Ensembles from the caret package.
-   [Non-Informative Model](inst/models/caret_null_classification.dcf) -
    Non-Informative Model from the caret package.
-   [Non-Informative Model](inst/models/caret_null_regression.dcf) -
    Non-Informative Model from the caret package.
-   [Single Rule
    Classification](inst/models/caret_OneR_classification.dcf) - Single
    Rule Classification from the caret package.
-   [Penalized Ordinal
    Regression](inst/models/caret_ordinalNet_classification.dcf) -
    Penalized Ordinal Regression from the caret package.
-   [Random Forest](inst/models/caret_ordinalRF_classification.dcf) -
    Random Forest from the caret package.
-   [Oblique Random
    Forest](inst/models/caret_ORFlog_classification.dcf) - Oblique
    Random Forest from the caret package.
-   [Oblique Random
    Forest](inst/models/caret_ORFpls_classification.dcf) - Oblique
    Random Forest from the caret package.
-   [Oblique Random
    Forest](inst/models/caret_ORFridge_classification.dcf) - Oblique
    Random Forest from the caret package.
-   [Oblique Random
    Forest](inst/models/caret_ORFsvm_classification.dcf) - Oblique
    Random Forest from the caret package.
-   [Optimal Weighted Nearest Neighbor
    Classifier](inst/models/caret_ownn_classification.dcf) - Optimal
    Weighted Nearest Neighbor Classifier from the caret package.
-   [Nearest Shrunken
    Centroids](inst/models/caret_pam_classification.dcf) - Nearest
    Shrunken Centroids from the caret package.
-   [Parallel Random
    Forest](inst/models/caret_parRF_classification.dcf) - Parallel
    Random Forest from the caret package.
-   [Parallel Random Forest](inst/models/caret_parRF_regression.dcf) -
    Parallel Random Forest from the caret package.
-   [Rule-Based Classifier](inst/models/caret_PART_classification.dcf) -
    Rule-Based Classifier from the caret package.
-   [partDSA](inst/models/caret_partDSA_classification.dcf) - partDSA
    from the caret package.
-   [partDSA](inst/models/caret_partDSA_regression.dcf) - partDSA from
    the caret package.
-   [Neural Networks with Feature
    Extraction](inst/models/caret_pcaNNet_classification.dcf) - Neural
    Networks with Feature Extraction from the caret package.
-   [Neural Networks with Feature
    Extraction](inst/models/caret_pcaNNet_regression.dcf) - Neural
    Networks with Feature Extraction from the caret package.
-   [Principal Component
    Analysis](inst/models/caret_pcr_regression.dcf) - Principal
    Component Analysis from the caret package.
-   [Penalized Discriminant
    Analysis](inst/models/caret_pda_classification.dcf) - Penalized
    Discriminant Analysis from the caret package.
-   [Penalized Discriminant
    Analysis](inst/models/caret_pda2_classification.dcf) - Penalized
    Discriminant Analysis from the caret package.
-   [Penalized Linear
    Regression](inst/models/caret_penalized_regression.dcf) - Penalized
    Linear Regression from the caret package.
-   [Penalized Linear Discriminant
    Analysis](inst/models/caret_PenalizedLDA_classification.dcf) -
    Penalized Linear Discriminant Analysis from the caret package.
-   [Penalized Logistic
    Regression](inst/models/caret_plr_classification.dcf) - Penalized
    Logistic Regression from the caret package.
-   [Partial Least Squares](inst/models/caret_pls_classification.dcf) -
    Partial Least Squares from the caret package.
-   [Partial Least Squares](inst/models/caret_pls_regression.dcf) -
    Partial Least Squares from the caret package.
-   [Partial Least Squares Generalized Linear
    Models](inst/models/caret_plsRglm_classification.dcf) - Partial
    Least Squares Generalized Linear Models from the caret package.
-   [Partial Least Squares Generalized Linear
    Models](inst/models/caret_plsRglm_regression.dcf) - Partial Least
    Squares Generalized Linear Models from the caret package.
-   [Ordered Logistic or Probit
    Regression](inst/models/caret_polr_classification.dcf) - Ordered
    Logistic or Probit Regression from the caret package.
-   [Projection Pursuit
    Regression](inst/models/caret_ppr_regression.dcf) - Projection
    Pursuit Regression from the caret package.
-   [Prediction Rule
    Ensembles](inst/models/caret_pre_classification.dcf) - Prediction
    Rule Ensembles from the caret package.
-   [Prediction Rule Ensembles](inst/models/caret_pre_regression.dcf) -
    Prediction Rule Ensembles from the caret package.
-   [Patient Rule Induction
    Method](inst/models/caret_PRIM_classification.dcf) - Patient Rule
    Induction Method from the caret package.
-   [Greedy Prototype
    Selection](inst/models/caret_protoclass_classification.dcf) - Greedy
    Prototype Selection from the caret package.
-   [Quadratic Discriminant
    Analysis](inst/models/caret_qda_classification.dcf) - Quadratic
    Discriminant Analysis from the caret package.
-   [Robust Quadratic Discriminant
    Analysis](inst/models/caret_QdaCov_classification.dcf) - Robust
    Quadratic Discriminant Analysis from the caret package.
-   [Quantile Random Forest](inst/models/caret_qrf_regression.dcf) -
    Quantile Random Forest from the caret package.
-   [Quantile Regression Neural
    Network](inst/models/caret_qrnn_regression.dcf) - Quantile
    Regression Neural Network from the caret package.
-   [Ensembles of Generalized Linear
    Models](inst/models/caret_randomGLM_classification.dcf) - Ensembles
    of Generalized Linear Models from the caret package.
-   [Ensembles of Generalized Linear
    Models](inst/models/caret_randomGLM_regression.dcf) - Ensembles of
    Generalized Linear Models from the caret package.
-   [Random Forest](inst/models/caret_ranger_classification.dcf) -
    Random Forest from the caret package.
-   [Random Forest](inst/models/caret_ranger_regression.dcf) - Random
    Forest from the caret package.
-   [Radial Basis Function
    Network](inst/models/caret_rbf_classification.dcf) - Radial Basis
    Function Network from the caret package.
-   [Radial Basis Function
    Network](inst/models/caret_rbf_regression.dcf) - Radial Basis
    Function Network from the caret package.
-   [Radial Basis Function
    Network](inst/models/caret_rbfDDA_classification.dcf) - Radial Basis
    Function Network from the caret package.
-   [Radial Basis Function
    Network](inst/models/caret_rbfDDA_regression.dcf) - Radial Basis
    Function Network from the caret package.
-   [Random Forest](inst/models/caret_Rborist_classification.dcf) -
    Random Forest from the caret package.
-   [Random Forest](inst/models/caret_Rborist_regression.dcf) - Random
    Forest from the caret package.
-   [Regularized Discriminant
    Analysis](inst/models/caret_rda_classification.dcf) - Regularized
    Discriminant Analysis from the caret package.
-   [Regularized Logistic
    Regression](inst/models/caret_regLogistic_classification.dcf) -
    Regularized Logistic Regression from the caret package.
-   [Relaxed Lasso](inst/models/caret_relaxo_regression.dcf) - Relaxed
    Lasso from the caret package.
-   [Random Forest](inst/models/caret_rf_classification.dcf) - Random
    Forest from the caret package.
-   [Random Forest](inst/models/caret_rf_regression.dcf) - Random Forest
    from the caret package.
-   [Random Ferns](inst/models/caret_rFerns_classification.dcf) - Random
    Ferns from the caret package.
-   [Factor-Based Linear Discriminant
    Analysis](inst/models/caret_RFlda_classification.dcf) - Factor-Based
    Linear Discriminant Analysis from the caret package.
-   [Random Forest Rule-Based
    Model](inst/models/caret_rfRules_classification.dcf) - Random Forest
    Rule-Based Model from the caret package.
-   [Random Forest Rule-Based
    Model](inst/models/caret_rfRules_regression.dcf) - Random Forest
    Rule-Based Model from the caret package.
-   [Ridge Regression](inst/models/caret_ridge_regression.dcf) - Ridge
    Regression from the caret package.
-   [Regularized Linear Discriminant
    Analysis](inst/models/caret_rlda_classification.dcf) - Regularized
    Linear Discriminant Analysis from the caret package.
-   [Robust Linear Model](inst/models/caret_rlm_regression.dcf) - Robust
    Linear Model from the caret package.
-   [Robust Mixture Discriminant
    Analysis](inst/models/caret_rmda_classification.dcf) - Robust
    Mixture Discriminant Analysis from the caret package.
-   [ROC-Based Classifier](inst/models/caret_rocc_classification.dcf) -
    ROC-Based Classifier from the caret package.
-   [Rotation
    Forest](inst/models/caret_rotationForest_classification.dcf) -
    Rotation Forest from the caret package.
-   [Rotation
    Forest](inst/models/caret_rotationForestCp_classification.dcf) -
    Rotation Forest from the caret package.
-   [CART](inst/models/caret_rpart_classification.dcf) - CART from the
    caret package.
-   [CART](inst/models/caret_rpart_regression.dcf) - CART from the caret
    package.
-   [CART](inst/models/caret_rpart1SE_classification.dcf) - CART from
    the caret package.
-   [CART](inst/models/caret_rpart1SE_regression.dcf) - CART from the
    caret package.
-   [CART](inst/models/caret_rpart2_classification.dcf) - CART from the
    caret package.
-   [CART](inst/models/caret_rpart2_regression.dcf) - CART from the
    caret package.
-   [Cost-Sensitive
    CART](inst/models/caret_rpartCost_classification.dcf) -
    Cost-Sensitive CART from the caret package.
-   [CART or Ordinal
    Responses](inst/models/caret_rpartScore_classification.dcf) - CART
    or Ordinal Responses from the caret package.
-   [Quantile Regression with LASSO
    penalty](inst/models/caret_rqlasso_regression.dcf) - Quantile
    Regression with LASSO penalty from the caret package.
-   [Non-Convex Penalized Quantile
    Regression](inst/models/caret_rqnc_regression.dcf) - Non-Convex
    Penalized Quantile Regression from the caret package.
-   [Regularized Random
    Forest](inst/models/caret_RRF_classification.dcf) - Regularized
    Random Forest from the caret package.
-   [Regularized Random Forest](inst/models/caret_RRF_regression.dcf) -
    Regularized Random Forest from the caret package.
-   [Regularized Random
    Forest](inst/models/caret_RRFglobal_classification.dcf) -
    Regularized Random Forest from the caret package.
-   [Regularized Random
    Forest](inst/models/caret_RRFglobal_regression.dcf) - Regularized
    Random Forest from the caret package.
-   [Robust Regularized Linear Discriminant
    Analysis](inst/models/caret_rrlda_classification.dcf) - Robust
    Regularized Linear Discriminant Analysis from the caret package.
-   [Robust SIMCA](inst/models/caret_RSimca_classification.dcf) - Robust
    SIMCA from the caret package.
-   [Relevance Vector Machines with Linear
    Kernel](inst/models/caret_rvmLinear_regression.dcf) - Relevance
    Vector Machines with Linear Kernel from the caret package.
-   [Relevance Vector Machines with Polynomial
    Kernel](inst/models/caret_rvmPoly_regression.dcf) - Relevance Vector
    Machines with Polynomial Kernel from the caret package.
-   [Relevance Vector Machines with Radial Basis Function
    Kernel](inst/models/caret_rvmRadial_regression.dcf) - Relevance
    Vector Machines with Radial Basis Function Kernel from the caret
    package.
-   [Subtractive Clustering and Fuzzy c-Means
    Rules](inst/models/caret_SBC_regression.dcf) - Subtractive
    Clustering and Fuzzy c-Means Rules from the caret package.
-   [Shrinkage Discriminant
    Analysis](inst/models/caret_sda_classification.dcf) - Shrinkage
    Discriminant Analysis from the caret package.
-   [Sparse Distance Weighted
    Discrimination](inst/models/caret_sdwd_classification.dcf) - Sparse
    Distance Weighted Discrimination from the caret package.
-   [Partial Least
    Squares](inst/models/caret_simpls_classification.dcf) - Partial
    Least Squares from the caret package.
-   [Partial Least Squares](inst/models/caret_simpls_regression.dcf) -
    Partial Least Squares from the caret package.
-   [Fuzzy Rules Using the Structural Learning Algorithm on Vague
    Environment](inst/models/caret_SLAVE_classification.dcf) - Fuzzy
    Rules Using the Structural Learning Algorithm on Vague Environment
    from the caret package.
-   [Stabilized Linear Discriminant
    Analysis](inst/models/caret_slda_classification.dcf) - Stabilized
    Linear Discriminant Analysis from the caret package.
-   [Sparse Mixture Discriminant
    Analysis](inst/models/caret_smda_classification.dcf) - Sparse
    Mixture Discriminant Analysis from the caret package.
-   [Stabilized Nearest Neighbor
    Classifier](inst/models/caret_snn_classification.dcf) - Stabilized
    Nearest Neighbor Classifier from the caret package.
-   [Sparse Linear Discriminant
    Analysis](inst/models/caret_sparseLDA_classification.dcf) - Sparse
    Linear Discriminant Analysis from the caret package.
-   [Spike and Slab
    Regression](inst/models/caret_spikeslab_regression.dcf) - Spike and
    Slab Regression from the caret package.
-   [Sparse Partial Least
    Squares](inst/models/caret_spls_classification.dcf) - Sparse Partial
    Least Squares from the caret package.
-   [Sparse Partial Least
    Squares](inst/models/caret_spls_regression.dcf) - Sparse Partial
    Least Squares from the caret package.
-   [Linear Discriminant Analysis with Stepwise Feature
    Selection](inst/models/caret_stepLDA_classification.dcf) - Linear
    Discriminant Analysis with Stepwise Feature Selection from the caret
    package.
-   [Quadratic Discriminant Analysis with Stepwise Feature
    Selection](inst/models/caret_stepQDA_classification.dcf) - Quadratic
    Discriminant Analysis with Stepwise Feature Selection from the caret
    package.
-   [Supervised Principal Component
    Analysis](inst/models/caret_superpc_regression.dcf) - Supervised
    Principal Component Analysis from the caret package.
-   [Support Vector Machines with Boundrange String
    Kernel](inst/models/caret_svmBoundrangeString_classification.dcf) -
    Support Vector Machines with Boundrange String Kernel from the caret
    package.
-   [Support Vector Machines with Boundrange String
    Kernel](inst/models/caret_svmBoundrangeString_regression.dcf) -
    Support Vector Machines with Boundrange String Kernel from the caret
    package.
-   [Support Vector Machines with Exponential String
    Kernel](inst/models/caret_svmExpoString_classification.dcf) -
    Support Vector Machines with Exponential String Kernel from the
    caret package.
-   [Support Vector Machines with Exponential String
    Kernel](inst/models/caret_svmExpoString_regression.dcf) - Support
    Vector Machines with Exponential String Kernel from the caret
    package.
-   [Support Vector Machines with Linear
    Kernel](inst/models/caret_svmLinear_classification.dcf) - Support
    Vector Machines with Linear Kernel from the caret package.
-   [Support Vector Machines with Linear
    Kernel](inst/models/caret_svmLinear_regression.dcf) - Support Vector
    Machines with Linear Kernel from the caret package.
-   [Support Vector Machines with Linear
    Kernel](inst/models/caret_svmLinear2_classification.dcf) - Support
    Vector Machines with Linear Kernel from the caret package.
-   [Support Vector Machines with Linear
    Kernel](inst/models/caret_svmLinear2_regression.dcf) - Support
    Vector Machines with Linear Kernel from the caret package.
-   [L2 Regularized Support Vector Machine (dual) with Linear
    Kernel](inst/models/caret_svmLinear3_classification.dcf) - L2
    Regularized Support Vector Machine (dual) with Linear Kernel from
    the caret package.
-   [L2 Regularized Support Vector Machine (dual) with Linear
    Kernel](inst/models/caret_svmLinear3_regression.dcf) - L2
    Regularized Support Vector Machine (dual) with Linear Kernel from
    the caret package.
-   [Linear Support Vector Machines with Class
    Weights](inst/models/caret_svmLinearWeights_classification.dcf) -
    Linear Support Vector Machines with Class Weights from the caret
    package.
-   [L2 Regularized Linear Support Vector Machines with Class
    Weights](inst/models/caret_svmLinearWeights2_classification.dcf) -
    L2 Regularized Linear Support Vector Machines with Class Weights
    from the caret package.
-   [Support Vector Machines with Polynomial
    Kernel](inst/models/caret_svmPoly_classification.dcf) - Support
    Vector Machines with Polynomial Kernel from the caret package.
-   [Support Vector Machines with Polynomial
    Kernel](inst/models/caret_svmPoly_regression.dcf) - Support Vector
    Machines with Polynomial Kernel from the caret package.
-   [Support Vector Machines with Radial Basis Function
    Kernel](inst/models/caret_svmRadial_classification.dcf) - Support
    Vector Machines with Radial Basis Function Kernel from the caret
    package.
-   [Support Vector Machines with Radial Basis Function
    Kernel](inst/models/caret_svmRadial_regression.dcf) - Support Vector
    Machines with Radial Basis Function Kernel from the caret package.
-   [Support Vector Machines with Radial Basis Function
    Kernel](inst/models/caret_svmRadialCost_classification.dcf) -
    Support Vector Machines with Radial Basis Function Kernel from the
    caret package.
-   [Support Vector Machines with Radial Basis Function
    Kernel](inst/models/caret_svmRadialCost_regression.dcf) - Support
    Vector Machines with Radial Basis Function Kernel from the caret
    package.
-   [Support Vector Machines with Radial Basis Function
    Kernel](inst/models/caret_svmRadialSigma_classification.dcf) -
    Support Vector Machines with Radial Basis Function Kernel from the
    caret package.
-   [Support Vector Machines with Radial Basis Function
    Kernel](inst/models/caret_svmRadialSigma_regression.dcf) - Support
    Vector Machines with Radial Basis Function Kernel from the caret
    package.
-   [Support Vector Machines with Class
    Weights](inst/models/caret_svmRadialWeights_classification.dcf) -
    Support Vector Machines with Class Weights from the caret package.
-   [Support Vector Machines with Spectrum String
    Kernel](inst/models/caret_svmSpectrumString_classification.dcf) -
    Support Vector Machines with Spectrum String Kernel from the caret
    package.
-   [Support Vector Machines with Spectrum String
    Kernel](inst/models/caret_svmSpectrumString_regression.dcf) -
    Support Vector Machines with Spectrum String Kernel from the caret
    package.
-   [Tree Augmented Naive Bayes
    Classifier](inst/models/caret_tan_classification.dcf) - Tree
    Augmented Naive Bayes Classifier from the caret package.
-   [Tree Augmented Naive Bayes Classifier Structure Learner
    Wrapper](inst/models/caret_tanSearch_classification.dcf) - Tree
    Augmented Naive Bayes Classifier Structure Learner Wrapper from the
    caret package.
-   [Bagged CART](inst/models/caret_treebag_classification.dcf) - Bagged
    CART from the caret package.
-   [Bagged CART](inst/models/caret_treebag_regression.dcf) - Bagged
    CART from the caret package.
-   [Variational Bayesian Multinomial Probit
    Regression](inst/models/caret_vbmpRadial_classification.dcf) -
    Variational Bayesian Multinomial Probit Regression from the caret
    package.
-   [Adjacent Categories Probability Model for Ordinal
    Data](inst/models/caret_vglmAdjCat_classification.dcf) - Adjacent
    Categories Probability Model for Ordinal Data from the caret
    package.
-   [Continuation Ratio Model for Ordinal
    Data](inst/models/caret_vglmContRatio_classification.dcf) -
    Continuation Ratio Model for Ordinal Data from the caret package.
-   [Cumulative Probability Model for Ordinal
    Data](inst/models/caret_vglmCumulative_classification.dcf) -
    Cumulative Probability Model for Ordinal Data from the caret
    package.
-   [Partial Least
    Squares](inst/models/caret_widekernelpls_classification.dcf) -
    Partial Least Squares from the caret package.
-   [Partial Least
    Squares](inst/models/caret_widekernelpls_regression.dcf) - Partial
    Least Squares from the caret package.
-   [Wang and Mendel Fuzzy Rules](inst/models/caret_WM_regression.dcf) -
    Wang and Mendel Fuzzy Rules from the caret package.
-   [Weighted Subspace Random
    Forest](inst/models/caret_wsrf_classification.dcf) - Weighted
    Subspace Random Forest from the caret package.
-   [eXtreme Gradient
    Boosting](inst/models/caret_xgbDART_classification.dcf) - eXtreme
    Gradient Boosting from the caret package.
-   [eXtreme Gradient
    Boosting](inst/models/caret_xgbDART_regression.dcf) - eXtreme
    Gradient Boosting from the caret package.
-   [eXtreme Gradient
    Boosting](inst/models/caret_xgbLinear_classification.dcf) - eXtreme
    Gradient Boosting from the caret package.
-   [eXtreme Gradient
    Boosting](inst/models/caret_xgbLinear_regression.dcf) - eXtreme
    Gradient Boosting from the caret package.
-   [eXtreme Gradient
    Boosting](inst/models/caret_xgbTree_classification.dcf) - eXtreme
    Gradient Boosting from the caret package.
-   [eXtreme Gradient
    Boosting](inst/models/caret_xgbTree_regression.dcf) - eXtreme
    Gradient Boosting from the caret package.
-   [Self-Organizing Maps](inst/models/caret_xyf_classification.dcf) -
    Self-Organizing Maps from the caret package.
-   [Self-Organizing Maps](inst/models/caret_xyf_regression.dcf) -
    Self-Organizing Maps from the caret package.
-   [Fable Arima Timeseries](inst/models/fable_arima_timeseries.dcf) -
    The R package fable provides a collection of commonly used
    univariate and multivariate time series forecasting models.
-   [fable_tslm_timeseries](inst/models/fable_tslm_timeseries.dcf) - The
    R package fable provides a collection of commonly used univariate
    and multivariate time series forecasting models.
-   [Linear Regression](inst/models/lm.dcf) - Linear regression using
    the stats::lm function.
-   [Logistic Regression](inst/models/logistic.dcf) - Logistic
    regression using the stats::glm function.
-   [Neural network
    logistic-classification](inst/models/neuralnet_logit_classification.dcf) -
    Neural network logistic-classification prediction model using the
    neuralnet R package.
-   [prophet_timeseries](inst/models/prophet_timeseries.dcf) - Prophet
    is a forecasting procedure implemented in R and Python.
-   [Random Forests
    Classification](inst/models/randomForest_classification.dcf) -
    Random forest prediction model usign the randomForest R package.
-   [Random Forest
    Regression](inst/models/randomForest_regression.dcf) - Random forest
    prediction model usign the randomForest R package.
-   [tm_bag_mars_classification](inst/models/tm_bag_mars_classification.dcf) -
    Ensemble of generalized linear models that use artificial features
    for some predictors.
-   [tm_bag_mars_regression](inst/models/tm_bag_mars_regression.dcf) -
    Ensemble of generalized linear models that use artificial features
    for some predictors.
-   [tm_bag_tree_C50_classification](inst/models/tm_bag_tree_C50_classification.dcf) -
    Creates an collection of decision trees forming an ensemble. All
    trees in the ensemble are combined to produce a final prediction.
-   [tm_bag_tree_rpart_classification](inst/models/tm_bag_tree_rpart_classification.dcf) -
    Ensembles of decision trees.
-   [tm_bag_tree_rpart_regression](inst/models/tm_bag_tree_rpart_regression.dcf) -
    Ensembles of decision trees.
-   [tm_bart_classification](inst/models/tm_bart_classification.dcf) -
    Defines a tree ensemble model that uses Bayesian analysis to
    assemble the ensemble. This function can fit classification and
    regression models.
-   [tm_bart_regression](inst/models/tm_bart_regression.dcf) - Defines a
    tree ensemble model that uses Bayesian analysis to assemble the
    ensemble. This function can fit classification and regression
    models.
-   [tm_boost_tree_C50_classification](inst/models/tm_boost_tree_C50_classification.dcf) -
    Defines a model that creates a series of decision trees forming an
    ensemble. Each tree depends on the results of previous trees. All
    trees in the ensemble are combined to produce a final prediction.
-   [tm_boost_tree_xgboost_classification](inst/models/tm_boost_tree_xgboost_classification.dcf) -
    Defines a model that creates a series of decision trees forming an
    ensemble. Each tree depends on the results of previous trees. All
    trees in the ensemble are combined to produce a final prediction.
-   [tm_boost_tree_xgboost_regression](inst/models/tm_boost_tree_xgboost_regression.dcf) -
    Defines a model that creates a series of decision trees forming an
    ensemble. Each tree depends on the results of previous trees. All
    trees in the ensemble are combined to produce a final prediction.
-   [tm_decision_tree_rpart_classification](inst/models/tm_decision_tree_rpart_classification.dcf) -
    Defines a model as a set of if/then statements that creates a
    tree-based structure.
-   [tm_decision_tree_rpart_regression](inst/models/tm_decision_tree_rpart_regression.dcf) -
    Defines a model as a set of if/then statements that creates a
    tree-based structure.
-   [tm_discrim_flexible_classification](inst/models/tm_discrim_flexible_classification.dcf) -
    Defines a model that fits a discriminant analysis model that can use
    nonlinear features created using multivariate adaptive regression
    splines (MARS).
-   [tm_discrim_linear_MASS_classification](inst/models/tm_discrim_linear_MASS_classification.dcf) -
    Defines a model that estimates a multivariate distribution for the
    predictors separately for the data in each class (usually Gaussian
    with a common covariance matrix). Bayes’ theorem is used to compute
    the probability of each class, given the predictor values.
-   [tm_discrim_linear_mda_classification](inst/models/tm_discrim_linear_mda_classification.dcf) -
    Defines a model that estimates a multivariate distribution for the
    predictors separately for the data in each class (usually Gaussian
    with a common covariance matrix). Bayes’ theorem is used to compute
    the probability of each class, given the predictor values.
-   [tm_discrim_linear_sda_classification](inst/models/tm_discrim_linear_sda_classification.dcf) -
    Defines a model that estimates a multivariate distribution for the
    predictors separately for the data in each class (usually Gaussian
    with a common covariance matrix). Bayes’ theorem is used to compute
    the probability of each class, given the predictor values.
-   [tm_discrim_linear_sparsediscrim_classification](inst/models/tm_discrim_linear_sparsediscrim_classification.dcf) -
    Defines a model that estimates a multivariate distribution for the
    predictors separately for the data in each class (usually Gaussian
    with a common covariance matrix). Bayes’ theorem is used to compute
    the probability of each class, given the predictor values.
-   [tm_discrim_regularized_classification](inst/models/tm_discrim_regularized_classification.dcf) -
    Defines a model that estimates a multivariate distribution for the
    predictors separately for the data in each class. The structure of
    the model can be LDA, QDA, or some amalgam of the two. Bayes’
    theorem is used to compute the probability of each class, given the
    predictor values.
-   [tm_exp_smoothing_ets_regression](inst/models/tm_exp_smoothing_regression.dcf) -
    exp_smoothing() is a way to generate a specification of an
    Exponential Smoothing model before fitting and allows the model to
    be created using different packages.
-   [tm_gen_additive_mod_mgcv_classification](inst/models/tm_gen_additive_mod_mgcv_classification.dcf) -
    gen_additive_mod() defines a model that can use smoothed functions
    of numeric predictors in a generalized linear model.
-   [tm_gen_additive_mod_mgcv_regression](inst/models/tm_gen_additive_mod_mgcv_regression.dcf) -
    gen_additive_mod() defines a model that can use smoothed functions
    of numeric predictors in a generalized linear model.
-   [tm_linear_reg_glm_regression](inst/models/tm_linear_reg_glm_regression.dcf) -
    linear_reg() defines a model that can predict numeric values from
    predictors using a linear function.
-   [tm_linear_reg_glmnet_regression](inst/models/tm_linear_reg_glmnet_regression.dcf) -
    linear_reg() defines a model that can predict numeric values from
    predictors using a linear function.
-   [tm_linear_reg_keras_regression](inst/models/tm_linear_reg_keras_regression.dcf) -
    linear_reg() defines a model that can predict numeric values from
    predictors using a linear function.
-   [tm_linear_reg_lm_regression](inst/models/tm_linear_reg_lm_regression.dcf) -
    linear_reg() defines a model that can predict numeric values from
    predictors using a linear function.
-   [tm_linear_reg_stan_regression](inst/models/tm_linear_reg_stan_regression.dcf) -
    linear_reg() defines a model that can predict numeric values from
    predictors using a linear function.
-   [tm_logistic_brulee_classification](inst/models/tm_logistic_brulee_classification.dcf) -
    Defines a generalized linear model for binary outcomes. A linear
    combination of the predictors is used to model the log odds of an
    event.
-   [tm_logistic_glm_classification](inst/models/tm_logistic_glm_classification.dcf) -
    Defines a generalized linear model for binary outcomes. A linear
    combination of the predictors is used to model the log odds of an
    event.
-   [tm_logistic_glmnet_classification](inst/models/tm_logistic_glmnet_classification.dcf) -
    Defines a generalized linear model for binary outcomes. A linear
    combination of the predictors is used to model the log odds of an
    event.
-   [tm_logistic_keras_classification](inst/models/tm_logistic_keras_classification.dcf) -
    Defines a generalized linear model for binary outcomes. A linear
    combination of the predictors is used to model the log odds of an
    event.
-   [tm_logistic_LiblineaR_classification](inst/models/tm_logistic_liblinear_classification.dcf) -
    Defines a generalized linear model for binary outcomes. A linear
    combination of the predictors is used to model the log odds of an
    event.
-   [tm_logistic_stan_classification](inst/models/tm_logistic_stan_classification.dcf) -
    Defines a generalized linear model for binary outcomes. A linear
    combination of the predictors is used to model the log odds of an
    event.
-   [tm_mars](inst/models/tm_mars_classification.dcf) - Defines a
    generalized linear model that uses artificial features for some
    predictors. These features resemble hinge functions and the result
    is a model that is a segmented regression in small dimensions.
-   [tm_mars_regression](inst/models/tm_mars_regression.dcf) - Defines a
    generalized linear model that uses artificial features for some
    predictors. These features resemble hinge functions and the result
    is a model that is a segmented regression in small dimensions.
-   [tm_mlp_brulee_classification](inst/models/tm_mlp_brulee_classification.dcf) -
    Defines a multilayer perceptron model (a.k.a. a single layer,
    feed-forward neural network).
-   [tm_mlp_brulee_regression](inst/models/tm_mlp_brulee_regression.dcf) -
    Defines a multilayer perceptron model (a.k.a. a single layer,
    feed-forward neural network).
-   [tm_mlp_keras_classification](inst/models/tm_mlp_keras_classification.dcf) -
    Defines a multilayer perceptron model (a.k.a. a single layer,
    feed-forward neural network).
-   [tm_mlp_keras_regression](inst/models/tm_mlp_keras_regression.dcf) -
    Defines a multilayer perceptron model (a.k.a. a single layer,
    feed-forward neural network).
-   [tm_mlp_nnet_classification](inst/models/tm_mlp_nnet_classification.dcf) -
    Defines a multilayer perceptron model (a.k.a. a single layer,
    feed-forward neural network).
-   [tm_mlp_nnet_regression](inst/models/tm_mlp_nnet_regression.dcf) -
    Defines a multilayer perceptron model (a.k.a. a single layer,
    feed-forward neural network).
-   [tm_naive_bayes_klaR](inst/models/tm_naive_bayes_klaR_classification.dcf) -
    Model that uses Bayes’ theorem to compute the probability of each
    class, given the predictor values.
-   [tm_naive_bayes_naivebayes](inst/models/tm_naive_bayes_naivebayes_classification.dcf) -
    Model that uses Bayes’ theorem to compute the probability of each
    class, given the predictor values.
-   [tm_nearest_neighbor_classification](inst/models/tm_nearest_neighbor_classification.dcf) -
    Model that uses the K most similar data points from the training set
    to predict new samples.
-   [tm_nearest_neighbor_regression](inst/models/tm_nearest_neighbor_regression.dcf) -
    Model that uses the K most similar data points from the training set
    to predict new samples.
-   [tm_null_model_classification](inst/models/tm_null_model_classification.dcf) -
    Defines a simple, non-informative model.
-   [tm_null_model_regression](inst/models/tm_null_model_regression.dcf) -
    Defines a simple, non-informative model.
-   [tm_pls_classification](inst/models/tm_pls_classification.dcf) -
    Defines a partial least squares model that uses latent variables to
    model the data. It is similar to a supervised version of principal
    component.
-   [tm_pls_regression](inst/models/tm_pls_regression.dcf) - Defines a
    partial least squares model that uses latent variables to model the
    data. It is similar to a supervised version of principal component.
-   [tm_poisson_reg_glm_regression](inst/models/tm_poisson_reg_glm_regression.dcf) -
    Defines a generalized linear model for count data that follow a
    Poisson distribution.
-   [tm_poisson_reg_glmnet_regression](inst/models/tm_poisson_reg_glmnet_regression.dcf) -
    Defines a generalized linear model for count data that follow a
    Poisson distribution.
-   [tm_poisson_reg_stan_regression](inst/models/tm_poisson_reg_stan_regression.dcf) -
    Defines a generalized linear model for count data that follow a
    Poisson distribution.
-   [tm_poisson_reg_zeroinfl_regression](inst/models/tm_poisson_reg_zeroinfl_regression.dcf) -
    Defines a generalized linear model for count data that follow a
    Poisson distribution.
-   [tm_rand_forest_randomForest_classification](inst/models/tm_rand_forest_randomForest_classification.dcf) -
    Defines a model that creates a large number of decision trees, each
    independent of the others. The final prediction uses all predictions
    from the individual trees and combines them.
-   [tm_rand_forest_randomForest_regression](inst/models/tm_rand_forest_randomForest_regression.dcf) -
    Defines a model that creates a large number of decision trees, each
    independent of the others. The final prediction uses all predictions
    from the individual trees and combines them.
-   [tm_rand_forest_ranger_classification](inst/models/tm_rand_forest_ranger_classification.dcf) -
    Defines a model that creates a large number of decision trees, each
    independent of the others. The final prediction uses all predictions
    from the individual trees and combines them.
-   [tm_rand_forest_ranger_regression](inst/models/tm_rand_forest_ranger_regression.dcf) -
    Defines a model that creates a large number of decision trees, each
    independent of the others. The final prediction uses all predictions
    from the individual trees and combines them.
-   [tm_rule_fit_xrf_classification](inst/models/tm_rule_fit_xrf_classification.dcf) -
    Defines a model that derives simple feature rules from a tree
    ensemble and uses them as features in a regularized model.
-   [tm_svm_linear_kernlab_classification](inst/models/tm_svm_linear_kernlab_classification.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    (using a linear class boundary). For regression, the model optimizes
    a robust loss function that is only affected by very large model
    residuals and uses a linear fit.
-   [tm_svm_linear_kernlab_regression](inst/models/tm_svm_linear_kernlab_regression.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    (using a linear class boundary). For regression, the model optimizes
    a robust loss function that is only affected by very large model
    residuals and uses a linear fit.
-   [tm_svm_linear_LiblineaR_classification](inst/models/tm_svm_linear_LiblineaR_classification.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    (using a linear class boundary). For regression, the model optimizes
    a robust loss function that is only affected by very large model
    residuals and uses a linear fit.
-   [tm_svm_poly_kernlab_classification](inst/models/tm_svm_poly_kernlab_classification.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    using a polynomial class boundary. For regression, the model
    optimizes a robust loss function that is only affected by very large
    model residuals and uses polynomial functions of the predictors.
-   [tm_svm_poly_kernlab_regression](inst/models/tm_svm_poly_kernlab_regression.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    using a polynomial class boundary. For regression, the model
    optimizes a robust loss function that is only affected by very large
    model residuals and uses polynomial functions of the predictors.
-   [tm_svm_rbf_kernlab_classification](inst/models/tm_svm_rbf_kernlab_classification.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    using a nonlinear class boundary. For regression, the model
    optimizes a robust loss function that is only affected by very large
    model residuals and uses nonlinear functions of the predictors.
-   [tm_svm_rbf_kernlab_regression](inst/models/tm_svm_rbf_kernlab_regression.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    using a polynomial class boundary. For regression, the model
    optimizes a robust loss function that is only affected by very large
    model residuals and uses polynomial functions of the predictors.
-   [weka_bagging_classification](inst/models/weka_bagging.dcf) -
    Bagging (Breiman, 1996)
-   [weka_decisionstump_classification](inst/models/weka_decisionstump.dcf) -
    Implements decision stumps (trees with a single split only), which
    are frequently used as base learners for meta learners such as
    Boosting.
-   [weka_ibk_classification](inst/models/weka_ibk.dcf) - Provides a
    k-nearest neighbors classifier, see Aha & Kibler (1991).
-   [weka_J48_classification](inst/models/weka_J48_classification.dcf) -
    Class for generating a pruned or unpruned C4.5 decision tree.
-   [weka_jrip_classification](inst/models/weka_jrip.dcf) - Implements a
    propositional rule learner, “Repeated Incremental Pruning to Produce
    Error Reduction” (RIPPER), as proposed by Cohen (1995).
-   [weka_lmt_classification](inst/models/weka_lmt.dcf) - Implements
    “Logistic Model Trees” (Landwehr, 2003; Landwehr et al., 2005).
-   [weka_logistic_classification](inst/models/weka_logistic.dcf) -
    Builds multinomial logistic regression models based on ridge
    estimation (le Cessie and van Houwelingen, 1992).
-   [weka_logitboost_classification](inst/models/weka_logitboost.dcf) -
    Performs boosting via additive logistic regression (Friedman, Hastie
    and Tibshirani, 2000).
-   [weka_oner_classification](inst/models/weka_oner.dcf) - Builds a
    simple 1-R classifier, see Holte (1993).
-   [weka_part_classification](inst/models/weka_part.dcf) - Generates
    PART decision lists using the approach of Frank and Witten (1998).
-   [weka_smo_classification](inst/models/weka_smo.dcf) - Implements
    John C. Platt’s sequential minimal optimization algorithm for
    training a support vector classifier using polynomial or RBF
    kernels.
-   [weka_stacking_classification](inst/models/weka_stacking.dcf) -
    Provides stacking (Wolpert, 1992).
-   [weka_adaboostm1_classification](inst/models/weka.adaboostm1.dcf) -
    Implements the AdaBoost M1 method of Freund and Schapire (1996).

## Creating Datasets

``` r
adult_data <- mldash::new_dataset(
    name = 'adult',
    type = 'classification',
    description = 'Prediction task is to determine whether a person makes over 50K a year.',
    source = 'https://archive.ics.uci.edu/ml/datasets/Adult',
    dir = 'inst/datasets',
    data = function() {
        destfile <- tempfile()
        download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", destfile)
        df <- read.csv(destfile, header = FALSE)
        names(df) <- c('age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                       'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'captial_loss',
                       'hours_per_week', 'native_country', 'greater_than_50k')
        df$greater_than_50k <- df$greater_than_50k == ' >50K'
        return(df)
    },
    model = greater_than_50k ~ .,
    overwrite = TRUE
)
```

Results in creating the following file:

    name: adult
    type: classification
    description: Prediction task is to determine whether a person makes over 50K a year.
    source: https://archive.ics.uci.edu/ml/datasets/Adult
    reference: APA reference for the dataset.
    data: function () 
        {
            destfile <- tempfile()
            download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", 
                destfile)
            df <- read.csv(destfile, header = FALSE)
            names(df) <- c("age", "workclass", "fnlwgt", "education", 
                "education-num", "marital-status", "occupation", "relationship", 
                "race", "sex", "capital-gain", "captial-loss", "hours-per-week", 
                "native-country", "greater_than_50k")
            df$greater_than_50k <- df$greater_than_50k == " >50K"
            return(df)
        }
    model: greater_than_50k ~ .
    note:

## Creating Models

``` r
rf_model <- mldash::new_model(
    name = 'randomForest_classification',
    type = 'classification',
    description = 'Random forest prediction model usign the randomForest R package.',
    train_fun = function(formula, data) {
        randomForest::randomForest(formula = formula, data = data, ntree = 1000)
    },
    predict_fun = function(model, newdata) {
        randomForest:::predict.randomForest(model, newdata = newdata, type = "prob")[,2,drop=TRUE]
    },
    packages = "randomForest",
    overwrite = TRUE
)
```

Results in the following file:

    name: randomForest_classification
    type: classification
    description: Random forest prediction model usign the randomForest R package.
    train: function (formula, data)
        {
            randomForest::randomForest(formula = formula, data = data,
                ntree = 1000)
        }
    predict: function (model, newdata)
        {
            randomForest:::predict.randomForest(model, newdata = newdata, type = "prob")[,2,drop=TRUE]
        }
    packages: randomForest
    note:

Note that for classification models, the `run_models()` function will
ensure that the dependent variable is coded as a factor. If the model
assumes another data type (e.g. TRUE or FALSE) it will need to convert
the variable. Otherwise, the data files (read in by the `read_data()`
function) should ensure all independent variables a properly coded.

## Code of Conduct

Please note that the mldash project is released with a [Contributor Code
of
Conduct](https://contributor-covenant.org/version/2/0/CODE_OF_CONDUCT.html).
By contributing to this project, you agree to abide by its terms.

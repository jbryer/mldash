
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

-   [abalone](https://github.com/jbryer/mldash/blob/master/inst/datasets/abalone.dcf) -
    Predicting the age of abalone from physical measurements.
-   [acute_inflammation](https://github.com/jbryer/mldash/blob/master/inst/datasets/acute_inflammation.dcf) -
    The data was created by a medical expert as a data set to test the
    expert system, which will perform the presumptive diagnosis of two
    diseases of the urinary system.
-   [adult](https://github.com/jbryer/mldash/blob/master/inst/datasets/adult.dcf) -
    Predict whether income exceeds \$50K/yr based on census data. Also
    known as “Census Income” dataset.
-   [air](https://github.com/jbryer/mldash/blob/master/inst/datasets/air.dcf) -
    Contains the responses of a gas multisensor device deployed on the
    field in an Italian city. Hourly responses averages are recorded
    along with gas concentrations references from a certified analyzer.
-   [ames](https://github.com/jbryer/mldash/blob/master/inst/datasets/ames.dcf) -
    Ames Housing Data.
-   [appliances_energy](https://github.com/jbryer/mldash/blob/master/inst/datasets/appliances_energy.dcf) -
    Experimental data used to create regression models of appliances
    energy use in a low energy building.
-   [audit](https://github.com/jbryer/mldash/blob/master/inst/datasets/audit.dcf) -
    Exhaustive one year non-confidential data in the year 2015 to 2016
    of firms is collected from the Auditor Office of India to build a
    predictor for classifying suspicious firms.
-   [bike_sharing_day](https://github.com/jbryer/mldash/blob/master/inst/datasets/bike_sharing_day.dcf) -
    Predication of daily bike rental count based on the environmental
    and seasonal settings
-   [breast_cancer](https://github.com/jbryer/mldash/blob/master/inst/datasets/breast_cancer.dcf) -
    Predict malignant or benign for in breast cancer patients
-   [cervical_cancer](https://github.com/jbryer/mldash/blob/master/inst/datasets/cervical_cancer.dcf) -
    The dataset contains 19 attributes regarding ca cervix behavior risk
    with class label is ca_cervix with 1 and 0 as values which means the
    respondent with and without ca cervix, respectively. predictor for
    classifying suspicious firms.
-   [cmc](https://github.com/jbryer/mldash/blob/master/inst/datasets/cmc.dcf) -
    The problem is to predict the current contraceptive method choice
    (no use, long-term methods, or short-term methods) of a woman based
    on her demographic and socio-economic characteristics.
-   [credit_card_app](https://github.com/jbryer/mldash/blob/master/inst/datasets/credit_card_app.dcf) -
    This data concerns credit card applications; good mix of attributes.
-   [energy](https://github.com/jbryer/mldash/blob/master/inst/datasets/energy.dcf) -
    Experimental data used to create regression models of appliances
    energy use in a low energy building.
-   [hs_graduate_earnings](https://github.com/jbryer/mldash/blob/master/inst/datasets/hs_graduate_earnings.dcf) -
    Predicting high school graduates median earnings based on their
    occupational industries
-   [mars_weather](https://github.com/jbryer/mldash/blob/master/inst/datasets/mars_weather.dcf) -
    Mars Weather
-   [microsoft_stock_price](https://github.com/jbryer/mldash/blob/master/inst/datasets/microsoft_stock_price.dcf) -
    Microsoft stock price from 2001 to the beginning of 2021
-   [mtcars](https://github.com/jbryer/mldash/blob/master/inst/datasets/mtcars.dcf) -
    Motor Trend Car Road Tests
-   [natural_gas_prices](https://github.com/jbryer/mldash/blob/master/inst/datasets/natural_gas_prices.dcf) -
    Time series of major Natural Gas Prices including US Henry Hub. Data
    comes from U.S. Energy Information Administration EIA.
-   [PedalMe](https://github.com/jbryer/mldash/blob/master/inst/datasets/PedalMe.dcf) -
    A dataset about the number of weekly bicycle package deliveries by
    Pedal Me in London during 2020 and 2021.
-   [psych_copay](https://github.com/jbryer/mldash/blob/master/inst/datasets/psych_copay.dcf) -
    Copay modes for established patients in US zip codes
-   [sales](https://github.com/jbryer/mldash/blob/master/inst/datasets/sales.dcf) -
    This is a transnational data set which contains all the transactions
    for a UK-based online retail.
-   [seattle_weather](https://github.com/jbryer/mldash/blob/master/inst/datasets/seattle_weather.dcf) -
    Seattle Weather
-   [sp500](https://github.com/jbryer/mldash/blob/master/inst/datasets/sp500.dcf) -
    Standard and Poor’s (S&P) 500 Index Data including Dividend,
    Earnings and P/E Ratio.
-   [tesla_stock_price](https://github.com/jbryer/mldash/blob/master/inst/datasets/tesla_stock_price.dcf) -
    Standard and Poor’s (S&P) 500 Index Data including Dividend,
    Earnings and P/E Ratio.
-   [titanic](https://github.com/jbryer/mldash/blob/master/inst/datasets/titanic.dcf) -
    The original Titanic dataset, describing the survival status of
    individual passengers on the Titanic.
-   [traffic](https://github.com/jbryer/mldash/blob/master/inst/datasets/traffic.dcf) -
    Hourly Minneapolis-St Paul, MN traffic volume for westbound I-94.
    Includes weather and holiday features from 2012-2018.
-   [wine](https://github.com/jbryer/mldash/blob/master/inst/datasets/wine.dcf) -
    The analysis determined the quantities of 13 constituents found in
    each of the three types of wines.

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
    Trees](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ada_classification.dcf.dcf) -
    Boosted Classification Trees from the caret package.
-   [Bagged
    AdaBoost](https://github.com/jbryer/mldash/blob/master/inst/models/caret_AdaBag_classification.dcf.dcf) -
    Bagged AdaBoost from the caret package.
-   [AdaBoost Classification
    Trees](https://github.com/jbryer/mldash/blob/master/inst/models/caret_adaboost_classification.dcf.dcf) -
    AdaBoost Classification Trees from the caret package.
-   [AdaBoost.M1](https://github.com/jbryer/mldash/blob/master/inst/models/caret_AdaBoost.M1_classification.dcf.dcf) -
    AdaBoost.M1 from the caret package.
-   [Adaptive Mixture Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_amdai_classification.dcf.dcf) -
    Adaptive Mixture Discriminant Analysis from the caret package.
-   [Adaptive-Network-Based Fuzzy Inference
    System](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ANFIS_regression.dcf.dcf) -
    Adaptive-Network-Based Fuzzy Inference System from the caret
    package.
-   [Model Averaged Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_avNNet_classification.dcf.dcf) -
    Model Averaged Neural Network from the caret package.
-   [Model Averaged Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_avNNet_regression.dcf.dcf) -
    Model Averaged Neural Network from the caret package.
-   [Naive Bayes Classifier with Attribute
    Weighting](https://github.com/jbryer/mldash/blob/master/inst/models/caret_awnb_classification.dcf.dcf) -
    Naive Bayes Classifier with Attribute Weighting from the caret
    package.
-   [Tree Augmented Naive Bayes Classifier with Attribute
    Weighting](https://github.com/jbryer/mldash/blob/master/inst/models/caret_awtan_classification.dcf.dcf) -
    Tree Augmented Naive Bayes Classifier with Attribute Weighting from
    the caret package.
-   [Bagged
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bag_classification.dcf.dcf) -
    Bagged Model from the caret package.
-   [Bagged
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bag_regression.dcf.dcf) -
    Bagged Model from the caret package.
-   [Bagged
    MARS](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bagEarth_classification.dcf.dcf) -
    Bagged MARS from the caret package.
-   [Bagged
    MARS](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bagEarth_regression.dcf.dcf) -
    Bagged MARS from the caret package.
-   [Bagged MARS using gCV
    Pruning](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bagEarthGCV_classification.dcf.dcf) -
    Bagged MARS using gCV Pruning from the caret package.
-   [Bagged MARS using gCV
    Pruning](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bagEarthGCV_regression.dcf.dcf) -
    Bagged MARS using gCV Pruning from the caret package.
-   [Bagged Flexible Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bagFDA_classification.dcf.dcf) -
    Bagged Flexible Discriminant Analysis from the caret package.
-   [Bagged FDA using gCV
    Pruning](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bagFDAGCV_classification.dcf.dcf) -
    Bagged FDA using gCV Pruning from the caret package.
-   [Generalized Additive Model using
    Splines](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bam_classification.dcf.dcf) -
    Generalized Additive Model using Splines from the caret package.
-   [Generalized Additive Model using
    Splines](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bam_regression.dcf.dcf) -
    Generalized Additive Model using Splines from the caret package.
-   [Bayesian Additive Regression
    Trees](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bartMachine_classification.dcf.dcf) -
    Bayesian Additive Regression Trees from the caret package.
-   [Bayesian Additive Regression
    Trees](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bartMachine_regression.dcf.dcf) -
    Bayesian Additive Regression Trees from the caret package.
-   [Bayesian Generalized Linear
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bayesglm_classification.dcf.dcf) -
    Bayesian Generalized Linear Model from the caret package.
-   [Bayesian Generalized Linear
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bayesglm_regression.dcf.dcf) -
    Bayesian Generalized Linear Model from the caret package.
-   [Binary Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_binda_classification.dcf.dcf) -
    Binary Discriminant Analysis from the caret package.
-   [Boosted
    Tree](https://github.com/jbryer/mldash/blob/master/inst/models/caret_blackboost_classification.dcf.dcf) -
    Boosted Tree from the caret package.
-   [Boosted
    Tree](https://github.com/jbryer/mldash/blob/master/inst/models/caret_blackboost_regression.dcf.dcf) -
    Boosted Tree from the caret package.
-   [The Bayesian
    lasso](https://github.com/jbryer/mldash/blob/master/inst/models/caret_blasso_regression.dcf.dcf) -
    The Bayesian lasso from the caret package.
-   [Bayesian Ridge Regression (Model
    Averaged)](https://github.com/jbryer/mldash/blob/master/inst/models/caret_blassoAveraged_regression.dcf.dcf) -
    Bayesian Ridge Regression (Model Averaged) from the caret package.
-   [Bayesian Ridge
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bridge_regression.dcf.dcf) -
    Bayesian Ridge Regression from the caret package.
-   [Bayesian Regularized Neural
    Networks](https://github.com/jbryer/mldash/blob/master/inst/models/caret_brnn_regression.dcf.dcf) -
    Bayesian Regularized Neural Networks from the caret package.
-   [Boosted Linear
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_BstLm_classification.dcf.dcf) -
    Boosted Linear Model from the caret package.
-   [Boosted Linear
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_BstLm_regression.dcf.dcf) -
    Boosted Linear Model from the caret package.
-   [Boosted Smoothing
    Spline](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bstSm_classification.dcf.dcf) -
    Boosted Smoothing Spline from the caret package.
-   [Boosted Smoothing
    Spline](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bstSm_regression.dcf.dcf) -
    Boosted Smoothing Spline from the caret package.
-   [Boosted
    Tree](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bstTree_classification.dcf.dcf) -
    Boosted Tree from the caret package.
-   [Boosted
    Tree](https://github.com/jbryer/mldash/blob/master/inst/models/caret_bstTree_regression.dcf.dcf) -
    Boosted Tree from the caret package.
-   [C5.0](https://github.com/jbryer/mldash/blob/master/inst/models/caret_C5.0_classification.dcf.dcf) -
    C5.0 from the caret package.
-   [Cost-Sensitive
    C5.0](https://github.com/jbryer/mldash/blob/master/inst/models/caret_C5.0Cost_classification.dcf.dcf) -
    Cost-Sensitive C5.0 from the caret package.
-   [Single C5.0
    Ruleset](https://github.com/jbryer/mldash/blob/master/inst/models/caret_C5.0Rules_classification.dcf.dcf) -
    Single C5.0 Ruleset from the caret package.
-   [Single C5.0
    Tree](https://github.com/jbryer/mldash/blob/master/inst/models/caret_C5.0Tree_classification.dcf.dcf) -
    Single C5.0 Tree from the caret package.
-   [Conditional Inference Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_cforest_classification.dcf.dcf) -
    Conditional Inference Random Forest from the caret package.
-   [Conditional Inference Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_cforest_regression.dcf.dcf) -
    Conditional Inference Random Forest from the caret package.
-   [CHi-squared Automated Interaction
    Detection](https://github.com/jbryer/mldash/blob/master/inst/models/caret_chaid_classification.dcf.dcf) -
    CHi-squared Automated Interaction Detection from the caret package.
-   [SIMCA](https://github.com/jbryer/mldash/blob/master/inst/models/caret_CSimca_classification.dcf.dcf) -
    SIMCA from the caret package.
-   [Conditional Inference
    Tree](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ctree_classification.dcf.dcf) -
    Conditional Inference Tree from the caret package.
-   [Conditional Inference
    Tree](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ctree_regression.dcf.dcf) -
    Conditional Inference Tree from the caret package.
-   [Conditional Inference
    Tree](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ctree2_classification.dcf.dcf) -
    Conditional Inference Tree from the caret package.
-   [Conditional Inference
    Tree](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ctree2_regression.dcf.dcf) -
    Conditional Inference Tree from the caret package.
-   [Cubist](https://github.com/jbryer/mldash/blob/master/inst/models/caret_cubist_regression.dcf.dcf) -
    Cubist from the caret package.
-   [Diagonal Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_dda_classification.dcf.dcf) -
    Diagonal Discriminant Analysis from the caret package.
-   [DeepBoost](https://github.com/jbryer/mldash/blob/master/inst/models/caret_deepboost_classification.dcf.dcf) -
    DeepBoost from the caret package.
-   [Dynamic Evolving Neural-Fuzzy Inference
    System](https://github.com/jbryer/mldash/blob/master/inst/models/caret_DENFIS_regression.dcf.dcf) -
    Dynamic Evolving Neural-Fuzzy Inference System from the caret
    package.
-   [Stacked AutoEncoder Deep Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_dnn_classification.dcf.dcf) -
    Stacked AutoEncoder Deep Neural Network from the caret package.
-   [Stacked AutoEncoder Deep Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_dnn_regression.dcf.dcf) -
    Stacked AutoEncoder Deep Neural Network from the caret package.
-   [Linear Distance Weighted
    Discrimination](https://github.com/jbryer/mldash/blob/master/inst/models/caret_dwdLinear_classification.dcf.dcf) -
    Linear Distance Weighted Discrimination from the caret package.
-   [Distance Weighted Discrimination with Polynomial
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_dwdPoly_classification.dcf.dcf) -
    Distance Weighted Discrimination with Polynomial Kernel from the
    caret package.
-   [Distance Weighted Discrimination with Radial Basis Function
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_dwdRadial_classification.dcf.dcf) -
    Distance Weighted Discrimination with Radial Basis Function Kernel
    from the caret package.
-   [Multivariate Adaptive Regression
    Spline](https://github.com/jbryer/mldash/blob/master/inst/models/caret_earth_classification.dcf.dcf) -
    Multivariate Adaptive Regression Spline from the caret package.
-   [Multivariate Adaptive Regression
    Spline](https://github.com/jbryer/mldash/blob/master/inst/models/caret_earth_regression.dcf.dcf) -
    Multivariate Adaptive Regression Spline from the caret package.
-   [Extreme Learning
    Machine](https://github.com/jbryer/mldash/blob/master/inst/models/caret_elm_classification.dcf.dcf) -
    Extreme Learning Machine from the caret package.
-   [Extreme Learning
    Machine](https://github.com/jbryer/mldash/blob/master/inst/models/caret_elm_regression.dcf.dcf) -
    Extreme Learning Machine from the caret package.
-   [Elasticnet](https://github.com/jbryer/mldash/blob/master/inst/models/caret_enet_regression.dcf.dcf) -
    Elasticnet from the caret package.
-   [Tree Models from Genetic
    Algorithms](https://github.com/jbryer/mldash/blob/master/inst/models/caret_evtree_classification.dcf.dcf) -
    Tree Models from Genetic Algorithms from the caret package.
-   [Tree Models from Genetic
    Algorithms](https://github.com/jbryer/mldash/blob/master/inst/models/caret_evtree_regression.dcf.dcf) -
    Tree Models from Genetic Algorithms from the caret package.
-   [Random Forest by
    Randomization](https://github.com/jbryer/mldash/blob/master/inst/models/caret_extraTrees_classification.dcf.dcf) -
    Random Forest by Randomization from the caret package.
-   [Random Forest by
    Randomization](https://github.com/jbryer/mldash/blob/master/inst/models/caret_extraTrees_regression.dcf.dcf) -
    Random Forest by Randomization from the caret package.
-   [Flexible Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_fda_classification.dcf.dcf) -
    Flexible Discriminant Analysis from the caret package.
-   [Fuzzy Rules Using Genetic Cooperative-Competitive Learning and
    Pittsburgh](https://github.com/jbryer/mldash/blob/master/inst/models/caret_FH.GBML_classification.dcf.dcf) -
    Fuzzy Rules Using Genetic Cooperative-Competitive Learning and
    Pittsburgh from the caret package.
-   [Fuzzy Inference Rules by Descent
    Method](https://github.com/jbryer/mldash/blob/master/inst/models/caret_FIR.DM_regression.dcf.dcf) -
    Fuzzy Inference Rules by Descent Method from the caret package.
-   [Ridge Regression with Variable
    Selection](https://github.com/jbryer/mldash/blob/master/inst/models/caret_foba_regression.dcf.dcf) -
    Ridge Regression with Variable Selection from the caret package.
-   [Fuzzy Rules Using Chi’s
    Method](https://github.com/jbryer/mldash/blob/master/inst/models/caret_FRBCS.CHI_classification.dcf.dcf) -
    Fuzzy Rules Using Chi’s Method from the caret package.
-   [Fuzzy Rules with Weight
    Factor](https://github.com/jbryer/mldash/blob/master/inst/models/caret_FRBCS.W_classification.dcf.dcf) -
    Fuzzy Rules with Weight Factor from the caret package.
-   [Simplified TSK Fuzzy
    Rules](https://github.com/jbryer/mldash/blob/master/inst/models/caret_FS.HGD_regression.dcf.dcf) -
    Simplified TSK Fuzzy Rules from the caret package.
-   [Generalized Additive Model using
    Splines](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gam_classification.dcf.dcf) -
    Generalized Additive Model using Splines from the caret package.
-   [Generalized Additive Model using
    Splines](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gam_regression.dcf.dcf) -
    Generalized Additive Model using Splines from the caret package.
-   [Boosted Generalized Additive
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gamboost_classification.dcf.dcf) -
    Boosted Generalized Additive Model from the caret package.
-   [Boosted Generalized Additive
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gamboost_regression.dcf.dcf) -
    Boosted Generalized Additive Model from the caret package.
-   [Generalized Additive Model using
    LOESS](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gamLoess_classification.dcf.dcf) -
    Generalized Additive Model using LOESS from the caret package.
-   [Generalized Additive Model using
    LOESS](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gamLoess_regression.dcf.dcf) -
    Generalized Additive Model using LOESS from the caret package.
-   [Generalized Additive Model using
    Splines](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gamSpline_classification.dcf.dcf) -
    Generalized Additive Model using Splines from the caret package.
-   [Generalized Additive Model using
    Splines](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gamSpline_regression.dcf.dcf) -
    Generalized Additive Model using Splines from the caret package.
-   [Gaussian
    Process](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gaussprLinear_classification.dcf.dcf) -
    Gaussian Process from the caret package.
-   [Gaussian
    Process](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gaussprLinear_regression.dcf.dcf) -
    Gaussian Process from the caret package.
-   [Gaussian Process with Polynomial
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gaussprPoly_classification.dcf.dcf) -
    Gaussian Process with Polynomial Kernel from the caret package.
-   [Gaussian Process with Polynomial
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gaussprPoly_regression.dcf.dcf) -
    Gaussian Process with Polynomial Kernel from the caret package.
-   [Gaussian Process with Radial Basis Function
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gaussprRadial_classification.dcf.dcf) -
    Gaussian Process with Radial Basis Function Kernel from the caret
    package.
-   [Gaussian Process with Radial Basis Function
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gaussprRadial_regression.dcf.dcf) -
    Gaussian Process with Radial Basis Function Kernel from the caret
    package.
-   [Stochastic Gradient
    Boosting](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gbm_classification.dcf.dcf) -
    Stochastic Gradient Boosting from the caret package.
-   [Gradient Boosting
    Machines](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gbm_h2o_classification.dcf.dcf) -
    Gradient Boosting Machines from the caret package.
-   [Gradient Boosting
    Machines](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gbm_h2o_regression.dcf.dcf) -
    Gradient Boosting Machines from the caret package.
-   [Stochastic Gradient
    Boosting](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gbm_regression.dcf.dcf) -
    Stochastic Gradient Boosting from the caret package.
-   [Multivariate Adaptive Regression
    Splines](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gcvEarth_classification.dcf.dcf) -
    Multivariate Adaptive Regression Splines from the caret package.
-   [Multivariate Adaptive Regression
    Splines](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gcvEarth_regression.dcf.dcf) -
    Multivariate Adaptive Regression Splines from the caret package.
-   [Fuzzy Rules via
    MOGUL](https://github.com/jbryer/mldash/blob/master/inst/models/caret_GFS.FR.MOGUL_regression.dcf.dcf) -
    Fuzzy Rules via MOGUL from the caret package.
-   [Genetic Lateral Tuning and Rule Selection of Linguistic Fuzzy
    Systems](https://github.com/jbryer/mldash/blob/master/inst/models/caret_GFS.LT.RS_regression.dcf.dcf) -
    Genetic Lateral Tuning and Rule Selection of Linguistic Fuzzy
    Systems from the caret package.
-   [Fuzzy Rules via
    Thrift](https://github.com/jbryer/mldash/blob/master/inst/models/caret_GFS.THRIFT_regression.dcf.dcf) -
    Fuzzy Rules via Thrift from the caret package.
-   [Generalized Linear
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_glm_classification.dcf.dcf) -
    Generalized Linear Model from the caret package.
-   [Generalized Linear
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_glm_regression.dcf.dcf) -
    Generalized Linear Model from the caret package.
-   [Negative Binomial Generalized Linear
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_glm.nb_regression.dcf.dcf) -
    Negative Binomial Generalized Linear Model from the caret package.
-   [Boosted Generalized Linear
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_glmboost_classification.dcf.dcf) -
    Boosted Generalized Linear Model from the caret package.
-   [Boosted Generalized Linear
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_glmboost_regression.dcf.dcf) -
    Boosted Generalized Linear Model from the caret package.
-   [glmnet](https://github.com/jbryer/mldash/blob/master/inst/models/caret_glmnet_classification.dcf.dcf) -
    glmnet from the caret package.
-   [glmnet](https://github.com/jbryer/mldash/blob/master/inst/models/caret_glmnet_h2o_classification.dcf.dcf) -
    glmnet from the caret package.
-   [glmnet](https://github.com/jbryer/mldash/blob/master/inst/models/caret_glmnet_h2o_regression.dcf.dcf) -
    glmnet from the caret package.
-   [glmnet](https://github.com/jbryer/mldash/blob/master/inst/models/caret_glmnet_regression.dcf.dcf) -
    glmnet from the caret package.
-   [Generalized Linear Model with Stepwise Feature
    Selection](https://github.com/jbryer/mldash/blob/master/inst/models/caret_glmStepAIC_classification.dcf.dcf) -
    Generalized Linear Model with Stepwise Feature Selection from the
    caret package.
-   [Generalized Linear Model with Stepwise Feature
    Selection](https://github.com/jbryer/mldash/blob/master/inst/models/caret_glmStepAIC_regression.dcf.dcf) -
    Generalized Linear Model with Stepwise Feature Selection from the
    caret package.
-   [Generalized Partial Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_gpls_classification.dcf.dcf) -
    Generalized Partial Least Squares from the caret package.
-   [Heteroscedastic Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_hda_classification.dcf.dcf) -
    Heteroscedastic Discriminant Analysis from the caret package.
-   [High Dimensional Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_hdda_classification.dcf.dcf) -
    High Dimensional Discriminant Analysis from the caret package.
-   [High-Dimensional Regularized Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_hdrda_classification.dcf.dcf) -
    High-Dimensional Regularized Discriminant Analysis from the caret
    package.
-   [Hybrid Neural Fuzzy Inference
    System](https://github.com/jbryer/mldash/blob/master/inst/models/caret_HYFIS_regression.dcf.dcf) -
    Hybrid Neural Fuzzy Inference System from the caret package.
-   [Independent Component
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_icr_regression.dcf.dcf) -
    Independent Component Regression from the caret package.
-   [C4.5-like
    Trees](https://github.com/jbryer/mldash/blob/master/inst/models/caret_J48_classification.dcf.dcf) -
    C4.5-like Trees from the caret package.
-   [Rule-Based
    Classifier](https://github.com/jbryer/mldash/blob/master/inst/models/caret_JRip_classification.dcf.dcf) -
    Rule-Based Classifier from the caret package.
-   [Partial Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_kernelpls_classification.dcf.dcf) -
    Partial Least Squares from the caret package.
-   [Partial Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_kernelpls_regression.dcf.dcf) -
    Partial Least Squares from the caret package.
-   [k-Nearest
    Neighbors](https://github.com/jbryer/mldash/blob/master/inst/models/caret_kknn_classification.dcf.dcf) -
    k-Nearest Neighbors from the caret package.
-   [k-Nearest
    Neighbors](https://github.com/jbryer/mldash/blob/master/inst/models/caret_kknn_regression.dcf.dcf) -
    k-Nearest Neighbors from the caret package.
-   [k-Nearest
    Neighbors](https://github.com/jbryer/mldash/blob/master/inst/models/caret_knn_classification.dcf.dcf) -
    k-Nearest Neighbors from the caret package.
-   [k-Nearest
    Neighbors](https://github.com/jbryer/mldash/blob/master/inst/models/caret_knn_regression.dcf.dcf) -
    k-Nearest Neighbors from the caret package.
-   [Polynomial Kernel Regularized Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_krlsPoly_regression.dcf.dcf) -
    Polynomial Kernel Regularized Least Squares from the caret package.
-   [Radial Basis Function Kernel Regularized Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_krlsRadial_regression.dcf.dcf) -
    Radial Basis Function Kernel Regularized Least Squares from the
    caret package.
-   [Least Angle
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_lars_regression.dcf.dcf) -
    Least Angle Regression from the caret package.
-   [Least Angle
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_lars2_regression.dcf.dcf) -
    Least Angle Regression from the caret package.
-   [The
    lasso](https://github.com/jbryer/mldash/blob/master/inst/models/caret_lasso_regression.dcf.dcf) -
    The lasso from the caret package.
-   [Linear Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_lda_classification.dcf.dcf) -
    Linear Discriminant Analysis from the caret package.
-   [Linear Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_lda2_classification.dcf.dcf) -
    Linear Discriminant Analysis from the caret package.
-   [Linear Regression with Backwards
    Selection](https://github.com/jbryer/mldash/blob/master/inst/models/caret_leapBackward_regression.dcf.dcf) -
    Linear Regression with Backwards Selection from the caret package.
-   [Linear Regression with Forward
    Selection](https://github.com/jbryer/mldash/blob/master/inst/models/caret_leapForward_regression.dcf.dcf) -
    Linear Regression with Forward Selection from the caret package.
-   [Linear Regression with Stepwise
    Selection](https://github.com/jbryer/mldash/blob/master/inst/models/caret_leapSeq_regression.dcf.dcf) -
    Linear Regression with Stepwise Selection from the caret package.
-   [Robust Linear Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_Linda_classification.dcf.dcf) -
    Robust Linear Discriminant Analysis from the caret package.
-   [Linear
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_lm_regression.dcf.dcf) -
    Linear Regression from the caret package.
-   [Linear Regression with Stepwise
    Selection](https://github.com/jbryer/mldash/blob/master/inst/models/caret_lmStepAIC_regression.dcf.dcf) -
    Linear Regression with Stepwise Selection from the caret package.
-   [Logistic Model
    Trees](https://github.com/jbryer/mldash/blob/master/inst/models/caret_LMT_classification.dcf.dcf) -
    Logistic Model Trees from the caret package.
-   [Localized Linear Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_loclda_classification.dcf.dcf) -
    Localized Linear Discriminant Analysis from the caret package.
-   [Bagged Logic
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_logicBag_classification.dcf.dcf) -
    Bagged Logic Regression from the caret package.
-   [Bagged Logic
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_logicBag_regression.dcf.dcf) -
    Bagged Logic Regression from the caret package.
-   [Boosted Logistic
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_LogitBoost_classification.dcf.dcf) -
    Boosted Logistic Regression from the caret package.
-   [Logic
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_logreg_classification.dcf.dcf) -
    Logic Regression from the caret package.
-   [Logic
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_logreg_regression.dcf.dcf) -
    Logic Regression from the caret package.
-   [Least Squares Support Vector
    Machine](https://github.com/jbryer/mldash/blob/master/inst/models/caret_lssvmLinear_classification.dcf.dcf) -
    Least Squares Support Vector Machine from the caret package.
-   [Least Squares Support Vector Machine with Polynomial
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_lssvmPoly_classification.dcf.dcf) -
    Least Squares Support Vector Machine with Polynomial Kernel from the
    caret package.
-   [Least Squares Support Vector Machine with Radial Basis Function
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_lssvmRadial_classification.dcf.dcf) -
    Least Squares Support Vector Machine with Radial Basis Function
    Kernel from the caret package.
-   [Learning Vector
    Quantization](https://github.com/jbryer/mldash/blob/master/inst/models/caret_lvq_classification.dcf.dcf) -
    Learning Vector Quantization from the caret package.
-   [Model
    Tree](https://github.com/jbryer/mldash/blob/master/inst/models/caret_M5_regression.dcf.dcf) -
    Model Tree from the caret package.
-   [Model
    Rules](https://github.com/jbryer/mldash/blob/master/inst/models/caret_M5Rules_regression.dcf.dcf) -
    Model Rules from the caret package.
-   [Model Averaged Naive Bayes
    Classifier](https://github.com/jbryer/mldash/blob/master/inst/models/caret_manb_classification.dcf.dcf) -
    Model Averaged Naive Bayes Classifier from the caret package.
-   [Mixture Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mda_classification.dcf.dcf) -
    Mixture Discriminant Analysis from the caret package.
-   [Maximum Uncertainty Linear Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_Mlda_classification.dcf.dcf) -
    Maximum Uncertainty Linear Discriminant Analysis from the caret
    package.
-   [Multi-Layer
    Perceptron](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlp_classification.dcf.dcf) -
    Multi-Layer Perceptron from the caret package.
-   [Multi-Layer
    Perceptron](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlp_regression.dcf.dcf) -
    Multi-Layer Perceptron from the caret package.
-   [Multilayer Perceptron Network with Weight
    Decay](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpKerasDecay_classification.dcf.dcf) -
    Multilayer Perceptron Network with Weight Decay from the caret
    package.
-   [Multilayer Perceptron Network with Weight
    Decay](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpKerasDecay_regression.dcf.dcf) -
    Multilayer Perceptron Network with Weight Decay from the caret
    package.
-   [Multilayer Perceptron Network with Weight
    Decay](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpKerasDecayCost_classification.dcf.dcf) -
    Multilayer Perceptron Network with Weight Decay from the caret
    package.
-   [Multilayer Perceptron Network with
    Dropout](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpKerasDropout_classification.dcf.dcf) -
    Multilayer Perceptron Network with Dropout from the caret package.
-   [Multilayer Perceptron Network with
    Dropout](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpKerasDropout_regression.dcf.dcf) -
    Multilayer Perceptron Network with Dropout from the caret package.
-   [Multilayer Perceptron Network with
    Dropout](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpKerasDropoutCost_classification.dcf.dcf) -
    Multilayer Perceptron Network with Dropout from the caret package.
-   [Multi-Layer Perceptron, with multiple
    layers](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpML_classification.dcf.dcf) -
    Multi-Layer Perceptron, with multiple layers from the caret package.
-   [Multi-Layer Perceptron, with multiple
    layers](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpML_regression.dcf.dcf) -
    Multi-Layer Perceptron, with multiple layers from the caret package.
-   [Multilayer Perceptron Network by Stochastic Gradient
    Descent](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpSGD_classification.dcf.dcf) -
    Multilayer Perceptron Network by Stochastic Gradient Descent from
    the caret package.
-   [Multilayer Perceptron Network by Stochastic Gradient
    Descent](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpSGD_regression.dcf.dcf) -
    Multilayer Perceptron Network by Stochastic Gradient Descent from
    the caret package.
-   [Multi-Layer
    Perceptron](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpWeightDecay_classification.dcf.dcf) -
    Multi-Layer Perceptron from the caret package.
-   [Multi-Layer
    Perceptron](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpWeightDecay_regression.dcf.dcf) -
    Multi-Layer Perceptron from the caret package.
-   [Multi-Layer Perceptron, multiple
    layers](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpWeightDecayML_classification.dcf.dcf) -
    Multi-Layer Perceptron, multiple layers from the caret package.
-   [Multi-Layer Perceptron, multiple
    layers](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mlpWeightDecayML_regression.dcf.dcf) -
    Multi-Layer Perceptron, multiple layers from the caret package.
-   [Monotone Multi-Layer Perceptron Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_monmlp_classification.dcf.dcf) -
    Monotone Multi-Layer Perceptron Neural Network from the caret
    package.
-   [Monotone Multi-Layer Perceptron Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_monmlp_regression.dcf.dcf) -
    Monotone Multi-Layer Perceptron Neural Network from the caret
    package.
-   [Multi-Step Adaptive
    MCP-Net](https://github.com/jbryer/mldash/blob/master/inst/models/caret_msaenet_classification.dcf.dcf) -
    Multi-Step Adaptive MCP-Net from the caret package.
-   [Multi-Step Adaptive
    MCP-Net](https://github.com/jbryer/mldash/blob/master/inst/models/caret_msaenet_regression.dcf.dcf) -
    Multi-Step Adaptive MCP-Net from the caret package.
-   [Penalized Multinomial
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_multinom_classification.dcf.dcf) -
    Penalized Multinomial Regression from the caret package.
-   [Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mxnet_classification.dcf.dcf) -
    Neural Network from the caret package.
-   [Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mxnet_regression.dcf.dcf) -
    Neural Network from the caret package.
-   [Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mxnetAdam_classification.dcf.dcf) -
    Neural Network from the caret package.
-   [Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_mxnetAdam_regression.dcf.dcf) -
    Neural Network from the caret package.
-   [Naive
    Bayes](https://github.com/jbryer/mldash/blob/master/inst/models/caret_naive_bayes_classification.dcf.dcf) -
    Naive Bayes from the caret package.
-   [Naive
    Bayes](https://github.com/jbryer/mldash/blob/master/inst/models/caret_nb_classification.dcf.dcf) -
    Naive Bayes from the caret package.
-   [Naive Bayes
    Classifier](https://github.com/jbryer/mldash/blob/master/inst/models/caret_nbDiscrete_classification.dcf.dcf) -
    Naive Bayes Classifier from the caret package.
-   [Semi-Naive Structure Learner
    Wrapper](https://github.com/jbryer/mldash/blob/master/inst/models/caret_nbSearch_classification.dcf.dcf) -
    Semi-Naive Structure Learner Wrapper from the caret package.
-   [Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_neuralnet_regression.dcf.dcf) -
    Neural Network from the caret package.
-   [Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_nnet_classification.dcf.dcf) -
    Neural Network from the caret package.
-   [Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_nnet_regression.dcf.dcf) -
    Neural Network from the caret package.
-   [Non-Negative Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_nnls_regression.dcf.dcf) -
    Non-Negative Least Squares from the caret package.
-   [Tree-Based
    Ensembles](https://github.com/jbryer/mldash/blob/master/inst/models/caret_nodeHarvest_classification.dcf.dcf) -
    Tree-Based Ensembles from the caret package.
-   [Tree-Based
    Ensembles](https://github.com/jbryer/mldash/blob/master/inst/models/caret_nodeHarvest_regression.dcf.dcf) -
    Tree-Based Ensembles from the caret package.
-   [Non-Informative
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_null_classification.dcf.dcf) -
    Non-Informative Model from the caret package.
-   [Non-Informative
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_null_regression.dcf.dcf) -
    Non-Informative Model from the caret package.
-   [Single Rule
    Classification](https://github.com/jbryer/mldash/blob/master/inst/models/caret_OneR_classification.dcf.dcf) -
    Single Rule Classification from the caret package.
-   [Penalized Ordinal
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ordinalNet_classification.dcf.dcf) -
    Penalized Ordinal Regression from the caret package.
-   [Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ordinalRF_classification.dcf.dcf) -
    Random Forest from the caret package.
-   [Oblique Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ORFlog_classification.dcf.dcf) -
    Oblique Random Forest from the caret package.
-   [Oblique Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ORFpls_classification.dcf.dcf) -
    Oblique Random Forest from the caret package.
-   [Oblique Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ORFridge_classification.dcf.dcf) -
    Oblique Random Forest from the caret package.
-   [Oblique Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ORFsvm_classification.dcf.dcf) -
    Oblique Random Forest from the caret package.
-   [Optimal Weighted Nearest Neighbor
    Classifier](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ownn_classification.dcf.dcf) -
    Optimal Weighted Nearest Neighbor Classifier from the caret package.
-   [Nearest Shrunken
    Centroids](https://github.com/jbryer/mldash/blob/master/inst/models/caret_pam_classification.dcf.dcf) -
    Nearest Shrunken Centroids from the caret package.
-   [Parallel Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_parRF_classification.dcf.dcf) -
    Parallel Random Forest from the caret package.
-   [Parallel Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_parRF_regression.dcf.dcf) -
    Parallel Random Forest from the caret package.
-   [Rule-Based
    Classifier](https://github.com/jbryer/mldash/blob/master/inst/models/caret_PART_classification.dcf.dcf) -
    Rule-Based Classifier from the caret package.
-   [partDSA](https://github.com/jbryer/mldash/blob/master/inst/models/caret_partDSA_classification.dcf.dcf) -
    partDSA from the caret package.
-   [partDSA](https://github.com/jbryer/mldash/blob/master/inst/models/caret_partDSA_regression.dcf.dcf) -
    partDSA from the caret package.
-   [Neural Networks with Feature
    Extraction](https://github.com/jbryer/mldash/blob/master/inst/models/caret_pcaNNet_classification.dcf.dcf) -
    Neural Networks with Feature Extraction from the caret package.
-   [Neural Networks with Feature
    Extraction](https://github.com/jbryer/mldash/blob/master/inst/models/caret_pcaNNet_regression.dcf.dcf) -
    Neural Networks with Feature Extraction from the caret package.
-   [Principal Component
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_pcr_regression.dcf.dcf) -
    Principal Component Analysis from the caret package.
-   [Penalized Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_pda_classification.dcf.dcf) -
    Penalized Discriminant Analysis from the caret package.
-   [Penalized Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_pda2_classification.dcf.dcf) -
    Penalized Discriminant Analysis from the caret package.
-   [Penalized Linear
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_penalized_regression.dcf.dcf) -
    Penalized Linear Regression from the caret package.
-   [Penalized Linear Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_PenalizedLDA_classification.dcf.dcf) -
    Penalized Linear Discriminant Analysis from the caret package.
-   [Penalized Logistic
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_plr_classification.dcf.dcf) -
    Penalized Logistic Regression from the caret package.
-   [Partial Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_pls_classification.dcf.dcf) -
    Partial Least Squares from the caret package.
-   [Partial Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_pls_regression.dcf.dcf) -
    Partial Least Squares from the caret package.
-   [Partial Least Squares Generalized Linear
    Models](https://github.com/jbryer/mldash/blob/master/inst/models/caret_plsRglm_classification.dcf.dcf) -
    Partial Least Squares Generalized Linear Models from the caret
    package.
-   [Partial Least Squares Generalized Linear
    Models](https://github.com/jbryer/mldash/blob/master/inst/models/caret_plsRglm_regression.dcf.dcf) -
    Partial Least Squares Generalized Linear Models from the caret
    package.
-   [Ordered Logistic or Probit
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_polr_classification.dcf.dcf) -
    Ordered Logistic or Probit Regression from the caret package.
-   [Projection Pursuit
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ppr_regression.dcf.dcf) -
    Projection Pursuit Regression from the caret package.
-   [Prediction Rule
    Ensembles](https://github.com/jbryer/mldash/blob/master/inst/models/caret_pre_classification.dcf.dcf) -
    Prediction Rule Ensembles from the caret package.
-   [Prediction Rule
    Ensembles](https://github.com/jbryer/mldash/blob/master/inst/models/caret_pre_regression.dcf.dcf) -
    Prediction Rule Ensembles from the caret package.
-   [Patient Rule Induction
    Method](https://github.com/jbryer/mldash/blob/master/inst/models/caret_PRIM_classification.dcf.dcf) -
    Patient Rule Induction Method from the caret package.
-   [Greedy Prototype
    Selection](https://github.com/jbryer/mldash/blob/master/inst/models/caret_protoclass_classification.dcf.dcf) -
    Greedy Prototype Selection from the caret package.
-   [Quadratic Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_qda_classification.dcf.dcf) -
    Quadratic Discriminant Analysis from the caret package.
-   [Robust Quadratic Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_QdaCov_classification.dcf.dcf) -
    Robust Quadratic Discriminant Analysis from the caret package.
-   [Quantile Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_qrf_regression.dcf.dcf) -
    Quantile Random Forest from the caret package.
-   [Quantile Regression Neural
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_qrnn_regression.dcf.dcf) -
    Quantile Regression Neural Network from the caret package.
-   [Ensembles of Generalized Linear
    Models](https://github.com/jbryer/mldash/blob/master/inst/models/caret_randomGLM_classification.dcf.dcf) -
    Ensembles of Generalized Linear Models from the caret package.
-   [Ensembles of Generalized Linear
    Models](https://github.com/jbryer/mldash/blob/master/inst/models/caret_randomGLM_regression.dcf.dcf) -
    Ensembles of Generalized Linear Models from the caret package.
-   [Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ranger_classification.dcf.dcf) -
    Random Forest from the caret package.
-   [Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ranger_regression.dcf.dcf) -
    Random Forest from the caret package.
-   [Radial Basis Function
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rbf_classification.dcf.dcf) -
    Radial Basis Function Network from the caret package.
-   [Radial Basis Function
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rbf_regression.dcf.dcf) -
    Radial Basis Function Network from the caret package.
-   [Radial Basis Function
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rbfDDA_classification.dcf.dcf) -
    Radial Basis Function Network from the caret package.
-   [Radial Basis Function
    Network](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rbfDDA_regression.dcf.dcf) -
    Radial Basis Function Network from the caret package.
-   [Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_Rborist_classification.dcf.dcf) -
    Random Forest from the caret package.
-   [Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_Rborist_regression.dcf.dcf) -
    Random Forest from the caret package.
-   [Regularized Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rda_classification.dcf.dcf) -
    Regularized Discriminant Analysis from the caret package.
-   [Regularized Logistic
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_regLogistic_classification.dcf.dcf) -
    Regularized Logistic Regression from the caret package.
-   [Relaxed
    Lasso](https://github.com/jbryer/mldash/blob/master/inst/models/caret_relaxo_regression.dcf.dcf) -
    Relaxed Lasso from the caret package.
-   [Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rf_classification.dcf.dcf) -
    Random Forest from the caret package.
-   [Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rf_regression.dcf.dcf) -
    Random Forest from the caret package.
-   [Random
    Ferns](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rFerns_classification.dcf.dcf) -
    Random Ferns from the caret package.
-   [Factor-Based Linear Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_RFlda_classification.dcf.dcf) -
    Factor-Based Linear Discriminant Analysis from the caret package.
-   [Random Forest Rule-Based
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rfRules_classification.dcf.dcf) -
    Random Forest Rule-Based Model from the caret package.
-   [Random Forest Rule-Based
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rfRules_regression.dcf.dcf) -
    Random Forest Rule-Based Model from the caret package.
-   [Ridge
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_ridge_regression.dcf.dcf) -
    Ridge Regression from the caret package.
-   [Regularized Linear Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rlda_classification.dcf.dcf) -
    Regularized Linear Discriminant Analysis from the caret package.
-   [Robust Linear
    Model](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rlm_regression.dcf.dcf) -
    Robust Linear Model from the caret package.
-   [Robust Mixture Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rmda_classification.dcf.dcf) -
    Robust Mixture Discriminant Analysis from the caret package.
-   [ROC-Based
    Classifier](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rocc_classification.dcf.dcf) -
    ROC-Based Classifier from the caret package.
-   [Rotation
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rotationForest_classification.dcf.dcf) -
    Rotation Forest from the caret package.
-   [Rotation
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rotationForestCp_classification.dcf.dcf) -
    Rotation Forest from the caret package.
-   [CART](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rpart_classification.dcf.dcf) -
    CART from the caret package.
-   [CART](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rpart_regression.dcf.dcf) -
    CART from the caret package.
-   [CART](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rpart1SE_classification.dcf.dcf) -
    CART from the caret package.
-   [CART](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rpart1SE_regression.dcf.dcf) -
    CART from the caret package.
-   [CART](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rpart2_classification.dcf.dcf) -
    CART from the caret package.
-   [CART](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rpart2_regression.dcf.dcf) -
    CART from the caret package.
-   [Cost-Sensitive
    CART](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rpartCost_classification.dcf.dcf) -
    Cost-Sensitive CART from the caret package.
-   [CART or Ordinal
    Responses](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rpartScore_classification.dcf.dcf) -
    CART or Ordinal Responses from the caret package.
-   [Quantile Regression with LASSO
    penalty](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rqlasso_regression.dcf.dcf) -
    Quantile Regression with LASSO penalty from the caret package.
-   [Non-Convex Penalized Quantile
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rqnc_regression.dcf.dcf) -
    Non-Convex Penalized Quantile Regression from the caret package.
-   [Regularized Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_RRF_classification.dcf.dcf) -
    Regularized Random Forest from the caret package.
-   [Regularized Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_RRF_regression.dcf.dcf) -
    Regularized Random Forest from the caret package.
-   [Regularized Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_RRFglobal_classification.dcf.dcf) -
    Regularized Random Forest from the caret package.
-   [Regularized Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_RRFglobal_regression.dcf.dcf) -
    Regularized Random Forest from the caret package.
-   [Robust Regularized Linear Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rrlda_classification.dcf.dcf) -
    Robust Regularized Linear Discriminant Analysis from the caret
    package.
-   [Robust
    SIMCA](https://github.com/jbryer/mldash/blob/master/inst/models/caret_RSimca_classification.dcf.dcf) -
    Robust SIMCA from the caret package.
-   [Relevance Vector Machines with Linear
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rvmLinear_regression.dcf.dcf) -
    Relevance Vector Machines with Linear Kernel from the caret package.
-   [Relevance Vector Machines with Polynomial
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rvmPoly_regression.dcf.dcf) -
    Relevance Vector Machines with Polynomial Kernel from the caret
    package.
-   [Relevance Vector Machines with Radial Basis Function
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_rvmRadial_regression.dcf.dcf) -
    Relevance Vector Machines with Radial Basis Function Kernel from the
    caret package.
-   [Subtractive Clustering and Fuzzy c-Means
    Rules](https://github.com/jbryer/mldash/blob/master/inst/models/caret_SBC_regression.dcf.dcf) -
    Subtractive Clustering and Fuzzy c-Means Rules from the caret
    package.
-   [Shrinkage Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_sda_classification.dcf.dcf) -
    Shrinkage Discriminant Analysis from the caret package.
-   [Sparse Distance Weighted
    Discrimination](https://github.com/jbryer/mldash/blob/master/inst/models/caret_sdwd_classification.dcf.dcf) -
    Sparse Distance Weighted Discrimination from the caret package.
-   [Partial Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_simpls_classification.dcf.dcf) -
    Partial Least Squares from the caret package.
-   [Partial Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_simpls_regression.dcf.dcf) -
    Partial Least Squares from the caret package.
-   [Fuzzy Rules Using the Structural Learning Algorithm on Vague
    Environment](https://github.com/jbryer/mldash/blob/master/inst/models/caret_SLAVE_classification.dcf.dcf) -
    Fuzzy Rules Using the Structural Learning Algorithm on Vague
    Environment from the caret package.
-   [Stabilized Linear Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_slda_classification.dcf.dcf) -
    Stabilized Linear Discriminant Analysis from the caret package.
-   [Sparse Mixture Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_smda_classification.dcf.dcf) -
    Sparse Mixture Discriminant Analysis from the caret package.
-   [Stabilized Nearest Neighbor
    Classifier](https://github.com/jbryer/mldash/blob/master/inst/models/caret_snn_classification.dcf.dcf) -
    Stabilized Nearest Neighbor Classifier from the caret package.
-   [Sparse Linear Discriminant
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_sparseLDA_classification.dcf.dcf) -
    Sparse Linear Discriminant Analysis from the caret package.
-   [Spike and Slab
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_spikeslab_regression.dcf.dcf) -
    Spike and Slab Regression from the caret package.
-   [Sparse Partial Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_spls_classification.dcf.dcf) -
    Sparse Partial Least Squares from the caret package.
-   [Sparse Partial Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_spls_regression.dcf.dcf) -
    Sparse Partial Least Squares from the caret package.
-   [Linear Discriminant Analysis with Stepwise Feature
    Selection](https://github.com/jbryer/mldash/blob/master/inst/models/caret_stepLDA_classification.dcf.dcf) -
    Linear Discriminant Analysis with Stepwise Feature Selection from
    the caret package.
-   [Quadratic Discriminant Analysis with Stepwise Feature
    Selection](https://github.com/jbryer/mldash/blob/master/inst/models/caret_stepQDA_classification.dcf.dcf) -
    Quadratic Discriminant Analysis with Stepwise Feature Selection from
    the caret package.
-   [Supervised Principal Component
    Analysis](https://github.com/jbryer/mldash/blob/master/inst/models/caret_superpc_regression.dcf.dcf) -
    Supervised Principal Component Analysis from the caret package.
-   [Support Vector Machines with Boundrange String
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmBoundrangeString_classification.dcf.dcf) -
    Support Vector Machines with Boundrange String Kernel from the caret
    package.
-   [Support Vector Machines with Boundrange String
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmBoundrangeString_regression.dcf.dcf) -
    Support Vector Machines with Boundrange String Kernel from the caret
    package.
-   [Support Vector Machines with Exponential String
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmExpoString_classification.dcf.dcf) -
    Support Vector Machines with Exponential String Kernel from the
    caret package.
-   [Support Vector Machines with Exponential String
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmExpoString_regression.dcf.dcf) -
    Support Vector Machines with Exponential String Kernel from the
    caret package.
-   [Support Vector Machines with Linear
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmLinear_classification.dcf.dcf) -
    Support Vector Machines with Linear Kernel from the caret package.
-   [Support Vector Machines with Linear
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmLinear_regression.dcf.dcf) -
    Support Vector Machines with Linear Kernel from the caret package.
-   [Support Vector Machines with Linear
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmLinear2_classification.dcf.dcf) -
    Support Vector Machines with Linear Kernel from the caret package.
-   [Support Vector Machines with Linear
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmLinear2_regression.dcf.dcf) -
    Support Vector Machines with Linear Kernel from the caret package.
-   [L2 Regularized Support Vector Machine (dual) with Linear
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmLinear3_classification.dcf.dcf) -
    L2 Regularized Support Vector Machine (dual) with Linear Kernel from
    the caret package.
-   [L2 Regularized Support Vector Machine (dual) with Linear
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmLinear3_regression.dcf.dcf) -
    L2 Regularized Support Vector Machine (dual) with Linear Kernel from
    the caret package.
-   [Linear Support Vector Machines with Class
    Weights](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmLinearWeights_classification.dcf.dcf) -
    Linear Support Vector Machines with Class Weights from the caret
    package.
-   [L2 Regularized Linear Support Vector Machines with Class
    Weights](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmLinearWeights2_classification.dcf.dcf) -
    L2 Regularized Linear Support Vector Machines with Class Weights
    from the caret package.
-   [Support Vector Machines with Polynomial
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmPoly_classification.dcf.dcf) -
    Support Vector Machines with Polynomial Kernel from the caret
    package.
-   [Support Vector Machines with Polynomial
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmPoly_regression.dcf.dcf) -
    Support Vector Machines with Polynomial Kernel from the caret
    package.
-   [Support Vector Machines with Radial Basis Function
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmRadial_classification.dcf.dcf) -
    Support Vector Machines with Radial Basis Function Kernel from the
    caret package.
-   [Support Vector Machines with Radial Basis Function
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmRadial_regression.dcf.dcf) -
    Support Vector Machines with Radial Basis Function Kernel from the
    caret package.
-   [Support Vector Machines with Radial Basis Function
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmRadialCost_classification.dcf.dcf) -
    Support Vector Machines with Radial Basis Function Kernel from the
    caret package.
-   [Support Vector Machines with Radial Basis Function
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmRadialCost_regression.dcf.dcf) -
    Support Vector Machines with Radial Basis Function Kernel from the
    caret package.
-   [Support Vector Machines with Radial Basis Function
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmRadialSigma_classification.dcf.dcf) -
    Support Vector Machines with Radial Basis Function Kernel from the
    caret package.
-   [Support Vector Machines with Radial Basis Function
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmRadialSigma_regression.dcf.dcf) -
    Support Vector Machines with Radial Basis Function Kernel from the
    caret package.
-   [Support Vector Machines with Class
    Weights](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmRadialWeights_classification.dcf.dcf) -
    Support Vector Machines with Class Weights from the caret package.
-   [Support Vector Machines with Spectrum String
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmSpectrumString_classification.dcf.dcf) -
    Support Vector Machines with Spectrum String Kernel from the caret
    package.
-   [Support Vector Machines with Spectrum String
    Kernel](https://github.com/jbryer/mldash/blob/master/inst/models/caret_svmSpectrumString_regression.dcf.dcf) -
    Support Vector Machines with Spectrum String Kernel from the caret
    package.
-   [Tree Augmented Naive Bayes
    Classifier](https://github.com/jbryer/mldash/blob/master/inst/models/caret_tan_classification.dcf.dcf) -
    Tree Augmented Naive Bayes Classifier from the caret package.
-   [Tree Augmented Naive Bayes Classifier Structure Learner
    Wrapper](https://github.com/jbryer/mldash/blob/master/inst/models/caret_tanSearch_classification.dcf.dcf) -
    Tree Augmented Naive Bayes Classifier Structure Learner Wrapper from
    the caret package.
-   [Bagged
    CART](https://github.com/jbryer/mldash/blob/master/inst/models/caret_treebag_classification.dcf.dcf) -
    Bagged CART from the caret package.
-   [Bagged
    CART](https://github.com/jbryer/mldash/blob/master/inst/models/caret_treebag_regression.dcf.dcf) -
    Bagged CART from the caret package.
-   [Variational Bayesian Multinomial Probit
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/caret_vbmpRadial_classification.dcf.dcf) -
    Variational Bayesian Multinomial Probit Regression from the caret
    package.
-   [Adjacent Categories Probability Model for Ordinal
    Data](https://github.com/jbryer/mldash/blob/master/inst/models/caret_vglmAdjCat_classification.dcf.dcf) -
    Adjacent Categories Probability Model for Ordinal Data from the
    caret package.
-   [Continuation Ratio Model for Ordinal
    Data](https://github.com/jbryer/mldash/blob/master/inst/models/caret_vglmContRatio_classification.dcf.dcf) -
    Continuation Ratio Model for Ordinal Data from the caret package.
-   [Cumulative Probability Model for Ordinal
    Data](https://github.com/jbryer/mldash/blob/master/inst/models/caret_vglmCumulative_classification.dcf.dcf) -
    Cumulative Probability Model for Ordinal Data from the caret
    package.
-   [Partial Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_widekernelpls_classification.dcf.dcf) -
    Partial Least Squares from the caret package.
-   [Partial Least
    Squares](https://github.com/jbryer/mldash/blob/master/inst/models/caret_widekernelpls_regression.dcf.dcf) -
    Partial Least Squares from the caret package.
-   [Wang and Mendel Fuzzy
    Rules](https://github.com/jbryer/mldash/blob/master/inst/models/caret_WM_regression.dcf.dcf) -
    Wang and Mendel Fuzzy Rules from the caret package.
-   [Weighted Subspace Random
    Forest](https://github.com/jbryer/mldash/blob/master/inst/models/caret_wsrf_classification.dcf.dcf) -
    Weighted Subspace Random Forest from the caret package.
-   [eXtreme Gradient
    Boosting](https://github.com/jbryer/mldash/blob/master/inst/models/caret_xgbDART_classification.dcf.dcf) -
    eXtreme Gradient Boosting from the caret package.
-   [eXtreme Gradient
    Boosting](https://github.com/jbryer/mldash/blob/master/inst/models/caret_xgbDART_regression.dcf.dcf) -
    eXtreme Gradient Boosting from the caret package.
-   [eXtreme Gradient
    Boosting](https://github.com/jbryer/mldash/blob/master/inst/models/caret_xgbLinear_classification.dcf.dcf) -
    eXtreme Gradient Boosting from the caret package.
-   [eXtreme Gradient
    Boosting](https://github.com/jbryer/mldash/blob/master/inst/models/caret_xgbLinear_regression.dcf.dcf) -
    eXtreme Gradient Boosting from the caret package.
-   [eXtreme Gradient
    Boosting](https://github.com/jbryer/mldash/blob/master/inst/models/caret_xgbTree_classification.dcf.dcf) -
    eXtreme Gradient Boosting from the caret package.
-   [eXtreme Gradient
    Boosting](https://github.com/jbryer/mldash/blob/master/inst/models/caret_xgbTree_regression.dcf.dcf) -
    eXtreme Gradient Boosting from the caret package.
-   [Self-Organizing
    Maps](https://github.com/jbryer/mldash/blob/master/inst/models/caret_xyf_classification.dcf.dcf) -
    Self-Organizing Maps from the caret package.
-   [Self-Organizing
    Maps](https://github.com/jbryer/mldash/blob/master/inst/models/caret_xyf_regression.dcf.dcf) -
    Self-Organizing Maps from the caret package.
-   [Fable Arima
    Timeseries](https://github.com/jbryer/mldash/blob/master/inst/models/fable_arima_timeseries.dcf.dcf) -
    The R package fable provides a collection of commonly used
    univariate and multivariate time series forecasting models.
-   [fable_tslm_timeseries](https://github.com/jbryer/mldash/blob/master/inst/models/fable_tslm_timeseries.dcf.dcf) -
    The R package fable provides a collection of commonly used
    univariate and multivariate time series forecasting models.
-   [Linear
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/lm.dcf.dcf) -
    Linear regression using the stats::lm function.
-   [Logistic
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/logistic.dcf.dcf) -
    Logistic regression using the stats::glm function.
-   [Neural network
    logistic-classification](https://github.com/jbryer/mldash/blob/master/inst/models/neuralnet_logit_classification.dcf.dcf) -
    Neural network logistic-classification prediction model using the
    neuralnet R package.
-   [prophet_timeseries](https://github.com/jbryer/mldash/blob/master/inst/models/prophet_timeseries.dcf.dcf) -
    Prophet is a forecasting procedure implemented in R and Python.
-   [Random Forests
    Classification](https://github.com/jbryer/mldash/blob/master/inst/models/randomForest_classification.dcf.dcf) -
    Random forest prediction model usign the randomForest R package.
-   [Random Forest
    Regression](https://github.com/jbryer/mldash/blob/master/inst/models/randomForest_regression.dcf.dcf) -
    Random forest prediction model usign the randomForest R package.
-   [tm_bag_mars_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_bag_mars_classification.dcf.dcf) -
    Ensemble of generalized linear models that use artificial features
    for some predictors.
-   [tm_bag_mars_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_bag_mars_regression.dcf.dcf) -
    Ensemble of generalized linear models that use artificial features
    for some predictors.
-   [tm_bag_tree_C50_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_bag_tree_C50_classification.dcf.dcf) -
    Creates an collection of decision trees forming an ensemble. All
    trees in the ensemble are combined to produce a final prediction.
-   [tm_bag_tree_rpart_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_bag_tree_rpart_classification.dcf.dcf) -
    Ensembles of decision trees.
-   [tm_bag_tree_rpart_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_bag_tree_rpart_regression.dcf.dcf) -
    Ensembles of decision trees.
-   [tm_bart_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_bart_classification.dcf.dcf) -
    Defines a tree ensemble model that uses Bayesian analysis to
    assemble the ensemble. This function can fit classification and
    regression models.
-   [tm_bart_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_bart_regression.dcf.dcf) -
    Defines a tree ensemble model that uses Bayesian analysis to
    assemble the ensemble. This function can fit classification and
    regression models.
-   [tm_boost_tree_C50_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_boost_tree_C50_classification.dcf.dcf) -
    Defines a model that creates a series of decision trees forming an
    ensemble. Each tree depends on the results of previous trees. All
    trees in the ensemble are combined to produce a final prediction.
-   [tm_boost_tree_xgboost_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_boost_tree_xgboost_classification.dcf.dcf) -
    Defines a model that creates a series of decision trees forming an
    ensemble. Each tree depends on the results of previous trees. All
    trees in the ensemble are combined to produce a final prediction.
-   [tm_boost_tree_xgboost_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_boost_tree_xgboost_regression.dcf.dcf) -
    Defines a model that creates a series of decision trees forming an
    ensemble. Each tree depends on the results of previous trees. All
    trees in the ensemble are combined to produce a final prediction.
-   [tm_decision_tree_rpart_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_decision_tree_rpart_classification.dcf.dcf) -
    Defines a model as a set of if/then statements that creates a
    tree-based structure.
-   [tm_decision_tree_rpart_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_decision_tree_rpart_regression.dcf.dcf) -
    Defines a model as a set of if/then statements that creates a
    tree-based structure.
-   [tm_discrim_flexible_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_discrim_flexible_classification.dcf.dcf) -
    Defines a model that fits a discriminant analysis model that can use
    nonlinear features created using multivariate adaptive regression
    splines (MARS).
-   [tm_discrim_linear_MASS_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_discrim_linear_MASS_classification.dcf.dcf) -
    Defines a model that estimates a multivariate distribution for the
    predictors separately for the data in each class (usually Gaussian
    with a common covariance matrix). Bayes’ theorem is used to compute
    the probability of each class, given the predictor values.
-   [tm_discrim_linear_mda_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_discrim_linear_mda_classification.dcf.dcf) -
    Defines a model that estimates a multivariate distribution for the
    predictors separately for the data in each class (usually Gaussian
    with a common covariance matrix). Bayes’ theorem is used to compute
    the probability of each class, given the predictor values.
-   [tm_discrim_linear_sda_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_discrim_linear_sda_classification.dcf.dcf) -
    Defines a model that estimates a multivariate distribution for the
    predictors separately for the data in each class (usually Gaussian
    with a common covariance matrix). Bayes’ theorem is used to compute
    the probability of each class, given the predictor values.
-   [tm_discrim_linear_sparsediscrim_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_discrim_linear_sparsediscrim_classification.dcf.dcf) -
    Defines a model that estimates a multivariate distribution for the
    predictors separately for the data in each class (usually Gaussian
    with a common covariance matrix). Bayes’ theorem is used to compute
    the probability of each class, given the predictor values.
-   [tm_discrim_regularized_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_discrim_regularized_classification.dcf.dcf) -
    Defines a model that estimates a multivariate distribution for the
    predictors separately for the data in each class. The structure of
    the model can be LDA, QDA, or some amalgam of the two. Bayes’
    theorem is used to compute the probability of each class, given the
    predictor values.
-   [tm_exp_smoothing_ets_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_exp_smoothing_regression.dcf.dcf) -
    exp_smoothing() is a way to generate a specification of an
    Exponential Smoothing model before fitting and allows the model to
    be created using different packages.
-   [tm_gen_additive_mod_mgcv_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_gen_additive_mod_mgcv_classification.dcf.dcf) -
    gen_additive_mod() defines a model that can use smoothed functions
    of numeric predictors in a generalized linear model.
-   [tm_gen_additive_mod_mgcv_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_gen_additive_mod_mgcv_regression.dcf.dcf) -
    gen_additive_mod() defines a model that can use smoothed functions
    of numeric predictors in a generalized linear model.
-   [tm_linear_reg_glm_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_linear_reg_glm_regression.dcf.dcf) -
    linear_reg() defines a model that can predict numeric values from
    predictors using a linear function.
-   [tm_linear_reg_glmnet_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_linear_reg_glmnet_regression.dcf.dcf) -
    linear_reg() defines a model that can predict numeric values from
    predictors using a linear function.
-   [tm_linear_reg_keras_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_linear_reg_keras_regression.dcf.dcf) -
    linear_reg() defines a model that can predict numeric values from
    predictors using a linear function.
-   [tm_linear_reg_lm_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_linear_reg_lm_regression.dcf.dcf) -
    linear_reg() defines a model that can predict numeric values from
    predictors using a linear function.
-   [tm_linear_reg_stan_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_linear_reg_stan_regression.dcf.dcf) -
    linear_reg() defines a model that can predict numeric values from
    predictors using a linear function.
-   [tm_logistic_brulee_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_logistic_brulee_classification.dcf.dcf) -
    Defines a generalized linear model for binary outcomes. A linear
    combination of the predictors is used to model the log odds of an
    event.
-   [tm_logistic_glm_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_logistic_glm_classification.dcf.dcf) -
    Defines a generalized linear model for binary outcomes. A linear
    combination of the predictors is used to model the log odds of an
    event.
-   [tm_logistic_glmnet_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_logistic_glmnet_classification.dcf.dcf) -
    Defines a generalized linear model for binary outcomes. A linear
    combination of the predictors is used to model the log odds of an
    event.
-   [tm_logistic_keras_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_logistic_keras_classification.dcf.dcf) -
    Defines a generalized linear model for binary outcomes. A linear
    combination of the predictors is used to model the log odds of an
    event.
-   [tm_logistic_LiblineaR_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_logistic_liblinear_classification.dcf.dcf) -
    Defines a generalized linear model for binary outcomes. A linear
    combination of the predictors is used to model the log odds of an
    event.
-   [tm_logistic_stan_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_logistic_stan_classification.dcf.dcf) -
    Defines a generalized linear model for binary outcomes. A linear
    combination of the predictors is used to model the log odds of an
    event.
-   [tm_mars](https://github.com/jbryer/mldash/blob/master/inst/models/tm_mars_classification.dcf.dcf) -
    Defines a generalized linear model that uses artificial features for
    some predictors. These features resemble hinge functions and the
    result is a model that is a segmented regression in small
    dimensions.
-   [tm_mars_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_mars_regression.dcf.dcf) -
    Defines a generalized linear model that uses artificial features for
    some predictors. These features resemble hinge functions and the
    result is a model that is a segmented regression in small
    dimensions.
-   [tm_mlp_brulee_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_mlp_brulee_classification.dcf.dcf) -
    Defines a multilayer perceptron model (a.k.a. a single layer,
    feed-forward neural network).
-   [tm_mlp_brulee_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_mlp_brulee_regression.dcf.dcf) -
    Defines a multilayer perceptron model (a.k.a. a single layer,
    feed-forward neural network).
-   [tm_mlp_keras_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_mlp_keras_classification.dcf.dcf) -
    Defines a multilayer perceptron model (a.k.a. a single layer,
    feed-forward neural network).
-   [tm_mlp_keras_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_mlp_keras_regression.dcf.dcf) -
    Defines a multilayer perceptron model (a.k.a. a single layer,
    feed-forward neural network).
-   [tm_mlp_nnet_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_mlp_nnet_classification.dcf.dcf) -
    Defines a multilayer perceptron model (a.k.a. a single layer,
    feed-forward neural network).
-   [tm_mlp_nnet_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_mlp_nnet_regression.dcf.dcf) -
    Defines a multilayer perceptron model (a.k.a. a single layer,
    feed-forward neural network).
-   [tm_naive_bayes_klaR](https://github.com/jbryer/mldash/blob/master/inst/models/tm_naive_bayes_klaR_classification.dcf.dcf) -
    Model that uses Bayes’ theorem to compute the probability of each
    class, given the predictor values.
-   [tm_naive_bayes_naivebayes](https://github.com/jbryer/mldash/blob/master/inst/models/tm_naive_bayes_naivebayes_classification.dcf.dcf) -
    Model that uses Bayes’ theorem to compute the probability of each
    class, given the predictor values.
-   [tm_nearest_neighbor_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_nearest_neighbor_classification.dcf.dcf) -
    Model that uses the K most similar data points from the training set
    to predict new samples.
-   [tm_nearest_neighbor_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_nearest_neighbor_regression.dcf.dcf) -
    Model that uses the K most similar data points from the training set
    to predict new samples.
-   [tm_null_model_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_null_model_classification.dcf.dcf) -
    Defines a simple, non-informative model.
-   [tm_null_model_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_null_model_regression.dcf.dcf) -
    Defines a simple, non-informative model.
-   [tm_pls_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_pls_classification.dcf.dcf) -
    Defines a partial least squares model that uses latent variables to
    model the data. It is similar to a supervised version of principal
    component.
-   [tm_pls_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_pls_regression.dcf.dcf) -
    Defines a partial least squares model that uses latent variables to
    model the data. It is similar to a supervised version of principal
    component.
-   [tm_poisson_reg_glm_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_poisson_reg_glm_regression.dcf.dcf) -
    Defines a generalized linear model for count data that follow a
    Poisson distribution.
-   [tm_poisson_reg_glmnet_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_poisson_reg_glmnet_regression.dcf.dcf) -
    Defines a generalized linear model for count data that follow a
    Poisson distribution.
-   [tm_poisson_reg_stan_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_poisson_reg_stan_regression.dcf.dcf) -
    Defines a generalized linear model for count data that follow a
    Poisson distribution.
-   [tm_poisson_reg_zeroinfl_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_poisson_reg_zeroinfl_regression.dcf.dcf) -
    Defines a generalized linear model for count data that follow a
    Poisson distribution.
-   [tm_rand_forest_randomForest_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_rand_forest_randomForest_classification.dcf.dcf) -
    Defines a model that creates a large number of decision trees, each
    independent of the others. The final prediction uses all predictions
    from the individual trees and combines them.
-   [tm_rand_forest_randomForest_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_rand_forest_randomForest_regression.dcf.dcf) -
    Defines a model that creates a large number of decision trees, each
    independent of the others. The final prediction uses all predictions
    from the individual trees and combines them.
-   [tm_rand_forest_ranger_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_rand_forest_ranger_classification.dcf.dcf) -
    Defines a model that creates a large number of decision trees, each
    independent of the others. The final prediction uses all predictions
    from the individual trees and combines them.
-   [tm_rand_forest_ranger_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_rand_forest_ranger_regression.dcf.dcf) -
    Defines a model that creates a large number of decision trees, each
    independent of the others. The final prediction uses all predictions
    from the individual trees and combines them.
-   [tm_rule_fit_xrf_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_rule_fit_xrf_classification.dcf.dcf) -
    Defines a model that derives simple feature rules from a tree
    ensemble and uses them as features in a regularized model.
-   [tm_svm_linear_kernlab_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_svm_linear_kernlab_classification.dcf.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    (using a linear class boundary). For regression, the model optimizes
    a robust loss function that is only affected by very large model
    residuals and uses a linear fit.
-   [tm_svm_linear_kernlab_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_svm_linear_kernlab_regression.dcf.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    (using a linear class boundary). For regression, the model optimizes
    a robust loss function that is only affected by very large model
    residuals and uses a linear fit.
-   [tm_svm_linear_LiblineaR_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_svm_linear_LiblineaR_classification.dcf.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    (using a linear class boundary). For regression, the model optimizes
    a robust loss function that is only affected by very large model
    residuals and uses a linear fit.
-   [tm_svm_poly_kernlab_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_svm_poly_kernlab_classification.dcf.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    using a polynomial class boundary. For regression, the model
    optimizes a robust loss function that is only affected by very large
    model residuals and uses polynomial functions of the predictors.
-   [tm_svm_poly_kernlab_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_svm_poly_kernlab_regression.dcf.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    using a polynomial class boundary. For regression, the model
    optimizes a robust loss function that is only affected by very large
    model residuals and uses polynomial functions of the predictors.
-   [tm_svm_rbf_kernlab_classification](https://github.com/jbryer/mldash/blob/master/inst/models/tm_svm_rbf_kernlab_classification.dcf.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    using a nonlinear class boundary. For regression, the model
    optimizes a robust loss function that is only affected by very large
    model residuals and uses nonlinear functions of the predictors.
-   [tm_svm_rbf_kernlab_regression](https://github.com/jbryer/mldash/blob/master/inst/models/tm_svm_rbf_kernlab_regression.dcf.dcf) -
    Defines a support vector machine model. For classification, the
    model tries to maximize the width of the margin between classes
    using a polynomial class boundary. For regression, the model
    optimizes a robust loss function that is only affected by very large
    model residuals and uses polynomial functions of the predictors.
-   [weka_bagging_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka_bagging.dcf.dcf) -
    Bagging (Breiman, 1996)
-   [weka_decisionstump_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka_decisionstump.dcf.dcf) -
    Implements decision stumps (trees with a single split only), which
    are frequently used as base learners for meta learners such as
    Boosting.
-   [weka_ibk_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka_ibk.dcf.dcf) -
    Provides a k-nearest neighbors classifier, see Aha & Kibler (1991).
-   [weka_J48_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka_J48_classification.dcf.dcf) -
    Class for generating a pruned or unpruned C4.5 decision tree.
-   [weka_jrip_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka_jrip.dcf.dcf) -
    Implements a propositional rule learner, “Repeated Incremental
    Pruning to Produce Error Reduction” (RIPPER), as proposed by Cohen
    (1995).
-   [weka_lmt_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka_lmt.dcf.dcf) -
    Implements “Logistic Model Trees” (Landwehr, 2003; Landwehr et al.,
    2005).
-   [weka_logistic_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka_logistic.dcf.dcf) -
    Builds multinomial logistic regression models based on ridge
    estimation (le Cessie and van Houwelingen, 1992).
-   [weka_logitboost_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka_logitboost.dcf.dcf) -
    Performs boosting via additive logistic regression (Friedman, Hastie
    and Tibshirani, 2000).
-   [weka_oner_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka_oner.dcf.dcf) -
    Builds a simple 1-R classifier, see Holte (1993).
-   [weka_part_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka_part.dcf.dcf) -
    Generates PART decision lists using the approach of Frank and Witten
    (1998).
-   [weka_smo_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka_smo.dcf.dcf) -
    Implements John C. Platt’s sequential minimal optimization algorithm
    for training a support vector classifier using polynomial or RBF
    kernels.
-   [weka_stacking_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka_stacking.dcf.dcf) -
    Provides stacking (Wolpert, 1992).
-   [weka_adaboostm1_classification](https://github.com/jbryer/mldash/blob/master/inst/models/weka.adaboostm1.dcf.dcf) -
    Implements the AdaBoost M1 method of Freund and Schapire (1996).

## Code of Conduct

Please note that the mldash project is released with a [Contributor Code
of
Conduct](https://contributor-covenant.org/version/2/0/CODE_OF_CONDUCT.html).
By contributing to this project, you agree to abide by its terms.

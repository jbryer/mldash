
<img src="man/figures/mldash.png" align="right" width="120" />

# `mldash`: Machine Learning Dashboard

<!-- badges: start -->

[![Project Status: WIP – Initial development is in progress, but there
has not yet been a stable, usable release suitable for the
public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
<!-- badges: end -->

**Author: [Jason Bryer, Ph.D.](mailto:jason@bryer.org)**  
**Website: <https://github.com/jbryer/mldash>**

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
tensorflow::use_condaenv("mldash")
keras::use_condaenv("mldash")
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

-   [abalone](inst/datasets/abalone.dcf) - Predicting the age of abalone
    from physical measurements.
-   [adult](inst/datasets/adult.dcf) - Prediction task is to determine
    whether a person makes over 50K a year.
-   [ames](inst/datasets/ames.dcf) - Ames Housing Data.
-   [titanic](inst/datasets/titanic.dcf) - The original Titanic dataset,
    describing the survival status of individual passengers on the
    Titanic.

## Available Models

Each model is defined in a Debian Control File (DCF) format the details
of which are described below. Below is the list of models included in
the `mldash` package. Note that models that begin with `tm_` are models
implemented with the [`tidymodels`](https://www.tidymodels.org) R
package; models that begin with `weka_` are models implemented with the
the [`RWeka`](https://cran.r-project.org/web/packages/RWeka/index.html)
which is a wrapper to the [Weka](https://www.cs.waikato.ac.nz/ml/weka/)
collection of machine learning algorithms.

-   [lm](inst/models/lm.dcf) - Linear regression using the stats::lm
    function.
-   [logistic](inst/models/logistic.dcf) - Logistic regression using the
    stats::glm function.
-   [randomForest_classification](inst/models/randomForest_classification.dcf) -
    Random forest prediction model usign the randomForest R package.
-   [randomForest_regression](inst/models/randomForest_regression.dcf) -
    Random forest prediction model usign the randomForest R package.
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
-   [weka_jrip_classification](inst/models/wkea_jrip.dcf) - Implements a
    propositional rule learner, “Repeated Incremental Pruning to Produce
    Error Reduction” (RIPPER), as proposed by Cohen (1995).

## Creating Datasets

``` r
adult_data <- mldash::new_dataset(
    name = 'adult',
    type = 'classification',
    description = 'Prediction task is to determine whether a person makes over 50K a year.',
    source = 'https://archive.ics.uci.edu/ml/datasets/Adult',
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
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
    url: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
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

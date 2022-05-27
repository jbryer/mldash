
# mldash

<!-- badges: start -->
<!-- badges: end -->

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
#> For binary classification, the first factor level is assumed to be the event.
#> Use the argument `event_level = "second"` to alter this as needed.
#> [1 / 3] Loading abalone data...
#>    [1 / 13] Running lm model...
#>    [2 / 13] Running randomForest_regression model...
#>    [3 / 13] Running tm_bag_mars_regression model...
#>    [4 / 13] Running tm_bag_tree_rpart_regression model...
#>    [5 / 13] Running tm_bart_regression model...
#>    [6 / 13] Running tm_boost_tree_xgboost_regression model...
#>    [7 / 13] Running tm_decision_tree_rpart_regression model...
#>    [8 / 13] Running tm_exp_smoothing_ets_regression model...
#>    [9 / 13] Running tm_gen_additive_mod_mgcv_regression model...
#>    [10 / 13] Running tm_linear_reg_glm_regression model...
#>    [11 / 13] Running tm_linear_reg_glmnet_regression model...
#>    [12 / 13] Running tm_linear_reg_lm_regression model...
#>    [13 / 13] Running tm_linear_reg_stan_regression model...
#> [2 / 3] Loading ames data...
#>    [1 / 13] Running lm model...
#>    [2 / 13] Running randomForest_regression model...
#>    [3 / 13] Running tm_bag_mars_regression model...
#>    [4 / 13] Running tm_bag_tree_rpart_regression model...
#>    [5 / 13] Running tm_bart_regression model...
#>    [6 / 13] Running tm_boost_tree_xgboost_regression model...
#>    [7 / 13] Running tm_decision_tree_rpart_regression model...
#>    [8 / 13] Running tm_exp_smoothing_ets_regression model...
#>    [9 / 13] Running tm_gen_additive_mod_mgcv_regression model...
#>    [10 / 13] Running tm_linear_reg_glm_regression model...
#>    [11 / 13] Running tm_linear_reg_glmnet_regression model...
#>    [12 / 13] Running tm_linear_reg_lm_regression model...
#>    [13 / 13] Running tm_linear_reg_stan_regression model...
#> [3 / 3] Loading titanic data...
#>    [1 / 30] Running logistic model...
#>    [2 / 30] Running randomForest_classification model...
#>    [3 / 30] Running tm_bag_mars_classification model...
#>    [4 / 30] Running tm_bag_tree_C50_classification model...
#>    [5 / 30] Running tm_bag_tree_rpart_classification model...
#>    [6 / 30] Running tm_bart_classification model...
#>    [7 / 30] Running tm_boost_tree_C50_classification model...
#>    [8 / 30] Running tm_boost_tree_xgboost_classification model...
#>    [9 / 30] Running tm_decision_tree_rpart_classification model...
#>    [10 / 30] Running tm_discrim_flexible_classification model...
#>    [11 / 30] Running tm_discrim_linear_MASS_classification model...
#>    [12 / 30] Running tm_discrim_linear_mda_classification model...
#>    [13 / 30] Running tm_discrim_linear_sda_classification model...
#>    [14 / 30] Running tm_discrim_linear_sparsediscrim_classification model...
#>    [15 / 30] Running tm_discrim_regularized_classification model...
#>    [16 / 30] Running tm_gen_additive_mod_mgcv_classification model...
#>    [17 / 30] Running tm_naive_bayes model...
#>    [18 / 30] Running weka_bagging_classification model...
#>    [19 / 30] Running weka_decisionstump_classification model...
#>    [20 / 30] Running weka_ibk_classification model...
#>    [21 / 30] Running weka_J48_classification model...
#>    [22 / 30] Running weka_lmt_classification model...
#>    [23 / 30] Running weka_logistic_classification model...
#>    [24 / 30] Running weka_logitboost_classification model...
#>    [25 / 30] Running weka_oner_classification model...
#>    [26 / 30] Running weka_part_classification model...
#>    [27 / 30] Running weka_smo_classification model...
#>    [28 / 30] Running weka_stacking_classification model...
#>    [29 / 30] Running weka_adaboostm1_classification model...
#>    [30 / 30] Running weka_jrip_classification model...
```

The `metrics` parameter to `run_models()` takes a list of metrics from
the [`yardstick`](https://yardstick.tidymodels.org/index.html) package
(Kuhn & Vaughan, 2021). The full list of metris is available here:
<https://yardstick.tidymodels.org/articles/metric-types.html>

## Available Datasets

-   [abalone](inst/datasets/abalone.dcf) - Predicting the age of abalone
    from physical measurements.
-   [ames](inst/datasets/ames.dcf) - Ames Housing Data.
-   [titanic](inst/datasets/titanic.dcf) - The original Titanic dataset,
    describing the survival status of individual passengers on the
    Titanic.

## Available Models

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
-   [tm_linear_reg_lm_regression](inst/models/tm_linear_reg_lm_regression.dcf) -
    linear_reg() defines a model that can predict numeric values from
    predictors using a linear function.
-   [tm_linear_reg_stan_regression](inst/models/tm_linear_reg_stan_regression.dcf) -
    linear_reg() defines a model that can predict numeric values from
    predictors using a linear function.
-   [tm_naive_bayes](inst/models/tm_naive_bayes_classification.dcf) -
    Model that uses Bayes’ theorem to compute the probability of each
    class, given the predictor values.
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

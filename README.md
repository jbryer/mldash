
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
ml_results <- mldash::run_models(datasets = ml_datasets, models = ml_models)
#> For binary classification, the first factor level is assumed to be the event.
#> Use the argument `event_level = "second"` to alter this as needed.
#> [1 / 3] Loading abalone data...
#>    [1 / 5] Running bag_mars_regression model...
#>    [2 / 5] Running bag_tree_rpart_regression model...
#>    [3 / 5] Running bart_regression model...
#>    [4 / 5] Running lm model...
#>    [5 / 5] Running randomForest_regression model...
#> [2 / 3] Loading ames data...
#>    [1 / 5] Running bag_mars_regression model...
#>    [2 / 5] Running bag_tree_rpart_regression model...
#>    [3 / 5] Running bart_regression model...
#>    [4 / 5] Running lm model...
#>    [5 / 5] Running randomForest_regression model...
#> [3 / 3] Loading titanic data...
#>    [1 / 7] Running bag_mars_classification model...
#>    [2 / 7] Running bag_tree_C50_classification model...
#>    [3 / 7] Running bag_tree_rpart_classification model...
#>    [4 / 7] Running bart_classification model...
#>    [5 / 7] Running logistic model...
#>    [6 / 7] Running naive_bayes model...
#>    [7 / 7] Running randomForest_classification model...
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

-   [bag_mars_classification](inst/models/bag_mars_classification.dcf) -
    Ensemble of generalized linear models that use artificial features
    for some predictors.
-   [bag_mars_regression](inst/models/bag_mars_regression.dcf) -
    Ensemble of generalized linear models that use artificial features
    for some predictors.
-   [bag_tree_C50_classification](inst/models/bag_tree_C50_classification.dcf) -
    Creates an collection of decision trees forming an ensemble. All
    trees in the ensemble are combined to produce a final prediction.
-   [bag_tree_rpart_classification](inst/models/bag_tree_rpart_classification.dcf) -
    Ensembles of decision trees.
-   [bag_tree_rpart_regression](inst/models/bag_tree_rpart_regression.dcf) -
    Ensembles of decision trees.
-   [bart_classification](inst/models/bart_classification.dcf) - Defines
    a tree ensemble model that uses Bayesian analysis to assemble the
    ensemble. This function can fit classification and regression
    models.
-   [bart_regression](inst/models/bart_regression.dcf) - Defines a tree
    ensemble model that uses Bayesian analysis to assemble the ensemble.
    This function can fit classification and regression models.
-   [lm](inst/models/lm.dcf) - Linear regression using the stats::lm
    function.
-   [logistic](inst/models/logistic.dcf) - Logistic regression using the
    stats::glm function.
-   [naive_bayes](inst/models/naive_bayes_classification.dcf) - Model
    that uses Bayes’ theorem to compute the probability of each class,
    given the predictor values.
-   [randomForest_classification](inst/models/randomForest_classification.dcf) -
    Random forest prediction model usign the randomForest R package.
-   [randomForest_regression](inst/models/randomForest_regression.dcf) -
    Random forest prediction model usign the randomForest R package.

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

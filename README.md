
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
                                        cache_dir = 'data-raw')
ml_datasets %>% select(name, type, model)
#>                name           type
#> abalone.dcf abalone     regression
#> adult.dcf     adult classification
#> titanic.dcf titanic classification
#>                                                                        model
#> abalone.dcf                                             rings ~ length + sex
#> adult.dcf                                               greater_than_50k ~ .
#> titanic.dcf survived ~  pclass + sex + age + sibsp + parch + fare + embarked
```

Similarly, the `read_ml_models` will read in the models. The `dir`
parameter defines where to look for model files.

``` r
ml_models <- mldash::read_ml_models(dir = 'inst/models')
ml_models
#>                          name           type
#> lm.dcf                     lm     regression
#> logistic.dcf         logistic classification
#> randomForest.dcf randomForest classification
#>                                                                       description
#> lm.dcf                            Linear regression using the stats::lm function.
#> logistic.dcf                   Logistic regression using the stats::glm function.
#> randomForest.dcf Random forest prediction model usign the randomForest R package.
#>                  note     packages
#> lm.dcf           <NA>         <NA>
#> logistic.dcf     <NA>         <NA>
#> randomForest.dcf      randomForest
```

Once the datasets and models have been loaded, the `run_models` will
train and evaluate each model for each dataset as appropriate for the
model type.

``` r
ml_results <- mldash::run_models(datasets = ml_datasets, models = ml_models)
#> [1 / 3] Loading abalone data...
#>    [1 / 1] Running lm model...
#> [2 / 3] Loading adult data...
#>    [1 / 2] Running logistic model...
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
#> Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
#> prediction from a rank-deficient fit may be misleading
#>    [2 / 2] Running randomForest model...
#> Timing stopped at: 0.007 0 0.008
#>    Error running randomForest.dcf model
#> <simpleError in eval(predvars, data, env): object 'education-num' not found>
#> [3 / 3] Loading titanic data...
#>    [1 / 2] Running logistic model...
#>    [2 / 2] Running randomForest model...
ml_results
#>   dataset            model           type base_accuracy time_user time_system
#> 1 abalone           lm.dcf     regression            NA     0.002       0.000
#> 2   adult     logistic.dcf classification          0.75     3.324       0.050
#> 3 titanic     logistic.dcf classification          0.60     0.002       0.000
#> 4 titanic randomForest.dcf classification          0.60     0.363       0.013
#>   time_elapsed accuracy kappa sensitivity specificity roc_auc r_squared rmse
#> 1        0.002       NA    NA          NA          NA      NA       0.3  2.6
#> 2        3.380     0.85  0.57        0.94        0.59   0.093        NA   NA
#> 3        0.003     0.76  0.49        0.85        0.62   0.176        NA   NA
#> 4        0.377     0.80  0.57        0.91        0.65   0.174        NA   NA
```

## Available Datasets

-   [abalone](inst/datasets/abalone.dcf) - Predicting the age of abalone
    from physical measurements.
-   [adult](inst/datasets/adult.dcf) - Prediction task is to determine
    whether a person makes over 50K a year.
-   [titanic](inst/datasets/titanic.dcf) - The original Titanic dataset,
    describing the survival status of individual passengers on the
    Titanic.

## Available Models

-   [lm](inst/models/lm.dcf) - Linear regression using the stats::lm
    function.
-   [logistic](inst/models/logistic.dcf) - Logistic regression using the
    stats::glm function.
-   [randomForest](inst/models/randomForest.dcf) - Random forest
    prediction model usign the randomForest R package.

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
    name = 'randomForest',
    type = 'classification',
    description = 'Random forest prediction model usign the randomForest R package.',
    train_fun = function(formula, data) {
        y_var <- all.vars(formula)[1]
        if(!is.factor(data[,y_var])) {
            data[,y_var] <- as.factor(data[,y_var])
        }
        randomForest::randomForest(formula = formula, data = data, ntree = 1000)
    },
    predict_fun = function(model, newdata) {
        y_var <- all.vars(model$terms)[1]
        if(!is.factor(newdata[,y_var])) {
            newdata[,y_var] <- as.factor(newdata[,y_var])
        }
        randomForest:::predict.randomForest(model, newdata = newdata, type = "prob")[,2,drop=TRUE]
    },
    packages = "randomForest",
    overwrite = TRUE
)
```

Results in the following file:

    name: randomForest
    type: classification
    description: Random forest prediction model usign the randomForest R package.
    train: function (formula, data) 
        {
            y_var <- all.vars(formula)[1]]
            if(!is.factor(data[,y_var)) {
                data[,y_var] <- as.factor(data[,y_var])
            }
            randomForest::randomForest(formula = formula, data = data, 
                ntree = 1000)
        }
    predict: function (model, newdata) 
        {
            y_var <- all.vars(formula)[1]]
            if(!is.factor(newdata[,y_var)) {
                newdata[,y_var] <- as.factor(newdata[,y_var])
            }
            randomForest:::predict.randomForest(model, newdata = newdata, type = "prob")[,2,drop=TRUE]
        }
    note:

## Code of Conduct

Please note that the mldash project is released with a [Contributor Code
of
Conduct](https://contributor-covenant.org/version/2/0/CODE_OF_CONDUCT.html).
By contributing to this project, you agree to abide by its terms.

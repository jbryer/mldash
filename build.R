library(devtools)
library(usethis)

usethis::use_tidy_description()
usethis::use_spell_check()

devtools::document()
devtools::install()
devtools::build()
devtools::check()


# Added package dependencies
usethis::use_package('tidymodels', type = "Imports")
usethis::use_package('RWeka', type = "Imports")
usethis::use_package('randomForest', type = "Imports")
usethis::use_package('shinyWidgets', type = "Imports")

usethis::use_package_doc()

##### Build data sets ##########################################################

# UCI data sets are located here: https://archive.ics.uci.edu/ml/datasets.php
# OpenML datasets here: https://www.openml.org/search?type=data&sort=runs
# Weka datasets here: https://waikato.github.io/weka-wiki/datasets/

library(mldash)

# Clean data cache
unlink('inst/datasets/*.rds')

# TODO:
# * Look for data within the package for built-in datasets.
# * Create Roxygen2 documentation for data from the metadata files.
# * Vignette.
# * Shiny application
# * add testthat tests
# * add an example using OpenML package
# * check for package dependencies from the model files.


adult <- mldash::get_data(dataname = 'adult', dir = 'inst/datasets')

abalone <- mldash::get_data(dataname = 'abalone', dir = 'inst/datasets')
formula <- rings ~ length + sex

thedata <- abalone

# [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
library(ucimlr) # remotes::install_github("tyluRp/ucimlr")
ucidata() |> View()
automobile


# Tidymodels example
library(parsnip)
library(baguette)
fit <- bag_mars(num_terms = 7) |>
	set_mode("regression") |>
	set_engine("earth", times = 3) |>
	fit(formula, data = abalone)
parsnip::predict_raw.model_fit(fit, new_data = abalone)[,1,drop=TRUE]
predict.model_fit(fit, new_data = abalone)

# Weka example
# https://www.zeileis.org/papers/Hornik+Buchta+Zeileis-2009.pdf
# https://cran.r-project.org/web/packages/RWeka/vignettes/RWeka.pdf
# https://www.cs.waikato.ac.nz/~ml/weka/
library(RWeka)
WPM("refresh-cache")
WPM("list-packages", "installed")
WPM("list-packages", "available")

WPM('install-package', 'lazyBayesianRules')
LBR <- make_Weka_classifier("weka/classifiers/lazy/LBR",
						    c("LBR", "Weka_lazy"),
						    package = "lazyBayesianRules")
m1 <- LBR(Species ~ ., data = iris)

J48 <- make_Weka_classifier("weka/classifiers/trees/J48", c("bar", "Weka_tree"))
m1 <- J48(formu, data = thedata)
m1
predict(m1, thedata, type = 'probability')

titanic <- readRDS('inst/datasets/titanic.rds')
titanic.formula <- survived ~  pclass + sex + age + sibsp + parch + fare + embarked

WPM('install-package', 'weka/classifiers/AnDE')
WPM('package-info', 'repository', 'AnDE')
AnDE <- make_Weka_classifier('weka/classifiers/bayes/AveragedNDependenceEstimators/A1DE')

# Classifier Functions
logistic.fit <- RWeka::Logistic(titanic.formula, titanic)
smo.fit <- RWeka::SMO(titanic.formula, titanic)

# Lazy
ibk.fit <- RWeka::IBk(titanic.formula, titanic)
# WPM('install-package', 'lazyBayesianRules')
# lbr.fit <- RWeka::LBR(titanic.formula, titanic)

# Meta
adaboostm1.fit <- RWeka::AdaBoostM1(titanic.formula, titanic)
bagging.fit <- RWeka::Bagging(titanic.formula, titanic)
logitboost.fit <- RWeka::LogitBoost(titanic.formula, titanic)
# WPM('install-package', 'MultiBoostAB')
# multiboostab.fit <- RWeka::MultiBoostAB(titanic.formula, titanic)
stacking.fit <- RWeka::Stacking(titanic.formula, titanic)
# CostSensitiveClassifier.fit <- RWeka::CostSensitiveClassifier(titanic.formula, titanic)

# Rule Learners
jrip.fit <- RWeka::JRip(titanic.formula, titanic)
# m5rules <- RWeka::M5Rules(titanic.formula, titanic)
oner.fit <- RWeka::OneR(titanic.formula, titanic)
part.fit <- RWeka::PART(titanic.formula, titanic)

# Classifier Trees
j48.fit <- RWeka::J48(titanic.formula, titanic)
lmt.fit <- RWeka::LMT(titanic.formula, titanic)
# m5p.fit <- RWeka::M5P(titanic.formula, titanic)
DecisionStump.fit <- RWeka::DecisionStump(titanic.formula, titanic)




##### Run models
devtools::install()

model_pattern <- 'weka*'
model_pattern <- 'tm_discrim_*'
model_pattern <- '*.dcf'

ml_datasets <- mldash::read_ml_datasets(dir = 'inst/datasets',
										cache_dir = 'inst/datasets')

ml_models <- mldash::read_ml_models(dir = 'inst/models',
									pattern = model_pattern)

ml_results <- mldash::run_models(datasets = ml_datasets, models = ml_models, print_errors = FALSE)
ml_results |> View()

errors <- attr(ml_results, 'errors')
names(errors)

si <- attr(ml_results, 'session_info')
ls(si)
attr(ml_results, 'start_time')
si$platform$os

# Run only classification models/datasets
datasets <- ml_datasets %>% filter(type == 'classification')
models <- ml_models %>% filter(type == 'classification')



(si <- sessioninfo::session_info())

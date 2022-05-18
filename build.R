library(devtools)
library(usethis)

usethis::use_tidy_description()

document()
install()
build()
check()

# Added package dependencies
usethis::use_package('tidymodels', type = "Imports")
usethis::use_package('RWeka', type = "Imports")
usethis::use_package('randomForest', type = "Imports")
usethis::use_package('yardstick', type = "Imports")

usethis::use_package_doc()

##### Build data sets ##########################################################

# UCI data sets are located here: https://archive.ics.uci.edu/ml/datasets.php
# OpenML datasets here: https://www.openml.org/search?type=data&sort=runs

library(mldash)

# Clean data cache
unlink('data-raw/*.rds')

# TODO:
# * Look for data within the package for built-in datasets.
# * Create Roxygen2 documentation for data from the metadata files.
# * Vignette.
# * Shiny application
# * add testthat tests
# * add an example using OpenML package
# * check for package dependencies from the model files.

##### Run models

ml_datasets <- mldash::read_ml_datasets(dir = 'inst/datasets',
										cache_dir = 'data-raw')

ml_models <- mldash::read_ml_models(dir = 'inst/models')

ml_results <- mldash::run_models(datasets = ml_datasets, models = ml_models)
ml_results

# Run only classification models/datasets
datasets <- ml_datasets %>% filter(type == 'classification')
models <- ml_models %>% filter(type == 'classification')


sessioninfo::session_info()

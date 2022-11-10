library(devtools)
library(usethis)

usethis::use_tidy_description()
usethis::use_spell_check()

devtools::document()
devtools::install(dependencies = FALSE, build_vignettes = FALSE) # Note, should set this to TRUE every so often
devtools::install(dependencies = FALSE, build_vignettes = TRUE) # Note, should set this to TRUE every so often
devtools::build()
devtools::check()

# Build pkgdown site
# usethis::use_pkgdown()
pkgdown::build_site()

mldash::check_java()
mldash::check_python()

# Move these to check_python?
torch::torch_is_installed()
torch::install_torch()


usethis::use_package('R.utils', type = 'Imports')


# Test datasets and models from package source directory
ml_datasets <- mldash::read_ml_datasets(dir = 'inst/datasets')
ml_models <- mldash::read_ml_models(dir = 'inst/models')
ml_datasets <- ml_datasets |> dplyr::filter(id %in% c('titanic', 'PedalMe'))
ml_results <- mldash::run_models(datasets = ml_datasets, models = ml_models, timeout = Inf)





# Test core functions from package installation directory
ml_datasets <- mldash::read_ml_datasets()
ml_models <- mldash::read_ml_models()
ml_results <- mldash::run_models(datasets = ml_datasets, models = ml_models)


save_model_run <- function(results,
						   dir = 'inst/model_runs') {
	if(!'mldash_summary' %in% class(results)) {
		stop('results needs to be the output of mldash::run_models.')
	}
	d <- attr(ml_results, 'start_time')
	d <- gsub(' ', '_', d)
	si <- Sys.info()
	saveRDS(ml_results, file = paste0(dir, '/', unname(si['user']), '-', d, '.rds'))
}

save_model_run(ml_results)










# Examine error messages
names(attributes(ml_results))
ml_errors <- attr(ml_results, 'errors')
ml_errors



ml_datasets2 <- ml_datasets |> dplyr::filter(name == 'titanic')
ml_models2 <- ml_models |> dplyr::filter(name == 'logistic')
ml_results <- mldash::run_models(datasets = ml_datasets2,
								 models = ml_models2,
								 print_errors = FALSE,
								 seed = 1234)

class(ml_results)
ml_results_attributes <- attributes(ml_results)
names(ml_results_attributes)
errors <- attr(ml_results, 'errors')
names(errors)
errors[['titanic_tm_null_model_classification.dcf_accuracy']]
warnings <- attr(ml_results, 'warnings')
names(warnings)
attr(ml_results, 'session_info')

# TODO: Allow for sourcing of data files.

# Added package dependencies
usethis::use_package('tidyr', type = "Imports")
# usethis::use_package('RWeka', type = "Imports")
# usethis::use_package('shinyWidgets', type = "Imports")

ml_models <- mldash::read_ml_models(dir = 'inst/models')

# Add the packages defined in the model files to the Enhances field
pkgs <- ml_models[!is.na(ml_models$packages),]$packages |>
	strsplit(',') |>
	unlist() |>
	trimws() |>
	unique()
for(i in pkgs) {
	usethis::use_package(i, type = 'Enhances')
}

Sys.setenv("RETICULATE_PYTHON" = "~/miniforge3/envs/mldash/bin/python")

# Setup to be able to run the Python based models.
# https://medium.com/codex/installing-tensorflow-on-m1-macs-958767a7a4b3
remotes::install_github(sprintf("rstudio/%s", c("reticulate", "tensorflow", "keras", "torch")))
reticulate::miniconda_uninstall() # start with a blank slate
reticulate::install_miniconda()
reticulate::conda_create("mldash")
reticulate::use_condaenv("mldash")
reticulate::conda_install("mldash", c("jupyterlab", "pandas", "statsmodels",
									  "scipy", "scikit-learn", "matplotlib",
									  "seaborn", "numpy", "pytorch", "tensorflow"))

# keras::install_keras()
# torch::install_torch()
# tensorflow::install_tensorflow()

# Sys.setenv("RETICULATE_PYTHON" = "~/miniforge3/bin/python")

# tensorflow::use_python("~/miniforge3/bin/python")
tensorflow::use_condaenv("mldash")
keras::use_condaenv("mldash")

# Test TensorFlow
library(tensorflow)
hello <- tf$constant("Hello")
print(hello)

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

##### Model examples ###########################################################

# Model familes (from Fernández-Delgado, Cernadas, Barro, & Amorim 2014)
model_families <- c('discriminant analysis',
					'Bayesian',
					'neural networks',
					'support vector machines',
					'decision trees',
					'rule-based classifiers',
					'boosting',
					'bagging',
					'stacking',
					'random forests and other ensembles',
					'generalized linear models',
					'nearest-neighbors',
					'partial least squares and principal component regression',
					'logistic and multinomial regression',
					'multiple adaptive regression splines',
					'other methods')

# Tidymodels example
library(parsnip)
library(baguette)
fit <- bag_mars(num_terms = 7) |>
	set_mode("regression") |>
	set_engine("earth", times = 3) |>
	fit(formula, data = abalone)
parsnip::predict_raw.model_fit(fit, new_data = abalone)[,1,drop=TRUE]
predict.model_fit(fit, new_data = abalone)



##### Run models ###############################################################
model_pattern <- 'weka_*'          # Weka only models
model_pattern <- 'tm_logistic_*'   # Tidymodels only models
model_pattern <- 'tm_svm_rbf_*'
model_pattern <- '*.dcf'          # All models

ml_datasets <- mldash::read_ml_datasets()

# These assume working from development directory structure
ml_datasets <- mldash::read_ml_datasets(dir = 'inst/datasets',
										cache_dir = 'inst/datasets')

ml_datasets <- ml_datasets |> dplyr::filter(name == 'adult')

ml_models <- mldash::read_ml_models(dir = 'inst/models',
									pattern = model_pattern)

ml_models <- mldash::read_ml_models()

ml_results <- mldash::run_models(datasets = ml_datasets,
								 models = ml_models,
								 print_errors = FALSE,
								 seed = 1234)
# ml_results |> View()

ml_errors <- attr(ml_results, 'errors')
names(ml_errors)
ml_errors[[1]]

ml_warnings <- attr(ml_results, 'warnings')
names(ml_warnings)
ml_warnings[[1]]

si <- attr(ml_results, 'session_info')
ls(si)
attr(ml_results, 'start_time')
si$platform$os

save_model_run <- function(results,
						   dir = 'inst/model_runs') {
	if(!'mldash_summary' %in% class(results)) {
		stop('results needs to be the output of mldash::run_models.')
	}
	d <- attr(ml_results, 'start_time')
	d <- gsub(' ', '_', d)
	si <- Sys.info()
	saveRDS(ml_results, file = paste0(dir, '/', unname(si['user']), '-', d, '.rds'))
}

save_model_run(ml_results)

ml_results |>
	dplyr::select(!c(fit, error)) |>
	View()

# This is a long running model with the psych_copay dataset
model_name <- 'tm_svm_linear_kernlab_regression.dcf'
dataset_name <- 'psych_copay'
ml_datasets <- mldash::read_ml_datasets(dir = 'inst/datasets') |> dplyr::filter(id == dataset_name)
ml_models <- mldash::read_ml_models(dir = 'inst/models', pattern = model_name)
test_results <- mldash::run_models(ml_datasets, ml_models, print_errors = TRUE, timeout = 10)

# Run only classification models/datasets
ml_datasets <- ml_datasets %>% filter(type == 'classification')
ml_models <- ml_models %>% filter(type == 'classification')


(si <- sessioninfo::session_info())

##### OpenML Test ##############################################################
library(OpenML)
library(farff)

listOMLTaskTypes()
openml_classif <- listOMLTasks(task.type = 'Supervised Classification')

openml_data <- getOMLDataSet(data.id = 52)
openml_task <- getOMLTask(task.id = 52)


openml_datasets <- listOMLDataSets(number.of.instances = c(100000, 200000),
						   number.of.features = c(1, 5))

##### Hex Logo #################################################################
library(hexSticker)
p <- "man/figures/speed_icon.png"
hexSticker::sticker(p,
					filename = 'man/figures/mldash.png',
					p_size = 8,
					package = 'mldash',
					url = "github.com/mldash",
					u_size = 2.5,
					s_width = .75, s_height = .75,
					s_x = 1, s_y = 1,
					p_x = 1, p_y = .60,
					p_color = "#c51b8a",
					h_fill = '#fde0dd',
					h_color = '#c51b8a',
					u_color = '#fa9fb5',
					white_around_sticker = FALSE)

library(devtools)
library(usethis)

usethis::use_tidy_description()
usethis::use_spell_check()

devtools::document()
devtools::install(dependencies = FALSE) # Note, should set this to TRUE every so often
devtools::build()
devtools::check()


# Test core functions
ml_datasets <- mldash::read_ml_datasets()
ml_models <- mldash::read_ml_models()
ml_results <- mldash::run_models(datasets = ml_datasets,
								 models = ml_models,
								 print_errors = FALSE,
								 seed = 1234)

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

# Add the packages defined in the model files to the Suggests field
pkgs <- ml_models[!is.na(ml_models$packages),]$packages |>
	strsplit(',') |>
	unlist() |>
	trimws() |>
	unique()
for(i in pkgs) {
	usethis::use_package(i, type = 'Suggests')
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


##### Test code ################################################################
data <- get_all_vars(readRDS('inst/datasets/titanic.rds'))
formula <- survived ~  pclass + sex + age + sibsp + parch + fare + embarked
type <- 'classification'
y_var <- all.vars(formula)[1]
if(type == 'classification' & !is.factor(data[, y_var])) {
	data[, y_var] <- as.factor(data[, y_var])
}

# data <- model.matrix(formula, data) |> as.data.frame()
data <- cbind(
	y_var = data[,y_var],
	fastDummies::dummy_columns(data[,!names(data) %in% y_var],
							   remove_most_frequent_dummy = TRUE,
							   remove_selected_columns = TRUE) )
names(data)[1] <- y_var
formula <- as.formula(paste0(y_var, ' ~ ', paste0(names(data)[2:ncol(data)], collapse = ' + ')))
str(data)

parsnip::logistic_reg() |>
	parsnip::set_mode("classification") |>
	parsnip::set_engine("brulee") |>
	parsnip::fit(formula, data = data)

model <- parsnip::logistic_reg() |>
	parsnip::set_mode("classification") |>
	parsnip::set_engine("LiblineaR") |>
	parsnip::fit(formula, data = data)
ls(model)

model <- parsnip::svm_linear() |>
	parsnip::set_mode("classification") |>
	parsnip::set_engine("LiblineaR") |>
	parsnip::fit(formula, data = data)
predict.model_fit(model, new_data = data, type = "class")[,1,drop=TRUE]


train_fun <- function() {
	parsnip::rule_fit() |>
		parsnip::set_mode("classification") |>
		parsnip::set_engine("xrf") |>
		parsnip::fit(formula, data = data)
}

quiet_train_fun <- purrr::quietly(train_fun)
model <- quiet_train_fun()

predict.model_fit(model, new_data = data, type = "prob")[,1,drop=TRUE]

python_pkgs <- c('reticulate', 'keras', 'tensorflow', 'torch')

pkgs <- ml_models$packages |> strsplit(',')
pkgs <- lapply(pkgs, trimws)
avail_pkgs <- available.packages()
# deps <- tools::package_dependencies("brulee", db = avail_pkgs, which = 'strong', recursive = FALSE)
# any(unlist(deps) %in% python_pkgs)
deps <- lapply(pkgs, tools::package_dependencies, db = avail_pkgs, which = 'strong', recursive = TRUE)

test <- unlist(lapply(deps, FUN = function(x) { any(unlist(x) %in% python_pkgs) }))
names(test) <- ml_models$name
test[test]


data('adult', package = 'ucimlr')
adult <- adult |>
	dplyr::mutate(income_gt_50k = income == '>50K',
				  workclass = factor(workclass),
				  education = factor(education,
				  				   levels = c('Preschool', '1st-4th', '5th-6th', '7th-8th',
				  				   		   '9th', '10th', '11th', '12th',
				  				   		   'HS-grad', 'Prof-school', 'Some-college',
				  				   		   'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Doctorate'),
				  				   ordered = TRUE),
				  marital_status = factor(marital_status),
				  occupation = factor(occupation),
				  relationship = factor(relationship),
				  race = factor(race),
				  sex = factor(sex),
				  native_country = factor(native_country)) |>
	tidyr::drop_na()

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

# Run only classification models/datasets
ml_datasets <- ml_datasets %>% filter(type == 'classification')
ml_models <- ml_models %>% filter(type == 'classification')


(si <- sessioninfo::session_info())

helptext <- help('accuracy', package = 'yardstick')
helptext <- utils:::.getHelpFile(as.character(helptext))
tools:::Rd2txt(helptext)


db <- tools::Rd_db("yardstick")
# db <- db[names(db) == 'accuracy.Rd']
# db <- db[grep('accuracy.Rd', names(db))]
lapply(db, tools:::.Rd_get_metadata, "name")
lapply(db, tools:::.Rd_get_metadata, "description")

yardstick_base_url <- 'https://yardstick.tidymodels.org/reference/'
metrics <- mldash::get_all_metrics()
for(i in names(metrics)) {
	db_fun <- db[names(db) == paste0(i, '.Rd')]
	desc <- lapply(db_fun, tools:::.Rd_get_metadata, "description")
	gsub('\n', ' ', desc)
	cat(paste0('* [', i, '](', yardstick_base_url, i, '.html) - ',
			   desc, '\n'))
}



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

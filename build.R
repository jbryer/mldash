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


library(parsnip)
library(baguette)
fit <- bag_mars(num_terms = 7) |>
	set_mode("regression") |>
	set_engine("earth", times = 3) |>
	fit(formula, data = abalone)
parsnip::predict_raw.model_fit(fit, new_data = abalone)[,1,drop=TRUE]
predict.model_fit(fit, new_data = abalone)

##### Run models

ml_datasets <- mldash::read_ml_datasets(dir = 'inst/datasets',
										cache_dir = 'inst/datasets')

ml_models <- mldash::read_ml_models(dir = 'inst/models')
# ml_models |> dplyr::select(name, type, packages)

ml_results <- mldash::run_models(datasets = ml_datasets, models = ml_models)
ml_results |> View()

# Run only classification models/datasets
datasets <- ml_datasets %>% filter(type == 'classification')
models <- ml_models %>% filter(type == 'classification')


##### Look for metrics
library(yardstick)
metrics <- list()
pos <- which(search() == 'package:yardstick')
rd_yardstick <- tools::Rd_db('yardstick')
for(i in ls('package:yardstick')) {
	obj <- get(i, pos = pos)
	if(is.function(obj) & 'metric' %in% class(obj)) {
		# therd <- rd_yardstick[grep(paste0(i, ".Rd"), names(rd_yardstick), value = TRUE)]
		# title <- c(therd[[1]][[1]][[1]])
		metrics[[i]] <- obj
	}
}
names(metrics)


(si <- sessioninfo::session_info())

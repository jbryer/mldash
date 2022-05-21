#' Runs the predictive models for the given datasets.
#'
#' @param datasets the datasets to run the models with. Results from [mldash::read_ml_datasets()].
#' @param models the models to run. Results from [mldash::read_ml_models()].
#' @param seed random seed to set before randomly selecting the training dataset.
#' @param training_size the proportion of the data that should be used for training.
#'        The remaining percentage will be used for validation.
#' @param metrics list of model performance metrics from the `yardstick` package.
#'        See https://yardstick.tidymodels.org/articles/metric-types.html for more information.
#' @return a data.frame with the results of all the models run against all the datasets.
#' @export
#' @import yardstick
#' @importFrom sessioninfo session_info
run_models <- function(
		datasets,
		models,
		seed,
		training_size = 0.7,
		metrics = get_all_metrics()
		# metrics = list(
		# 	'r_squared' = yardstick::rsq,
		# 	'rmse' = yardstick::rmse,
		# 	'accuracy' = yardstick::accuracy,
		# 	'kappa' = yardstick::kap,
		# 	'sensitivity' = yardstick::sensitivity,
		# 	'specificity' = yardstick::specificity,
		# 	'roc_auc' = yardstick::roc_auc
		# )
) {
	start_time <- Sys.time()

	metric_types <- sapply(metrics, FUN = function(x) { class(x)[1] })
	numeric_metrics <- metrics[metric_types == 'numeric_metric']
	class_metrics <- metrics[metric_types == 'class_metric']
	class_probability_metrics <- metrics[metric_types == 'prob_metric']

	ml_summary <- data.frame(
		dataset = character(),
		model = character(),
		type = character(),
		base_accuracy = numeric(),
		cm = list(),
		time_user = numeric(),
		time_system = numeric(),
		time_elapsed = numeric(),
		stringsAsFactors = FALSE
	)
	ml_summary$cm <- list()

	for(i in c(names(class_metrics),
			   names(class_probability_metrics),
			   names(numeric_metrics) )) {
		ml_summary[,i] <- numeric()
	}

	cache_dir <- attr(datasets, 'cache_dir')

	for(d in seq_len(nrow(datasets))) {
		message(paste0('[', d, ' / ', nrow(datasets), '] Loading ', datasets[d,]$name, ' data...'))
		thedata <- readRDS(paste0(cache_dir, '/', datasets[d,]$name, '.rds'))
		formu <- as.formula(datasets[d,]$model)
		if(!missing(seed)) {
			set.seed(seed)
		}

		type <- datasets[d,]$type

		y_var <- all.vars(formu)[1]
		if(type == 'classification' & !is.factor(thedata[, y_var])) {
			thedata[, y_var] <- as.factor(thedata[, y_var])
		}

		training_rows <- sample(nrow(thedata), size = training_size * nrow(thedata))
		train_data <- thedata[training_rows,]
		valid_data <- thedata[-training_rows,]
		data_models <- models[models$type == type,]
		for(m in seq_len(nrow(data_models))) {
			message(paste0('   [', m, ' / ', nrow(data_models), '] Running ', data_models[m,]$name, ' model...'))
			modelname <- row.names(data_models)[m]
			tmp <- attr(data_models, 'functions')
			train_fun <- tmp[[modelname]]$train
			predict_fun <- tmp[[modelname]]$predict
			# results <- data.frame()
			tryCatch({
				if(!is.null(data_models[m,]$packages) &
				   !is.na(data_models[m,]$packages)) {
					pkgs <- trimws(unlist(strsplit(data_models[m,]$packages, ',')))
					for(i in seq_len(length(pkgs))) {
						suppressPackageStartupMessages(
							library(package = pkgs[i],
									character.only = TRUE,
									quietly = TRUE,
									verbose = FALSE)
						)
					}
				}

				# TODO: Save warnings an messages to results
				exec_time <- as.numeric(system.time({
					suppressWarnings({
						train <- train_fun(formu, train_data)
					})
				}))

				y_var <- all.vars(formu)[1]
				suppressWarnings({
					validate <- data.frame(
						estimate = predict_fun(train, valid_data),
						truth = valid_data[,y_var,drop=TRUE]
					)
				})

				results <- data.frame(
					dataset = datasets[d,]$name,
					model = modelname,
					type = type,
					base_accuracy = NA_real_,
					time_user = exec_time[1],
					time_system = exec_time[2],
					time_elapsed = exec_time[3],
					stringsAsFactors = FALSE
				)
				results$cm <- NA

				for(i in c(names(class_metrics),
						   names(class_probability_metrics),
						   names(numeric_metrics) )) {
					results[,i] <- NA_real_
				}

				if(type == 'classification') {
					results[1,]$base_accuracy <- max(prop.table(table(validate$truth)))
					validate$truth <- as.factor(validate$truth)
					for(i in names(class_probability_metrics)) {
						tryCatch({
							results[1,i] <- class_probability_metrics[[i]](validate,
																		   truth = truth,
																		   estimate = estimate)[1,3]
						}, error = function(e) {
							message(paste0('Error calculating ', i, ' metric.'))
						})
					}
					validate$estimate <- as.factor(validate$estimate > 0.5) # TODO: Allow for other break points
					cm <- table(validate$truth, validate$estimate)
					results[1,]$cm <- list(list(cm))
					for(i in names(class_metrics)) {
						tryCatch({
							results[1,i] <- class_metrics[[i]](validate,
															   truth = truth,
															   estimate = estimate)[1,3]
						}, error = function(e) {
							message(paste0('Error calculating ', i, ' metric.'))
						})
					}
				} else if(type == 'regression') {
					for(i in names(numeric_metrics)) {
						tryCatch({
							results[1,i] <- numeric_metrics[[i]](validate,
																 truth = truth,
																 estimate = estimate)[1,3]
						}, error = function(e) {
							message(paste0('Error calculating ', i, ' metric.'))
						})
					}
				} else {
					warning("Unknown predictive modeling type!")
				}

			},
			error = function(e) {
				message(paste0('   Error running ', modelname,' model'))
				print(e)
				results <- data.frame(
					dataset = datasets[d,]$name,
					model = modelname,
					type = type,
					base_accuracy = NA,
					time_user = NA,
					time_system = NA,
					time_elapsed = NA,
					stringsAsFactors = FALSE
				)
				for(i in c(names(class_metrics),
						   names(class_probability_metrics),
						   names(numeric_metrics) )) {
					results[,i] <- NA
				}
			})
			ml_summary <- rbind(ml_summary, results[,names(ml_summary)])
		}
	}

	attr(ml_summary, 'start_time') <- start_time
	attr(ml_summary, 'end_time') <- Sys.time()
	attr(ml_summary, 'session_info') <- sessioninfo::session_info()

	return(ml_summary)
}

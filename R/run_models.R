#' Runs the predictive models for the given datasets.
#'
#' @param datasets the datasets to run the models with. Results from [mldash::read_ml_datasets()].
#' @param models the models to run. Results from [mldash::read_ml_models()].
#' @param seed random seed to set before randomly selecting the training dataset.
#' @param training_size the proportion of the data that should be used for training.
#'        The remaining percentage will be used for validation.
#' @param print_errors if TRUE errors will be printed while the function runs.
#'        Errors will always be saved in the returned object.
#' @param metrics list of model performance metrics from the `yardstick` package.
#'        See https://yardstick.tidymodels.org/articles/metric-types.html for more information.
#' @return a data.frame with the results of all the models run against all the datasets.
#' @export
#' @import yardstick
#' @import parsnip
#' @import RWeka
#' @importFrom sessioninfo session_info
#' @importFrom purrr quietly
#' @importFrom reticulate py_capture_output
run_models <- function(
		datasets,
		models,
		seed = sample(1:2^15, 1),
		training_size = 0.7,
		save_trained_models = FALSE,
		print_errors = FALSE,
		metrics = mldash::get_all_metrics()
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

	model_errors <- list()
	model_warnings <- list()

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
	ml_summary$train_output <- character()
	ml_summary$train_warnings <- character()
	ml_summary$train_messages <- character()
	ml_summary$python_output <- character()

	for(i in c(names(class_metrics),
			   names(class_probability_metrics),
			   names(numeric_metrics) )) {
		ml_summary[,i] <- numeric()
	}

	cache_dir <- attr(datasets, 'cache_dir')
	trained_models <- list()

	for(d in seq_len(nrow(datasets))) {
		datasetname <- datasets[d,]$name
		message(paste0('[', d, ' / ', nrow(datasets), '] Loading ', datasetname, ' data...'))
		trained_models[[datasetname]] <- list()
		thedata <- readRDS(paste0(cache_dir, '/', datasetname, '.rds'))
		formu <- as.formula(datasets[d,]$model)
		if(!missing(seed)) {
			set.seed(seed)
		}

		type <- datasets[d,]$type

		y_var <- all.vars(formu)[1]
		if(type == 'classification' & !is.factor(thedata[, y_var,drop=TRUE])) {
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

			results <- ml_summary[0,]
			results[1,]$dataset <- datasetname
			results[1,]$model <- modelname
			results[1,]$type <- type

			tryCatch({
				if(!is.null(data_models[m,]$packages) &
				   !is.na(data_models[m,]$packages)) {
					pkgs <- trimws(unlist(strsplit(data_models[m,]$packages, ',')))
					for(i in seq_len(length(pkgs))) {
						suppressPackageStartupMessages(
							pkg_loaded <- require(package = pkgs[i],
												  character.only = TRUE,
												  quietly = TRUE)
						)
						if(!pkg_loaded) {
							warning(paste0(pkgs[i], ' package could not be loaded.'))
						}
					}
				}

				exec_time <- as.numeric(system.time({
					tryCatch({
						quiet_train_fun <- purrr::quietly(train_fun)
						py_output <- reticulate::py_capture_output({
							output <- quiet_train_fun(formu, train_data)
						})
						train <- output$result
						results[1,]$train_output <- ifelse(length(output$output) > 0,
														   paste0(output$output, collapse = '\n'),
														   NA)
						results[1,]$train_warnings <- ifelse(length(output$warnings) > 0,
															 paste0(output$output, collapse = '\n'),
															 NA)
						results[1,]$train_messages <- ifelse(length(output$messages) > 0,
															 paste0(output$messages, collapse = '\n'),
															 NA)
						results[1,]$python_output <- ifelse(length(py_output) > 0,
															paste0(py_output, collapse = '\n'),
															NA)
					})
				}))

				results[1,]$time_user = exec_time[1]
				results[1,]$time_system = exec_time[2]
				results[1,]$time_elapsed = exec_time[3]

				if(is.null(train)) {
					next;
				}

				if(save_trained_models) {
					trained_models[[datasetname]][[modelname]] <- train
				}

				y_var <- all.vars(formu)[1]
				suppressWarnings({
					validate <- data.frame(
						estimate = predict_fun(train, valid_data),
						truth = valid_data[,y_var,drop=TRUE]
					)
				})

				if(type == 'classification') {
					results[1,]$base_accuracy <- max(prop.table(table(validate$truth)))
					if(!is.factor(validate$truth)) {
						validate$truth <- factor(validate$truth)
					}
					for(i in names(class_probability_metrics)) {
						tryCatch({
							quiet_metric_fun <- purrr::quietly(class_probability_metrics[[i]])
							fun_out <- quiet_metric_fun(validate,
														truth = truth,
														estimate = estimate)
							if(!is.null(fun_out$result[1,3])) {
								results[1,i] <- fun_out$result[1,3]
							}
							if(length(fun_out$warnings) > 0) {
								model_warnings[[paste0(datasetname, '_', modelname, '_', i)]] <- fun_out$warnings
							}
							# results[1,i] <- class_probability_metrics[[i]](validate,
							# 											   truth = truth,
							# 											   estimate = estimate)[1,3]
						}, error = function(e) {
							model_errors[[paste0(datasetname, '_', modelname, '_', i)]] <<- e
							if(print_errors) {
								print(e)
							}
						})
					}
					if(is.numeric(validate$estimate)) {
						validate$estimate <- as.factor(validate$estimate > 0.5) # TODO: Allow for other break points
					}
					# if(length(levels(validate$estimate)) < length(levels(validate$truth))) {
					# 	levels(validate$estimate) <- levels(validate$truth)
					# }
					cm <- table(validate$truth, validate$estimate)
					results[1,]$cm <- list(list(cm))
					for(i in names(class_metrics)) {
						tryCatch({
							quiet_metric_fun <- purrr::quietly(class_metrics[[i]])
							fun_out <- quiet_metric_fun(validate,
														truth = truth,
														estimate = estimate)
							if(!is.null(fun_out$result[1,3])) {
								results[1,i] <- fun_out$result[1,3]
							}
							if(length(fun_out$warnings) > 0) {
								model_warnings[[paste0(datasetname, '_', modelname, '_', i)]] <- fun_out$warnings
							}
							# results[1,i] <- class_metrics[[i]](validate,
							# 								   truth = truth,
							# 								   estimate = estimate)[1,3]
						}, error = function(e) {
							model_errors[[paste0(datasetname, '_', modelname, '_', i)]] <<- e
							if(print_errors) {
								print(e)
							}
						})
					}
				} else if(type == 'regression') {
					for(i in names(numeric_metrics)) {
						tryCatch({
							quiet_metric_fun <- purrr::quietly(numeric_metrics[[i]])
							fun_out <- quiet_metric_fun(validate,
														truth = truth,
														estimate = estimate)
							if(!is.null(fun_out$result[1,3])) {
								results[1,i] <- fun_out$result[1,3]
							}
							if(length(fun_out$warnings) > 0) {
								model_warnings[[paste0(datasetname, '_', modelname, '_', i)]] <- fun_out$warnings
							}
							# results[1,i] <- numeric_metrics[[i]](validate,
							# 									 truth = truth,
							# 									 estimate = estimate)[1,3]
						}, error = function(e) {
							# message(paste0('Error calculating ', i, ' metric.'))
							model_errors[[paste0(datasetname, '_', modelname, '_', i)]] <<- e
							if(print_errors) {
								print(e)
							}
						})
					}
				} else {
					warning("Unknown predictive modeling type!")
				}

			},
			error = function(e) {
				message(paste0('   Error running ', modelname,' model'))
				model_errors[[paste0(datasetname, '_', modelname)]] <<- e
				if(print_errors) {
					print(e)
				}
			})
			ml_summary <- rbind(ml_summary, results[,names(ml_summary)])
		}
	}

	row.names(ml_summary) <- 1:nrow(ml_summary)

	attr(ml_summary, 'start_time') <- start_time
	attr(ml_summary, 'end_time') <- Sys.time()
	attr(ml_summary, 'seed') <- seed
	attr(ml_summary, 'training_size') <- training_size
	attr(ml_summary, 'models') <- ml_models
	attr(ml_summary, 'datasets') <- ml_datasets
	attr(ml_summary, 'metrics') <- metrics
	attr(ml_summary, 'session_info') <- sessioninfo::session_info()
	attr(ml_summary, 'errors') <- model_errors
	attr(ml_summary, 'warnings') <- model_warnings
	if(save_trained_models) {
		attr(ml_summary, 'trained_models') <- trained_models
	}

	class(ml_summary) <- c('mldash_summary', 'data.frame')

	return(ml_summary)
}

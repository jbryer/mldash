#' Returns all metrics from the `yardstick` package.
#'
#' This is a convenience function that will return all the metrics from the
#' `yardstick` package that can be used with the [run_models()] function.
#'
#' @seealso run_models
#' @seealso https://yardstick.tidymodels.org/articles/metric-types.html
#' @export
#' @return a list with all the metric functions from the `yardstick` package.
#' @import yardstick
get_all_metrics <- function(
		# include_classification = TRUE,
		# include_regression = TRUE,
		include_multiclass = FALSE
) {
	metrics <- list()
	library(yardstick)
	pos <- which(search() == 'package:yardstick')
	# rd_yardstick <- tools::Rd_db('yardstick')
	for(i in ls('package:yardstick')) {
		obj <- get(i, pos = pos)
		if(is.function(obj) & 'metric' %in% class(obj)) {
			# therd <- rd_yardstick[grep(paste0(i, ".Rd"), names(rd_yardstick), value = TRUE)]
			# title <- c(therd[[1]][[1]][[1]])
			metrics[[i]] <- obj
		}
	}
	# metric_types <- sapply(metrics, FUN = function(x) { class(x)[1] })
	# numeric_metrics <- metrics[metric_types == 'numeric_metric']
	# class_metrics <- metrics[metric_types == 'class_metric']
	# class_probability_metrics <- metrics[metric_types == 'prob_metric']
	# if(!include_classification) {
	#
	# }
	# if(!include_regression) {
	#
	# }
	if(!include_multiclass) {
		metrics <- metrics[!names(metrics) %in% c('roc_aunp', 'roc_aunu')] # Excluding multiclass metrics
	}
	invisible(metrics)
}

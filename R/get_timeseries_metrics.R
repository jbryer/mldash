#' Returns time series metrics from the `yardstick` package.
#'
#' This is a convenience function that will return time series  metrics from the
#' `yardstick` package that can be used with the [run_models()] function.
#'
#' @seealso run_models
#' @seealso https://yardstick.tidymodels.org/articles/metric-types.html
#' @export
#' @return a list with time series metric functions from the `yardstick` package.
#' @import yardstick
get_timeseries_metrics <- function(
		include_multiclass = FALSE
) {
	metrics <- list()
	library(yardstick)
	pos <- which(search() == 'package:yardstick')
	for(i in ls('package:yardstick')) {
		obj <- get(i, pos = pos)
		if(is.function(obj) & 'metric' %in% class(obj)) {
			# therd <- rd_yardstick[grep(paste0(i, ".Rd"), names(rd_yardstick), value = TRUE)]
			# title <- c(therd[[1]][[1]][[1]])
			metrics[[i]] <- obj
		}
	}

	metrics <- metrics[c('mae', 'mape', 'mase', 'mpe', 'rmse', 'rsq')]

	invisible(metrics)
}

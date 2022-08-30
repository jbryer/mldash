#' Run the Loess shiny app.
#'
#' This will start the shiny app to demonstrate the loess_vis function.
#'
#' @param span_range the range of values the user is allowed to set the span.
#' @param ... not currently used.
#' @import shiny
#' @export
shiny_mldash <- function(
	model_run_dir = paste0(find.package('mldash'), '/', model_runs),
	ml_results = run_models(ml_datasets, ml_models, seed = 2112),
	...
) {
	shiny_env <- new.env()
	assign('ml_datasets', ml_datasets, shiny_env)
	assign('ml_models', ml_models, shiny_env)
	assign('ml_results', ml_results, shiny_env)
	environment(shiny_ui) <- shiny_env
	environment(shiny_server) <- shiny_env
	app <- shiny::shinyApp(
		ui = mldash::shiny_ui,
		server = mldash::shiny_server
	)
	return(app)
}

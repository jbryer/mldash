library(shiny)
library(tidyverse)
library(DT)

LOCAL <- interactive()

if(LOCAL) {
	message('Running locally...')
	source('../../R/shiny_server.R')
	source('../../R/shiny_ui.R')
	model_run_dir <<- '../model_runs'
	model_runs <<- list.files(model_run_dir, pattern = '*.rds')
	i <- 0
	if(length(model_runs) > 1) {
		msg <- 'Select the mldash results to view:'
		i <- menu(choices = model_runs, title = msg)
	} else if(length(model_runs) == 1) {
		i <- 1
	}
	if(i == 0) {
		stop('Could not find an model run files.')
	}
	ml_results <<- readRDS(paste0(model_run_dir, '/', model_runs[i]))
	names(attributes(ml_results))
	ml_models <<- attr(ml_results, 'models')
	ml_datasets <<- attr(ml_results, 'datasets')
	# ml_results <<- mldash::run_models(datasets = ml_datasets, models = ml_models, seed = 2112)
} else {
	message('Running from mldash package...')
	library(mldash)
}

shiny::shinyApp(ui = shiny_ui, server = shiny_server)

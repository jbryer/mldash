library(shiny)
library(tidyverse)
library(DT)

LOCAL <- TRUE

if(LOCAL) {
	message('Running locally...')
	source('../../R/shiny_server.R')
	source('../../R/shiny_ui.R')
	model_run_dir <<- '../model_runs'
	model_runs <<- list.files(model_run_dir, pattern = '*.rds')
	# ml_results <<- mldash::run_models(datasets = ml_datasets, models = ml_models, seed = 2112)
} else {
	message('Running from mldash package...')
	library(mldash)
}

shiny::shinyApp(ui = shiny_ui, server = shiny_server)

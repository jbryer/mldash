#' The Shiny App Server.
#'
#' The server code for the mldash shiny app.
#'
#' @param input input set by Shiny.
#' @param output output set by Shiny.
#' @param session session set by Shiny.
#' @import shiny
#' @export
shiny_server <- function(input, output, session) {

	##### Model Run Summary ####################################################

	get_mlrun_summary <- reactive({
		req(input$model_run)
		mr <- readRDS(paste0(model_run_dir, '/', input$model_run))
		return(mr)
	})

	output$model_run_session_info <- renderPrint({
		mr <- get_mlrun_summary()
		attr(mr, 'session_info')
	})

	output$model_run_summary <- DT::renderDataTable({
		mr <- get_mlrun_summary()
		# TODO: Select metrics to include
		mr |>
			select('dataset', 'model', 'type', 'base_accuracy', 'time_elapsed', input$metrics) |>
			# filter(type %in% input$mr_type) |>
			# filter(dataset %in% input$mr_dataset) |>
			mutate(dataset = as.factor(dataset),
				   model = as.factor(model),
				   type = as.factor(type)) |>
			datatable(options = list(pageLength = 50),
					  rownames = FALSE,
					  select = 'single',
					  filter = 'top') |>
			formatRound(c('base_accuracy', 'time_elapsed', input$metrics), digits = 2)
	})

	output$mr_type <- renderUI({
		mr <- get_mlrun_summary()
		selectInput(inputId = 'mr_type',
					label = 'Model Type:',
					multiple = TRUE,
					choices = unique(mr$type),
					selected = unique(mr$type))
	})

	output$mr_dataset <- renderUI({
		mr <- get_mlrun_summary()
		selectInput(inputId = 'mr_dataset',
					label = 'Dataset:',
					multiple = TRUE,
					choices = unique(mr$dataset),
					selected = unique(mr$dataset))
	})

	output$mr_metrics <- renderUI({
		mr <- get_mlrun_summary()
		metrics <- attr(mr, 'metrics')
		selectInput('metrics',
					label = 'Metrics:',
					multiple = TRUE,
					choices = names(metrics),
					selected = c('accuracy', 'rsq')) # TODO: Update this default, just randomly picked one from each type
	})

	##### Datasets #############################################################

	get_dataset <- reactive({
		ml_datasets %>% filter(name == input$dataset)
	})

	output$dataset <- DT::renderDataTable({
		d <- get_dataset()
		cache_dir <- attr(ml_datasets, 'cache_dir')
		formu <- d[1,]$model
		get_all_vars(formu, readRDS(paste0(cache_dir, '/', d[1,]$name, '.rds')))
	})

	output$formula <- renderText({
		d <- get_dataset()
		d[1,]$model
	})

	##### Models ###############################################################

	get_model <- reactive({
		req(input$model)
		ml_models %>% filter(name == input$model)
	})

	output$test_data <- renderUI({
		req(input$model_type)
		datasets <- ml_datasets %>% filter(type == input$model_type)
		selectInput('test_dataset',
					label = 'Test with dataset:',
					choices = datasets$name)
	})

	output$model_train_output <- renderPrint({
		req(input$test_train)
		isolate(dataset <- input$test_dataset)
		isolate(model_train <- input$model_train)
		data <- readRDS(paste0(attr(ml_datasets, 'cache_dir'), '/', dataset, '.rds'))
		formula <- as.formula(ml_datasets[ml_datasets$name == dataset,]$model)

		if(!is.null(input$model_packages) &
		   !is.na(input$model_packages)) {
			pkgs <- trimws(unlist(strsplit(input$model_packages, ',')))
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

		eval(parse(text = paste0('train <- ', model_train)))
		train(formula, data)
	})

	output$model_predict_output <- renderPrint({
		req(input$test_predict)
	})

	# output$model_name <- renderUI({
	# 	m <- get_model()
	# 	textInput('model_name', 'Name:',
	# 			  value = m[1,]$name)
	# })

	output$model_type <- renderUI({
		m <- get_model()
		selectInput('model_type', 'Type:',
					choices = c('classification', 'regression'),
					selected = m[1,]$type)
	})

	output$model_description <- renderUI({
		m <- get_model()
		textAreaInput('model_description', 'Description:',
					  width = '100%', height = '75px',
					  value = ifelse(is.na(m[1,]$description), '', m[1,]$description))
	})

	output$model_note <- renderUI({
		m <- get_model()
		textAreaInput('model_note', 'Notes:',
					  width = '100%', height = '75px',
					  value = ifelse(is.na(m[1,]$notes), '', m[1,]$notes))
	})

	output$model_packages <- renderUI({
		m <- get_model()
		textInput('model_packages', 'Packages (comma separated):',
				  width = '100%',
				  value = ifelse(is.na(m[1,]$packages), '', m[1,]$packages))
	})

	output$model_train <- renderUI({
		f <- attr(ml_models, 'functions')[[paste0(input$model, '.dcf')]]
		textAreaInput('model_train', 'Training Function:', width = '100%', height = '200px',
					  value = paste0(deparse(f[['train']]), collapse = '\n'))
	})

	output$model_predict <- renderUI({
		f <- attr(ml_models, 'functions')[[paste0(input$model, '.dcf')]]
		textAreaInput('model_predict', 'Prediction Function:', width = '100%', height = '200px',
					  value = paste0(deparse(f[['predict']]), collapse = '\n'))
	})

}

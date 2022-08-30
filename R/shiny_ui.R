#' The Shiny App UI.
#'
#' The user interface for the Loess shiny app.
#'
#' @import shiny
#' @importFrom shinyWidgets chooseSliderSkin
#' @importFrom shinyBS bsPopover
#' @export
shiny_ui <- function() {
	navbarPage(
		id = 'navbarpage',
		title = "Machine Learning Dashboard",
		# shinyWidgets::chooseSliderSkin("Shiny", color="seagreen"),

		tabPanel(
			'Overview',
			sidebarLayout(
				sidebarPanel(
					width = 3,
					selectInput('model_run',
								'Model Run:',
								choices = model_runs),
					# uiOutput('mr_type'),
					# uiOutput('mr_dataset'),
					uiOutput('mr_metrics'),
					# selectInput('model_type',
					# 			'Model Type:',
					# 			choices = c('classification', 'regression')),
					# selectInput('dataset',
					# 			'Dataset:',
					# 			choices = ml_datasets$name),
					# selectInput('model',
					# 			'Model:',
					# 			choices = ml_models$name),
					helpText(" ")
				),

				mainPanel(
					width = 9,
					tabsetPanel(
						tabPanel(
							'Table',
							DT::dataTableOutput('model_run_summary')
						),
						tabPanel(
							'Session Info',
							verbatimTextOutput('model_run_session_info')
						)
					)
				)
			)
		),
		tabPanel(
			'Models',
			fluidRow(
				column(4, selectInput('model',
									  'Model:',
									  choices = ml_models$name)),
				column(4, uiOutput('model_type'))
			),
			fluidRow(
				column(6, uiOutput('model_description')),
				column(6, uiOutput('model_note'))
			),
			fluidRow(
				column(6, uiOutput('model_packages'))
			),
			fluidRow(
				column(3, uiOutput('test_data')),
				column(3, br(), actionButton('test_train', 'Test Training Function')),
				column(3, br(), actionButton('test_predict', 'Test Prediction Function'))
			),
			fluidRow(
				column(12, uiOutput('model_train'))
			),
			fluidRow(
				column(12, verbatimTextOutput('model_train_output'))
			),
			fluidRow(
				column(12, uiOutput('model_predict'))
			)
		),
		tabPanel(
			'Datasets',
			fluidRow(
				column(4,
					   selectInput('dataset',
					   			'Dataset:',
					   			choices = ml_datasets$name)
				),
				column(8, p(strong('Model formula: '), textOutput('formula')))
			),
			DT::dataTableOutput('dataset')
		)
	)
}

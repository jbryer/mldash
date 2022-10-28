# This script will help test new models in the inst/models directory.
# Change the model_name parameter to the new model name.
# This script assumes you are running locally with the source package.

model_name <- 'lm'

# We will use the titanic and mtcars datasets for classification and regression
# models, respectively.
dataset_names <- c('titanic', 'mtcars')

# Read in the datasets from the inst/datasets directory.
ml_datasets <- mldash::read_ml_datasets(dir = 'inst/datasets')
# Get only the datasets we wish to use (i.e. small datasets that run fast)
ml_datasets <- ml_datasets |> dplyr::filter(name %in% dataset_names)
# Get the new model we wish to test
ml_models <- mldash::read_ml_models(dir = 'inst/models') |>
	dplyr::filter(name == model_name)

# Run the model
ml_results <- mldash::run_models(datasets = ml_datasets,
								 models = ml_models,
								 print_errors = FALSE,
								 seed = 1234)

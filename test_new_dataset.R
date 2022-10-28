# This script will help test new datasets in the inst/datasets directory.
# Change the dataset_name parameter to the new dataset
# This script assumes you are running locally with the source package.
dataset_name <- c('titanic')

model_names <- c('lm', 'logistic')

# We will use the titanic and mtcars datasets for classification and regression
# models, respectively.

# Read in the datasets from the inst/datasets directory.
ml_datasets <- mldash::read_ml_datasets(dir = 'inst/datasets')
# Get only the datasets we wish to use (i.e. small datasets that run fast)
ml_datasets <- ml_datasets |> dplyr::filter(name %in% dataset_name)
# Get the new model we wish to test
ml_models <- mldash::read_ml_models(dir = 'inst/models') |>
	dplyr::filter(name %in% model_names)

# Run the model
ml_results <- mldash::run_models(datasets = ml_datasets,
								 models = ml_models,
								 print_errors = FALSE,
								 seed = 1234)

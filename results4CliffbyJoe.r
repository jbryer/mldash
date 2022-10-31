results <- function(){

	"Required Packages"

	library(mldash)
	library(tidyr)
	library(qtl2)
	library(archive)


	"Loading Model and Dataset to run"
	ml_models <- read_ml_models(dir = 'inst/models')

	ml_datasets <- read_ml_datasets(dir = 'inst/datasets',
											cache_dir = 'inst/datasets')


	"Specifying dataset and model type; target variable is binary (0,1), used simple logist. regression."
	audit <- ml_datasets[6,]

	logistic_reg <- ml_models[2,]


	"Running Logistic Regression model"
	logi_results <- run_models(datasets = audit, models = logistic_reg, seed = 1234)


	"Unlisting, converting to dataframe to save as .csv"
	logi_results <- unlist(logi_results)
	t <- data.frame(t(sapply(logi_results,c)))
	results <- write.csv(t,"MLDASH_logistic_regression_results_AUDITdf.csv")
}


# Save function as .rds
saveRDS(results, file = "results.rds")

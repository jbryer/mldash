#' Create a new data set metadata file.
#'
#' @param filename the name of the file that will be created. This will be the id for the model.
#' @param name the  name of the model.
#' @param dir the directory to store the model file.
#' @param type type of model, either regression (quantitative dependent variables),
#'        or classification (for qualitative dependent variables).
#' @param description brief description of the model.
#' @param train_fun function that will return a trained model. Must have two parameters:
#'        `formula` and `data`.
#' @param predict_fun function that will return the predicted probabilities for
#'        classification models and predicted Y-values for regression models.
#' @param packages comma separated list of any packages that need to be loaded
#'        prior to running either `train_fun` or `predict_fun`.
#' @param note any additional descriptions about the model.
#' @param open whether to open the file for editing after it has been created.
#' @param overwrite whether to overwrite the model file if it already exists.
#' @return a list with the metadata information.
#' @export
new_model <- function(
		filename,
		name,
		type,
		dir = './inst/models',
		description = "Description of the dataset",
		train_fun = 'function(formula, data) {\n\t# Run predictive model.\n\t}',
		predict_fun = 'function(model, newdata) {\n\t# Return predicted probabilities for classification or nemeric values for regression.\n\t}',
		packages = '',
		note = "",
		open = interactive(),
		overwrite = FALSE
) {
	if(missing(type)) {
		error("type is a required parameter.")
	}
	file <- paste0(dir, '/', filename, '.dcf')
	if(file.exists(file) & !overwrite) {
		stop('File already exists')
	}
	meta <- data.frame(
		name = name,
		type = type[1],
		description = description,
		train = ifelse(is.function(train_fun),
					   paste0(deparse(train_fun), collapse = '\n\t'),
					   train_fun),
		predict = ifelse(is.function(predict_fun),
						 paste0(deparse(predict_fun), collapse = '\n\t'),
						 predict_fun),
		packages = packages,
		note = note,
		stringsAsFactors = FALSE
	)
	write.dcf(meta, file, keep.white = c('train', 'predict'))
	if(open) {
		file.edit(file)
	}
	invisible(read_ml_models(dir = dir, pattern = paste0(filename, '.dcf')))
}

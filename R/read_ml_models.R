required_model_fields <- c('name', 'type', 'train', 'predict')

#' Reads data files for running predictive models.
#'
#' Information about datasets that can be used in the predictive models are stored
#' in Debian Control Files (dcf). This is the similar to the format used in
#' RMarkdown YAML (i.e. metadata).
#'
#' \itemize{
#'     \item{name*}{The name of the dataset.}
#'     \item{type*}{Whether this is for a regression or classification model.}
#'     \item{description}{Description of the dataset.}
#'     \item{train*}{The model formula used for the predictive model.}
#'     \item{predict*}{Any additional information.}
#' }
#' * denotes required fields.
#'
#'
#' @param dir directory containing the dcf files for the datasets.
#' @param pattern optional regular expression that is used when finding files
#'        to read in. It defaults to all dcf files in the \code{dir}, but could
#'        be a single filename to test a metadata file.
#' @param check_for_missing_packages if TRUE you will be prompted to install
#'        missing packages.
#' @return a data frame with the following fields:
#' \itemize{
#'     \item{name}{The name of the dataset.}
#'     \item{type}{Whether this is for a regression or classification model.}
#'     \item{description}{Description of the dataset.}
#'     \item{notes}{Any additional information.}
#' }
#' * denotes required fields.
#' @export
read_ml_models <- function(
		dir = paste0(find.package('mldash'), '/models'),
		pattern = "*.dcf",
		check_for_missing_packages = interactive()
) {
	modelfiles <- list.files(dir,
							 pattern = pattern,
							 include.dirs = FALSE)

	ml_models <- data.frame(
		name = tools::file_path_sans_ext(modelfiles),
		type = NA_character_,
		description = NA_character_,
		notes = NA_character_,
		packages = NA_character_,
		row.names = modelfiles,
		stringsAsFactors = FALSE
	)

	models <- list()

	for(i in modelfiles) {
		modelname <- tools::file_path_sans_ext(i)
		tmp <- as.data.frame(read.dcf(paste0(dir, '/', i)))

		if(!all(required_model_fields %in% names(tmp))) {
			warning(paste0('Not all required fields are found in ', i, '. Skipping.'))
			next;
		}

		thedata <- NULL
		tryCatch({
			eval(parse(text = paste0('train <- ', tmp$train[1])))
			eval(parse(text = paste0('predict <- ', tmp$predict[1])))
			models[[i]] <- list(train = train,
								predict = predict)
		}, error = function(e) {
			warning(paste0('Error getting model from ', modelname))
			print(e)
			next;
		})

		for(j in names(ml_models)) {
			if(j %in% names(tmp)) {
				ml_models[i,j] <- tmp[1,j]
			}
		}
	}

	pkgs <- ml_models[!is.na(ml_models$packages),]$packages |>
		strsplit(',') |>
		unlist() |>
		trimws() |>
		unique()

	not_installed <- pkgs[!pkgs %in% installed.packages()[,'Package']]
	if(length(not_installed) > 0) {
		if(check_for_missing_packages) {
			msg <- paste0('The following package',
						  ifelse(length(not_installed) > 1, 's are', 'is'),
						  ' not installed but required by the models: ',
						   paste0(not_installed, collapse = ', '),
						  '\nDo you want to install these packages?')
			ans <- menu(c('Yes', 'No'),
						title = msg)
			if(ans == 1) {
				install.packages(not_installed)
			}
		} else {
			warning(paste0('The following package',
						   ifelse(length(not_installed) > 1, 's are', 'is'),
						   ' not installed but required by the models: ',
						   paste0(not_installed, collapse = ', ')))
		}
	}

	ml_models$type <- tolower(ml_models$type)
	attr(ml_models, 'functions') <- models
	attr(ml_models, 'dir') <- normalizePath(dir)
	class(ml_models) <- c('mldash_models', 'data.frame')
	return(ml_models)
}

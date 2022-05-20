required_dataset_fields <- c('name', 'type', 'data', 'model')

#' Reads model files for running predictive models.
#'
#' Information about models that can be used in the predictive models are stored
#' in Debian Control Files (dcf). This is the similar to the format used in
#' RMarkdown YAML (i.e. metadata).
#'
#' \itemize{
#'     \item{name*}{The name of the dataset.}
#'     \item{type*}{Whether this is for a regression or classification model.}
#'     \item{description}{Description of the dataset.}
#'     \item{source}{The source of the dataset.}
#'     \item{reference}{Reference for the dataset (APA format preferred).}
#'     \item{url}{URL to download the dataset.}
#'     \item{data*}{An R function that returns a data.frame.}
#'     \item{model*}{The model formula used for the predictive model.}
#'     \item{note}{Any additional information.}
#' }
#' * denotes required fields.
#'
#'
#' @param dir directory containing the dcf files for the datasets.
#' @param data_cache directory where rds data files will be stored.
#' @param pattern optional regular expression that is used when finding files
#'        to read in. It defaults to all dcf files in the \code{dir}, but could
#'        be a single filename to test a metadata file.
#' @param use_cache whether to read data from the cache if available. If FALSE,
#'        then the data will be retrieved from the `data` function parameter.
#' @return a data frame with the following fields:
#' \itemize{
#'     \item{name*}{The name of the dataset.}
#'     \item{type*}{Whether this is for a regression or classification model.}
#'     \item{description}{Description of the dataset.}
#'     \item{source}{The source of the dataset.}
#'     \item{reference}{Reference for the dataset (APA format preferred).}
#'     \item{url}{URL to download the dataset.}
#'     \item{model*}{The model formula used for the predictive model.}
#'     \item{note}{Any additional information.}
#' }
#' * denotes required fields.
#' @export
read_ml_datasets <- function(
		dir = paste0(find.package('mldash'), '/datasets'),
		cache_dir = tempdir(),
		pattern = "*.dcf",
		use_cache = TRUE
) {
	if(!dir.exists(cache_dir)) {
		message(paste0('Creating ', cache_dir, ' directory...'))
		dir.create(cache_dir, recursive = TRUE)
	}

	datafiles <- list.files(dir, pattern = pattern)

	ml_datasets <- data.frame(
		name = tools::file_path_sans_ext(datafiles),
		type = NA_character_,
		description = NA_character_,
		source = NA_character_,
		reference = NA_character_,
		url = NA_character_,
		model = NA_character_,
		note = NA_character_,
		row.names = datafiles,
		stringsAsFactors = FALSE
	)

	for(i in datafiles) {
		dataname <- tools::file_path_sans_ext(i)
		tmp <- as.data.frame(read.dcf(paste0(dir, '/', i)))

		if(nrow(tmp) != 1) {
			warning(paste0('Error reading ', i, '. Skipping.'))
		}
		if(!all(required_dataset_fields %in% names(tmp))) {
			warning(paste0('Not all required fields are found in ', i, '. Skipping.'))
			next;
		}

		thedata <- NULL
		datafile <- paste0(cache_dir, '/', dataname, '.rds')
		if(file.exists(datafile) & use_cache) {
			message(paste0('Reading ', dataname, ' from cache.'))
			thedata <- readRDS(datafile)
		} else {
			tryCatch({
				eval(parse(text = paste0('getdata <- ', tmp[1,]$data)))
				thedata <- getdata()
				formu <- as.formula(tmp[1,]$model)
				if(!all(complete.cases(get_all_vars(formula = formu, data = thedata)))) {
					warning(paste0('Missing data found in ', dataname,
								   '. It is recommend that missing data be handled in the data file.'))
				}
				saveRDS(thedata, file = datafile)
				rm(getdata)
			}, error = function(e) {
				warning(paste0('Error getting data from ', dataname))
				print(e)
				next;
			})
		}

		for(j in names(ml_datasets)) {
			if(j %in% names(tmp)) {
				ml_datasets[i,j] <- tmp[1,j]
			}
		}
	}

	ml_datasets$type <- tolower(ml_datasets$type)

	attr(ml_datasets, 'cache_dir') <- normalizePath(cache_dir)
	class(ml_datasets) <- c('mldash_datasets', 'data.frame')
	return(ml_datasets)
}

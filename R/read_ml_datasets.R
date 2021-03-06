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
#' @importFrom tools file_path_sans_ext
read_ml_datasets <- function(
		dir = c(paste0(find.package('mldash'), '/datasets')),
		cache_dir = tempdir(),
		pattern = "*.dcf",
		use_cache = TRUE
) {
	for(i in cache_dir) {
		if(!dir.exists(i)) {
			message(paste0('Creating ', i, ' directory...'))
			dir.create(i, recursive = TRUE)
		}
	}

	if(length(dir) > 1 & length(cache_dir) == 1) {
		cache_dir <- rep(cache_dir, length(dir))
	}

	datafiles <- lapply(dir, list.files, pattern = pattern, full.names = TRUE)
	cache_dir <- rep(cache_dir, sapply(datafiles, length))
	datafiles <- unlist(datafiles)

	ml_datasets <- data.frame(
		name = basename(datafiles) |> tools::file_path_sans_ext(),
		file = datafiles,
		cache_file = paste0(cache_dir, '/', basename(datafiles) |> tools::file_path_sans_ext(), '.rds'),
		type = NA_character_,
		description = NA_character_,
		source = NA_character_,
		reference = NA_character_,
		url = NA_character_,
		model = NA_character_,
		note = NA_character_,
		nrow = NA_integer_,
		ncol = NA_integer_,
		stringsAsFactors = FALSE
	)

	for(i in seq_len(nrow(ml_datasets))) {
		tmp <- as.data.frame(read.dcf(ml_datasets[i,]$file))

		if(nrow(tmp) != 1) {
			warning(paste0('Error reading ', i, '. Skipping.'))
			next;
		}
		if(!all(required_dataset_fields %in% names(tmp))) {
			warning(paste0('Not all required fields are found in ', ml_datasets[i,]$file, '. Skipping.'))
			next;
		}

		thedata <- NULL
		datafile <- ml_datasets[i,]$cache_file

		file_info <- file.info(ml_datasets[i,]$file)
		cache_info <- file.info(datafile)

		if(file.exists(datafile) & use_cache & cache_info$mtime > file_info$mtime) {
			message(paste0('Reading ', ml_datasets[i,]$name, ' from cache.'))
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

		ml_datasets[i,]$nrow <- nrow(thedata)
		ml_datasets[i,]$ncol <- ncol(thedata)

		for(j in names(ml_datasets)) {
			if(j %in% names(tmp)) {
				ml_datasets[i,j] <- tmp[1,j]
			}
		}
	}

	dups <- duplicated(ml_datasets$name)
	if(sum(dups) > 0) {
		warning(paste0('Duplicate data files found. The following will be excluded: ',
					   paste0(ml_datasets[dups,]$file, collapse = ', ')))
		ml_datasets <- ml_datasets[!dups,]
	}

	ml_datasets$type <- tolower(ml_datasets$type)

	attr(ml_datasets, 'cache_dir') <- normalizePath(cache_dir)
	class(ml_datasets) <- c('mldash_datasets', 'data.frame')
	return(ml_datasets)
}

required_dataset_fields <- c('name', 'type', 'data', 'model')

#' Reads model files for running predictive models.
#'
#' Information about models that can be used in the predictive models are stored
#' in Debian Control Files (dcf). This is the similar to the format used in
#' RMarkdown YAML (i.e. metadata).
#'
#' \itemize{
#'     \item{name*}{The name of the dataset.}
#'     \item{type*}{Whether this is for a regression, classification, timeseries, or spatial model.}
#'     \item{description}{Description of the dataset.}
#'     \item{source}{The source of the dataset.}
#'     \item{reference}{Reference for the dataset (APA format preferred).}
#'     \item{data*}{An R function that returns a data.frame.}
#'     \item{model*}{The formula used for the predictive model.}
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
#' @param check_for_missing_packages if TRUE you will be prompted to install
#'        missing packages.
#' @return a data frame with the following fields:
#' \itemize{
#'     \item{id}{The filename of the dataset.}
#'     \item{title*}{The name of the dataset from the dcf file.}
#'     \item{type*}{Whether this is for a regression or classification model.}
#'     \item{description}{Description of the dataset.}
#'     \item{source}{The source of the dataset.}
#'     \item{reference}{Reference for the dataset (APA format preferred).}
#'     \item{model*}{The model formula used for the predictive model.}
#'     \item{note}{Any additional information.}
#' }
#' * denotes required fields.
#' @export
#' @importFrom tools file_path_sans_ext
read_ml_datasets <- function(
		dir = c(paste0(find.package('mldash'), '/datasets')),
		cache_dir = dir,
		pattern = "*.dcf",
		use_cache = TRUE,
		check_for_missing_packages = interactive()
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
		id = basename(datafiles) |> tools::file_path_sans_ext(),
		name = NA_character_,
		file = datafiles,
		cache_file = paste0(cache_dir, '/', basename(datafiles) |> tools::file_path_sans_ext(), '.rds'),
		type = NA_character_,
		description = NA_character_,
		packages = NA_character_,
		source = NA_character_,
		reference = NA_character_,
		model = NA_character_,
		note = NA_character_,
		nrow = NA_integer_,
		ncol = NA_integer_,
		model_params = NA_character_,
		stringsAsFactors = FALSE
	)

	for(i in seq_len(nrow(ml_datasets))) {
		tmp <- as.data.frame(read.dcf(ml_datasets[i,]$file))

		if(nrow(tmp) != 1) {
			warning(paste0('Error reading ', ml_datasets[i,]$file, '. Skipping.'))
			next;
		}
		if(!all(required_dataset_fields %in% names(tmp))) {
			warning(paste0('Not all required fields are found in ', ml_datasets[i,]$file, '. Skipping.'))
			next;
		}

		if('packages' %in% names(tmp)) {
			if(tmp['packages'] != '') {
				pkgs <- tmp[1,'packages'] |>
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
						ans <- utils::menu(c('Yes', 'No'), title = msg)
						if(ans == 1) {
							install.packages(not_installed)
						}
					} else {
						warning('The following package',
								ifelse(length(not_installed) > 1, 's are', 'is'),
								' not installed but required by the models: ',
								paste0(not_installed, collapse = ', '))
					}
				}
			}
		}

		thedata <- NULL
		datafile <- ml_datasets[i,]$cache_file

		file_info <- file.info(ml_datasets[i,]$file)
		cache_info <- file.info(datafile)

		if(file.exists(datafile) & use_cache & cache_info$mtime > file_info$mtime) {
			message(paste0('Reading ', ml_datasets[i,]$id, ' from cache...'))
			thedata <- readRDS(datafile)
		} else {
			message(paste0('Downloading ', ml_datasets[i,]$id, '...'))
			tryCatch({
				eval(parse(text = paste0('getdata <- ', tmp[1,]$data)))
				thedata <- getdata()
				formu <- as.formula(tmp[1,]$model)
				all_vars <- get_all_vars(formula = formu, data = thedata)
				if(!all(complete.cases(all_vars))) {
					missing <- paste(colnames(all_vars)[as.data.frame(which(is.na(all_vars), arr.ind=TRUE))$col %>% unique()], collapse = ", " )
					warning(paste0('Missing data found in ', ml_datasets[i,]$id,
                                                       ". It is recommend that missing data be handled in the data file.\n  Columns with missing data: ", missing))
				}
				if(use_cache) {
					saveRDS(thedata, file = datafile)
				}
				rm(getdata)
			}, error = function(e) {
				warning(paste0('Error getting data ', ml_datasets[i,]$id))
				print(e)
				next;
			})
		}

		ml_datasets[i,]$nrow <- nrow(thedata)
		ml_datasets[i,]$ncol <- ncol(thedata)

		# Add values to the data.frame if they exist in the dcf file
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

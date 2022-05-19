#' Reads in a data frame for a model file
#'
#' @param dir directory containing the dcf files for the datasets.
#' @param cache_dir directory where rds data files will be stored.
#' @param use_cache whether to read data from the cache if available. If FALSE,
#'        then the data will be retrieved from the `data` function parameter.
#' @return the data.frame of the data.
#' @export
get_data <- function(
		dataname,
		dir = paste0(find.package('mldash'), '/datasets'),
		cache_dir = tempdir(),
		use_cache = TRUE
) {
	if(!dir.exists(cache_dir)) {
		message(paste0('Creating ', cache_dir, ' directory...'))
		dir.create(cache_dir, recursive = TRUE)
	}

	datafile <- paste0(cache_dir, '/', dataname, '.rds')
	thedata <- NULL
	if(file.exists(datafile) & use_cache) {
		message(paste0('Reading ', dataname, ' from cache.'))
		thedata <- readRDS(datafile)
	} else {
		tryCatch({
			tmp <- as.data.frame(read.dcf(paste0(dir, '/', dataname, '.dcf')))
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

	# TODO: add model and other fields as attributes
	return(thedata)
}

#' Create a new data set metadata file.
#'
#' @return a list with the metadata information.
#' @export
new_dataset <- function(
		name,
		dir = './inst/datasets',
		type = c('classification', 'regression')[1],
		description = "Description of the dataset",
		source = "The source of the dataset.",
		reference = 'APA reference for the dataset.',
		data = "function() {\n\t# This function should return a data.frame.\n\t}",
		model = 'Y ~ X',
		note = '',
		open = interactive(),
		overwrite = FALSE
) {
	file <- paste0(dir, '/', name, '.dcf')
	if(file.exists(file) & !overwrite) {
		stop('File already exists')
	}
	meta <- data.frame(
		name = name,
		type = type,
		description = description,
		source = source,
		reference = reference,
		data = ifelse(is.function(data),
					  paste0(deparse(data), collapse = '\n\t'),
					  data),
		model = Reduce(paste, deparse(model)),
		note = note,
		stringsAsFactors = FALSE
	)
	write.dcf(meta, file, keep.white = c('data'), append = FALSE)
	if(open) {
		file.edit(file)
	}
	invisible(read_ml_datasets(dir = dir, pattern = paste0(name, '.dcf')))
}

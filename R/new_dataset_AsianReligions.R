#' Create a new data set metadata file.
#'
#' @return a list with the metadata information.
#' @export
new_dataset <- function(
		name = "AsianReligions",
		dir = './inst/datasets',
		type = c('classification', 'clustering')[1],
		description = "Most of the sacred texts provided to this dataset are from Project Gutenberg. The raw texts along with
		the authors' pre-precessed Document Term Matricies are provided. 'The attributes are the words from the bag of words pre-processed
		from the mini-corpus composed of the 8 religious books considered in this study. 8,265 words are used.'",
		source = "The source of the dataset.",
		reference = 'Sah, Fokoue, 2019',
		url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00512/AsianReligionsData.zip",
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
		url = url,
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

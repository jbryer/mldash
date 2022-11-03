#' Checks if Python is installed.
#'
#' This function checks to see if the RETICULATE_PYTHON environment variable is set. If
#' not it will check to see if python is installed on some platforms.
#'
#' @return TRUE if RETICULATE_PYTHON is set.
#' @export
check_python <- function() {
	if(Sys.getenv("RETICULATE_PYTHON") == "") {
		if(Sys.info()['sysname'] == 'Darwin') {
		} else if(Sys.info()['sysname'] == 'Windows') {
			# TODO: Don't have access to a Windows computer
		} else if(Sys.info()['sysname'] == 'Linux') {
		}
	}
	return(Sys.getenv("RETICULATE_PYTHON") != "")
}

#' Checks if Java is installed.
#'
#' This function checks to see if the JAVA_HOME environment variable is set. If
#' not it will check to see if Java is installed on some platforms.
#'
#' To install Java on Mac (assuming Homebrew is installed):
#' ```
#' brew tap adoptopenjdk/openjdk
#' brew install --cask adoptopenjdk8
#' ```
#'
#' To install Java on Ubuntu run:
#' ```
#' sudo apt-get install openjdk-8-jdk
#' ```
#'
#' @return TRUE if JAVA_HOME is set.
#' @export
check_java <- function() {
	if(Sys.getenv("JAVA_HOME") == "") {
		if(Sys.info()['sysname'] == 'Darwin') {
			jvms <- list.dirs('/Library/Java/JavaVirtualMachines/', recursive = FALSE)
			java_home <- ''
			if(length(jvms) == 1) {
				java_home <- paste0(jvms[1], '/Contents/Home/')
			} else if(length(jvms) > 1) {
				jvms_names <- basename(jvms)
				msg <- paste0('Multiple Java installations found. Select one:')
				ans <- menu(jvms_names, title = msg)
				java_home <- paste0(jvms[ans], '/Contents/Home/')
			}
			message(paste0('Setting JAVA_HOME=', java_home))
			Sys.setenv(JAVA_HOME = java_home)
		} else if(Sys.info()['sysname'] == 'Windows') {
			# TODO: Don't have access to a Windows computer
		} else if(Sys.info()['sysname'] == 'Linux') {
			jvms <- list.dirs('/usr/lib/jvm', recursive = FALSE)
			java_home <- ''
			if(length(jvms) == 1) {
				java_home <- jvms[1]
			} else if(length(jvms) > 1) {
				jvms_names <- basename(jvms)
				msg <- paste0('Multiple Java installations found. Select one:')
				ans <- menu(jvms_names, title = msg)
				java_home <- jvms[ans]
			}
			message(paste0('Setting JAVA_HOME=', java_home))
			Sys.setenv(JAVA_HOME = java_home)
		}
	}
	return(Sys.getenv("JAVA_HOME") != "")
}

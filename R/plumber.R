# plumber.R

# Listener that accepts rds files

# Add a rds and multipart form parser
#' @parser rds
#' @parser multi
#' @post /upload_rds
function(req, res, rds_file) {
    temp_file <- tempfile()
    saveRDS(rds_file, temp_file)
    ml_results <- readRDS(temp_file)

    # scan the file with clamav
    ## t1 <- system(paste0("clamdscan --no-summary --infected --quiet ", save_filename), intern = FALSE)
    ## if (t1 == 1) {
    ##     print ("infected")
    ##     file.remove(save_filename)
    ## } else {
    ##     print ("file is clean")
    ## }

    mldash_repo <- local_clone("https://github.com/cliftonleesps/mldash.git", verbose=TRUE)
    gh_results <- add_ml_results(ml_results, mldash_repo)


    TRUE # placeholder return value
}

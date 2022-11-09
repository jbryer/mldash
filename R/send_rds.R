library(httr)
library(magrittr)

send_ml_results <- function(ml_results, url = "http://0.0.0.0:7000/upload_rds") {
    temp_file <- tempfile()
    saveRDS(ml_results, temp_file)

    res <-
        POST(
            url,
            body = list(
                rds_file = upload_file(temp_file, "application/rds")
            )
        ) %>%
        content()
    return (str(res))
}

#' Clones the upstream mldash project, saves a ml_results object and creates a pull request
#' so others can see past results of run_ml_models().

upstream_owner <- "jbryer"
upstream_repo <- "mldash"
upstream_branch <- "master"

## upstream_url <- "https://github.com/cliftonleesps/tmp.git"
## upstream_owner <- "cliftonleesps"
## upstream_repo <- "tmp"
## upstream_branch <- "main"


branch_name <- "ml_results"

#' local_clone
#'
#' Clone the mldash project to a temp directory
#'
#' @param repo_upstream the  name of the model.
#' @param verbose show debug output of the cloning process
#' @export
local_clone <- function(
                        repo_upstream = NULL,
                        verbose = FALSE
                        ) {

    if (is.null(repo_upstream)) {
        print("repo_upstream is required; this should be a Github fork of https://github.com/jbryer/mldash.git.")
        return (NULL)
    }

    path_repo_1 <- tempfile(pattern="git2r-")

    screen_debug(paste0("Temp checkout dir ",path_repo_1), verbose)

    dir.create(path_repo_1)
    repo <- clone(repo_upstream, path_repo_1, progress=verbose)

    if (Sys.getenv("GITHUB_PAT") == "") {
        gca <- git_credential_ask('https://github.com')
        Sys.setenv(GITHUB_PAT = gca$password)
    }


    if (is.null(config(repo)$local$user.name)) {
        if (Sys.getenv("LOCAL_GIT_USERNAME") != "") {
            config(repo, user.name=Sys.getenv("LOCAL_GIT_USERNAME"))
        } else {
            print("Git repository is missing user.name. Please enter your Git username:")
            var = readline()
            Sys.setenv(LOCAL_GIT_USERNAME=var)
            config(repo, user.name=var)
        }
    }
    if (is.null(config(repo)$local$user.email)) {
        if (Sys.getenv("LOCAL_GIT_EMAIL") != "") {
            config(repo, user.email=Sys.getenv("LOCAL_GIT_EMAIL"))
        } else {
            print("Git repository is missing user.email. Please enter your Git email address:")
            var = readline()
            Sys.setenv(LOCAL_GIT_EMAIL=var)
            config(repo, user.email=var)
        }
    }

    return (repo)
}


#' add_ml_results
#'
#' Saves a ml_results object to a RDS file and creates a pull request to a
#' forked mldash project
#'
#' @param ml_results dataframe returned by the ml_dash::run_ml_models() function
#' @param repo git_repository object returned by local_clone()
#' @param prefix when creating RDS files, the optional prefix will be used; otherwise
#'        it will be set to the github username
#' @return results of the pull request action
#' @export
add_ml_results <- function( ml_results = NULL, repo = NULL, prefix = NULL ) {
    if (is.null(ml_results) | is.null(repo)) {
        warning("ml_results and repo are required")
        return(NULL)
    }

    if (is.null(prefix)) {
        if (Sys.getenv("LOCAL_GIT_USERNAME") != "") {
            prefix <- Sys.getenv("LOCAL_GIT_USERNAME")
        } else if ( !is.null(default_signature(repo)$name) ) {
            prefix <- default_signature(repo)$name
        } else {
            prefix <- 'ml_results'
        }
    }

    ml_results_filename <- paste0(tempfile(pattern=paste0(prefix,'-'), tmpdir=paste0(workdir(repo),'/inst/results')),'.rds')
    saveRDS(ml_results, ml_results_filename)

    checkout(object = repo, create=TRUE, force=TRUE, branch = branch_name)
    add(repo, ml_results_filename)
    commit_result <- commit(repo, paste0("Adding new ml_results file: ", ml_results_filename))
    push(repo, "origin", paste0("refs/heads/",branch_name), credentials=cred_token())

    # delete any pre-existing pull requests
    gh_result <- gh("GET /repos/{owner}/{repo}/pulls",
                    owner = upstream_owner,     # upstream owner
                    repo = upstream_repo,                # upstream repo
                    state = "open"                 # upstream branch
                    )

    for (i in gh_result) {
        if (i$user$login == Sys.getenv("LOCAL_GIT_USERNAME")) {
            gh_result <- gh(paste0("PATCH /repos/{owner}/{repo}/pulls/", i$number),
                            owner = upstream_owner,     # upstream owner
                            repo = upstream_repo,                # upstream repo
                            state = "closed"                 # upstream branch
                            )
            print(paste0("Closed previous pull request at: ", i$url))
        }
    }


    # now make a pr request
    gh_result <- gh("POST /repos/{owner}/{repo}/pulls",
       owner = upstream_owner,     # upstream owner
       repo = upstream_repo,                # upstream repo
       base=upstream_branch,                 # upstream branch

       head_repo=upstream_repo,             # source repo
       head= paste0(default_signature(repo)$name, ":", branch_name),     # source owner:branch

       body=paste0("Adding ml_result set: ", basename(ml_results_filename)),
       title=paste0("New ml_result: ", basename(ml_results_filename))
       )

    print(paste0("Created a pull request at: ", gh_result$html_url))
    return(gh_result)
}

# internal function just used for showing debug/troubleshooting information
screen_debug <- function( message ='', verbose = FALSE) {
    if (verbose) {
        print(message)
    }
}

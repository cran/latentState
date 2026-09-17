#' Build a fixed, non-randomized trial world for use with \code{run()}
#'
#' Constructs the \code{c_vec} (cue array) and \code{win} (reward array)
#' needed for the \code{my_world} argument of \code{run()}, letting you
#' specify an exact trial-by-trial cue and reward sequence instead of
#' having one randomly generated from \code{phase_def}.
#'
#' @param trial_patterns Character vector, one element per trial, giving
#'   the cues shown on each arm. Use letters for cues, \code{"_"} to
#'   separate arms, and \code{"0"} for an empty (unselectable) arm -- the
#'   same naming convention used for trial-type names in \code{phase_def}
#'   (e.g. \code{"A"}, \code{"AX_B"}, \code{"A_0"}). Every element must
#'   have the same number of arms.
#'
#' @param reward Numeric vector, or list of numeric vectors. Gives the
#'   reward value for each trial, applied identically to every arm on that
#'   trial regardless of which arm is later chosen by the agent (unlike
#'   \code{phase_def}, where reward probabilities can differ by cue/arm).
#'   A plain numeric vector is treated as a single reward dimension
#'   (\code{D = 1}); to specify multiple reward dimensions, supply a list
#'   of numeric vectors, one per dimension. Every vector must have length
#'   equal to \code{length(trial_patterns)}.
#'
#' @param L Integer. The total number of latent states the resulting
#'   \code{c_vec} is built for. Must match the total number of latent
#'   states used in the \code{run()} call this world is passed into (i.e.
#'   \code{ncop}), since \code{run()} validates \code{my_world}'s
#'   dimensions against that value. Defaults to \code{10}.
#'
#' @param features Character vector, or \code{NULL}. Lets you restrict
#'   which features (cues) are tracked, instead of the default of every
#'   individual letter plus every full multi-letter arm segment appearing
#'   in \code{trial_patterns}. Each entry must be composed of letters that
#'   co-occur together in at least one arm segment, and every arm segment
#'   in \code{trial_patterns} must be covered by at least one feature. This mirrors the \code{features} argument
#'   of \code{run()}, but validated against \code{trial_patterns} directly.
#'   Defaults to \code{NULL} (features generated automatically).
#'
#' @return A list with three elements, matching the format \code{run()}
#'   expects for \code{my_world}:
#'   \describe{
#'     \item{\code{c_vec}}{Array (\code{D x C x L x A x ntrials}) of 0/1
#'       cue activations, indicating which cues are active on each arm and
#'       trial, identical across every latent state.}
#'     \item{\code{win}}{Array (\code{D x A x ntrials}) of reward values,
#'       with the same value broadcast across every arm on a given trial.}
#'     \item{\code{type_seq}}{A one-element list wrapping
#'       \code{trial_patterns}, matching the phase-structured format
#'       \code{run()} produces internally.}
#'   }
#'
#' @examples
#' my_world <- make_world(
#'   trial_patterns = c("A", "A", "B", "A", "B"),
#'   reward         = c(1, 0, 1, 1, 0),
#'   L = 10
#' )
#' phase_def <- list(phase1 = list(trials = list(A = 3, B = 2)))
#' result <- run(phase_def, n = 1, my_world = my_world)
#' @export
make_world <- function(trial_patterns, reward, L = 10, features = NULL) {

  if (!is.character(trial_patterns)) stop(
    "trial_patterns must be a character vector of cue patterns, for example c(\"A\", \"B\", \"A\") or c(\"AX_B\", \"B_C\")."
  )
  if (any(grepl("[^A-Za-z_0]", trial_patterns))) stop(
    "trial_patterns may only contain letters, underscores, and 0. ",
    "Examples include \"A\", \"AX_B\", \"A_0\", and \"A_BC_X\"."
  )
  if (any(grepl("^_|_$|__", trial_patterns))) stop(
    "trial_patterns contains invalid underscore placement. ",
    "Underscores must separate arms and cannot appear at the start, end, or consecutively."
  )

  ntrials <- length(trial_patterns)

  # Determine number of arms from first trial
  A <- length(strsplit(trial_patterns[1], "_")[[1]])

  # Validate all trials have same number of arms
  arm_counts <- sapply(trial_patterns, function(tp) length(strsplit(tp, "_")[[1]]))
  if (any(arm_counts != A)) stop(paste0(
    "All trial patterns must have the same number of arms (", A, "), ",
    "but trials ", paste(which(arm_counts != A), collapse = ", "),
    " have ", arm_counts[arm_counts != A], " arm(s)."
  ))

  # Determine C from unique letters across all patterns. [_0] removes both _ and 0
  # Determine all unique arm segments (non-nil) across all trial patterns
  all_arm_segs <- unique(Filter(function(s) s != "0",
                                unlist(lapply(trial_patterns, function(tp) strsplit(tp, "_")[[1]]))
  ))

  # Valid atomic letters that appear in at least one arm segment
  valid_atoms <- sort(unique(unlist(strsplit(all_arm_segs, ""))))

  if (!is.null(features)) {

    if (!is.character(features)) stop(
      "features must be NULL or a character vector."
    )
    if (length(features) == 0L) stop(
      "features must be NULL or a non-empty character vector."
    )
    if (anyDuplicated(features)) {
      warning("features contains duplicate entries; duplicates will be ignored.")
      features <- unique(features)
    }

    # Validate each feature: all its letters must exist, and its letters must
    # be a subset of at least one arm segment's letters
    invalid_features <- character(0)
    for (f in features) {
      f_letters <- strsplit(f, "")[[1]]
      if (!all(f_letters %in% valid_atoms)) {
        invalid_features <- c(invalid_features, f)
        next
      }
      appears <- any(sapply(all_arm_segs, function(seg) {
        seg_letters <- strsplit(seg, "")[[1]]
        all(f_letters %in% seg_letters)
      }))
      if (!appears) invalid_features <- c(invalid_features, f)
    }
    if (length(invalid_features) > 0L) stop(
      "features contains entries whose letters never co-occur in any arm segment: ",
      paste(invalid_features, collapse = ", "), "."
    )

    # Validate coverage: every arm segment must have at least one feature active
    for (seg in all_arm_segs) {
      seg_letters <- strsplit(seg, "")[[1]]
      covered <- any(sapply(features, function(f) {
        f_letters <- strsplit(f, "")[[1]]
        all(f_letters %in% seg_letters)
      }))
      if (!covered) stop(
        "features does not cover arm segment '", seg, "'. ",
        "At least one feature whose letters are a subset of {",
        paste(seg_letters, collapse = ", "), "} must be included."
      )
    }

    base_cues <- features

  } else {

    # Default: individual letters + full multi-letter arm segments
    config_cues <- sort(setdiff(all_arm_segs, valid_atoms))
    base_cues   <- c(valid_atoms, config_cues)

  }

  C <- length(base_cues)

  # Normalize reward: allow plain vector (D=1) or list of vectors
  if (is.numeric(reward)) {
    reward <- list(reward)
  }
  if (!is.list(reward)) stop(
    "reward must be a numeric vector (for D=1) or a list of numeric vectors (one per dimension)."
  )

  # Infer D from number of reward vectors
  D <- length(reward)

  # Validate each reward vector has length ntrials
  for (d in 1:D) {
    if (!is.numeric(reward[[d]])) stop(paste0(
      "reward[[", d, "]] must be a numeric vector."
    ))
    if (length(reward[[d]]) != ntrials) stop(paste0(
      "reward[[", d, "]] must have length equal to ntrials (", ntrials, "), ",
      "but has length ", length(reward[[d]]), "."
    ))
  }

  # Build c_vec array
  c_vec <- array(0, dim = c(D, C, L, A, ntrials))

  for (t in 1:ntrials) {
    arms <- strsplit(trial_patterns[t], "_")[[1]]
    for (a in 1:A) {
      # leave c_vec of nil arms as all zeros
      if (arms[a] == "0") next
      seg_letters <- strsplit(arms[a], "")[[1]]
      # A feature is active on this arm if all its letters are present in the segment
      cue_idx <- which(sapply(base_cues, function(f) {
        f_letters <- strsplit(f, "")[[1]]
        all(f_letters %in% seg_letters)
      }))
      for (d in 1:D) {
        for (l in 1:L) {
          c_vec[d, cue_idx, l, a, t] <- 1
        }
      }
    }
  }

  # Build win array: D x A x ntrials
  win <- array(0, dim = c(D, A, ntrials))
  for (d in 1:D) {
    for (t in 1:ntrials) {
      win[d, , t] <- reward[[d]][t]
    }
  }

  return(list(
    c_vec = c_vec,
    win = win,
    type_seq  = list(trial_patterns) #wrapped in list to match TaskInit format
  ))
}

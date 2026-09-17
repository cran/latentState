#' Simulate latent-state reinforcement learning across participants
#'
#' Simulates outcomes of an updated latent-state reinforcement learning model
#' (based on Cochran & Cisler, 2019) across one or more participants, under a
#' user-specified experimental design. For full mathematical details of the underlying latent-state learning
#' model, including changes from the original Cochran & Cisler model, see the
#' supplementary materials on OSF: \url{https://osf.io/2whcu}
#'
#' @param phase_def A named list defining the task structure. Each element
#'   represents one phase of the experiment and must contain two entries,
#'   \code{trials} and (optionally) \code{rewards}.
#'
#'   \code{trials} is a named list where each entry is a cue label (e.g.
#'   \code{"A"}, \code{"AB"}) and its value is the number of trials of that
#'   type in the phase. Each letter represents a unique cue, and combining
#'   multiple letters (e.g. \code{"AB"}) means those cues are shown
#'   simultaneously on the same arm.
#'
#'   \code{rewards} is a named list with one entry per reward dimension
#'   (e.g. \code{"D1"}, \code{"D2"}). Each entry is a named numeric vector
#'   mapping cue labels to reward probabilities between 0 and 1. Only cues
#'   that appear in \code{trials} may be listed in \code{rewards}. By default,
#'   probabilities are applied deterministically: exactly that
#'   proportion of trials of that type are rewarded, rather than each trial
#'   being rewarded independently at random. This requires each probability
#'   to divide evenly into the trial count. E.g. 0.75 requires a trial
#'   count divisible by 4. To reward trials independently at random
#'   instead, without this divisibility requirement, set
#'   \code{probabilistic = TRUE}.
#'
#'   A simple example using a single reward dimension and one option per
#'   trial is shown below. In phase 1, cue \code{A} appears alone on 6 trials and together
#'   with cue \code{B} on another 6 trials; \code{A} is rewarded on every
#'   trial, while \code{B} is omitted from \code{rewards} and is therefore
#'   automatically assigned a reward probability of 0. In phase 2, cue
#'   \code{A} appears alone on 8 trials and together with \code{B} on 16
#'   trials; since \code{rewards} is omitted entirely for this phase, all
#'   trials are automatically unrewarded.
#'   \preformatted{
#'   phase_def <- list(
#'     phase1 = list(
#'       trials  = list(A = 6, AB = 6),
#'       rewards = list(c(A = 1))
#'     ),
#'     phase2 = list(
#'       trials = list(A = 8, AB = 16)
#'     )
#'   )
#'   }
#'
#'   In experiments with multiple arms (choices), indicate the cues shown on
#'   each arm within a trial type name, separating arms with \code{_}. For
#'   example, a three-arm trial where one arm shows cues \code{A} and
#'   \code{B}, one shows only \code{A}, and one shows only \code{C} would be
#'   named \code{"AB_A_C"}. All trial types within a phase must have the
#'   same number of arms. If you want a trial type to have fewer effective
#'   arms, mark an arm as empty using \code{"0"}; that arm is assigned a
#'   value of \code{-Inf} and can never be chosen. This can be used to force
#'   participants to select a specific option on certain trials -- for
#'   example, \code{"A_0_0"} means only the arm with cue \code{A} can be
#'   chosen on that trial.
#'
#'   See \strong{Examples} below for a \code{phase_def} that uses multiple
#'   arms per trial, empty (\code{"0"}) arms, and multiple reward
#'   dimensions.
#'
#' @param n Integer. The number of participants to simulate. Each participant
#'   is run independently through the full task defined by \code{phase_def},
#'   with parameters either fixed or freshly sampled (if specified as a
#'   distribution) for each participant. Defaults to \code{1}.
#'
#' @param seed Integer. A random seed passed to \code{set.seed()} at the start
#'   of the simulation, ensuring that results (including parameter sampling,
#'   trial randomization, and probabilistic choices) are reproducible across
#'   runs. Defaults to \code{42}.
#'
#' @param alpha0 Character string. The learning rate controlling how quickly
#'   associative strengths (\code{V}) update in response to prediction error.
#'   Must evaluate to a numeric value in \code{[0, 1]}. Passed as a string so
#'   it can specify either a fixed value (e.g. \code{alpha0 = "0.2"}) or a
#'   distribution to sample from independently for each participant (e.g.
#'   \code{alpha0 = "rbeta(1, 20, 80)"}). Defaults to \code{"0.2"}.
#'
#' @param alpha1 Character string. The learning rate controlling how quickly
#'   the variance estimate (\code{sigmasq}) updates based on squared
#'   prediction error. Must evaluate to a numeric value in \code{[0, 1]}.
#'   Passed as a string so it can specify either a fixed value (e.g.
#'   \code{alpha1 = "0.05"}) or a distribution to sample from independently
#'   for each participant (e.g. \code{alpha1 = "rbeta(1, 20, 80)"}).
#'   Defaults to \code{"0.05"}.
#'
#' @param alpha2 Character string. The learning rate controlling how quickly
#'   the effort matrix (\code{B}), which governs cue associability, updates
#'   toward the current belief-weighted cue structure. Must evaluate to a
#'   numeric value in \code{[0, 1]}. Passed as a string so it can specify
#'   either a fixed value (e.g. \code{alpha2 = "0.05"}) or a distribution to
#'   sample from independently for each participant (e.g. \code{alpha2 =
#'   "rbeta(1, 20, 80)"}). Defaults to \code{"0.05"}.
#'
#' @param gamma Character string. Controls the rate at which latent-state
#'   beliefs drift toward a uniform distribution over active states between
#'   trials. This same drift also determines the prior belief
#'   assigned to a newly created latent state: when a new state is added, it
#'   is initialized with a prior belief of \code{gamma}. Must evaluate to a
#'   numeric value in \code{[0, 1]}, where higher values cause beliefs to
#'   drift toward uniform more quickly (and give more initial credence to
#'   newly created states). Passed as a string so it can specify either a
#'   fixed value (e.g. \code{gamma = "0.01"}) or a distribution to sample
#'   from independently for each participant (e.g. \code{gamma =
#'   "rbeta(1, 20, 80)"}). Defaults to \code{"0.01"}.
#'
#' @param eta Character string. The change-point threshold: a new latent
#'   state is created once the accumulated change-point statistic reaches or
#'   exceeds this value. Must evaluate to a non-negative numeric value,
#'   where higher values make new-state creation less likely (requiring
#'   stronger, sustained evidence of a change before splitting off a new
#'   state). Passed as a string so it can specify either a fixed value (e.g.
#'   \code{eta = "1.2"}) or a distribution to sample from independently for
#'   each participant (e.g. \code{eta = "rgamma(1, shape = 25, rate = 50)"}). Defaults to
#'   \code{"1.2"}.
#'
#' @param delta Character string. A penalty that makes it harder for a new
#'   latent state to be created. Each trial, this penalty is subtracted from
#'   the evidence accumulating for a new state; if the evidence that trial
#'   isn't strong enough to outweigh the penalty, the accumulated evidence
#'   for a new state will decrease rather than increase. Higher
#'   values raise the bar for what counts as strong enough evidence, so with
#'   a large \code{delta}. Must evaluate to a
#'   non-negative numeric value. Passed as a string so it can specify either a
#'   fixed value (e.g. \code{delta = "0.6"}) or a distribution to sample from
#'   independently for each participant (e.g. \code{delta = "rgamma(1, shape = 25, rate = 50)"}).
#'   Defaults to \code{"0.6"}.
#'
#' @param sigma0 Character string. The agent's initial assumption about how
#'   much variability in outcomes is "expected," used to judge whether a
#'   prediction error is surprising enough to be considered evidence for a new state. A
#'   higher \code{sigma0} means the agent starts out expecting more noisy,
#'   unpredictable outcomes, so it takes larger prediction errors to be seen
#'   as informative. The agent's estimate of \code{sigma} will evolve throughout
#'   the simulation, so \code{sigma0} is mainly influential at the beginning of the simulation.
#'   Must evaluate to a positive numeric value. Passed as a
#'   string so it can specify either a fixed value (e.g. \code{sigma0 =
#'   "0.5"}) or a distribution to sample from independently for each
#'   participant (e.g. \code{sigma0 = "rgamma(1, shape = 25, rate = 50)"}). Defaults to
#'   \code{"0.5"}.
#'
#' @param tau Character string. The inverse temperature parameter controlling
#'   how deterministic choices are. Higher values make the agent more likely
#'   to consistently pick the option with the highest expected value, while
#'   lower values make choices more random, giving lower-valued options a
#'   better chance of being picked anyway. Must evaluate to a positive
#'   numeric value. Passed as a string so it can specify either a fixed
#'   value (e.g. \code{tau = "10"}) or a distribution to sample from
#'   independently for each participant (e.g. \code{tau =
#'   "rgamma(1, shape = 25, rate = 50)"}). Defaults to \code{"10"}.
#'
#' @param lambda Character string. Controls how much more strongly beliefs
#'   update on trials that are especially informative about which latent
#'   state is active. Informativeness is measured by how unevenly the
#'   evidence favors one state over the others on that trial.
#'   When one state clearly stands out, \code{lambda} amplifies how much
#'   beliefs shift toward it; when the evidence is ambiguous across states,
#'   \code{lambda} has little effect. Must evaluate to a non-negative
#'   numeric value, where higher values produce larger belief updates on
#'   informative trials. Passed as a string so it can specify either a
#'   fixed value (e.g. \code{lambda = "1"}) or a distribution to sample
#'   from independently for each participant (e.g. \code{lambda =
#'   "rgamma(1, shape = 25, rate = 50)"}). Defaults to \code{"1"}.
#'
#' @param pseudo Integer or \code{NULL}. Controls pseudorandomization of
#'   trial order within each phase: no more than \code{pseudo} consecutive
#'   trials may be of the same type. For example, \code{pseudo = 2} ensures
#'   the same trial type never appears more than twice in a row. If
#'   \code{NULL}, trial order is fully randomized with no such constraint.
#'   Ignored for any phase with only one trial type, since no reordering is
#'   possible. Defaults to \code{NULL}.
#'
#' @param block Integer, numeric vector, or \code{NULL}. The number of
#'   trials in each block, combining all trial types in their correct
#'   proportions. For example, if a phase has two trial types in a 1:3
#'   ratio and \code{block = 8}, each block will contain exactly 2 of the
#'   first type and 6 of the second, shuffled independently within that
#'   block. This produces a more even spread of trial types than
#'   randomizing order across the whole phase at once. A single number
#'   applies the same block length to every phase. A vector specifies a
#'   different length per phase. It
#'   must have one entry per phase. The number of trials in a phase must be
#'   evenly divisible by the block length, and each trial type's count must
#'   be evenly divisible by the resulting number of blocks. When
#'   \code{block} is used, \code{pseudo} is ignored. Defaults to
#'   \code{NULL}.
#'
#' @param probabilistic Logical. If \code{TRUE}, trial types are drawn
#'   independently according to their specified probabilities, rather than
#'   fixing the exact count of each type in advance. For example, with two
#'   trial types in a 1:3 ratio and 100 trials, the default (fixed-count)
#'   approach always produces exactly 25 and 75 trials. With
#'   \code{probabilistic = TRUE}, each trial is instead assigned
#'   independently with a 25% and 75% chance, so realized counts vary
#'   across phases.
#'   Defaults to \code{FALSE}.
#'
#' @param features Character vector, or \code{NULL}. Lets you restrict
#'   which features the agent tracks, instead of using every possible one.
#'   If \code{NULL} (default), features are generated automatically. Every
#'   single letter (e.g. \code{"A"}) becomes its own feature, and every
#'   multi-letter arm segment (e.g. \code{"AB"}) additionally becomes its
#'   own feature. For example, a trial with arm segment \code{"AB"} activates three
#'   features: \code{"A"}, \code{"B"}, and \code{"AB"}, meaning the agent
#'   tracks the AB combination as a distinct cue on top of tracking A and B
#'   individually.
#'
#'   To track only a subset of these -- for example, dropping \code{"AB"}
#'   as its own feature so it's represented only through \code{"A"} and
#'   \code{"B"}, or dropping \code{"A"} as its own feature so it's only
#'   ever represented as part of a larger combination like \code{"AB"} --
#'   supply a character vector specifying exactly the features you want.
#'   Each entry becomes exactly one feature, and no others are created.
#'   Defaults to \code{NULL}.
#'
#' @param V_noise_sd Numeric. Standard deviation of Gaussian noise added to
#'   the agent's associative strengths (\code{V}) on each trial, for
#'   plotting and output purposes only. The underlying \code{V} used for
#'   the agent's actual learning and choice behavior on future trials
#'   remains noise-free. The noisy version is computed separately each
#'   trial and only affects the overall \code{V} values recorded
#'   in the output (e.g. \code{V}, \code{V_cue},
#'   \code{V_shown}, \code{V_appear}), simulating measurement or
#'   observation noise without altering the agent's internal state. Noisy
#'   values are clamped to stay within \code{[-0.5, 0.5]}, the valid range
#'   for centered reward values. Defaults to \code{0} (no noise).
#'
#' @param ncop Integer >= 1. Maximum number of latent states the agent can
#'   create. Default is 10.
#'
#' @param group Character. Label for this batch's experimental group
#'   (e.g. \code{"ctrl"}, \code{"treatment"}), stored in the \code{group}
#'   column of the long-format output (\code{long}). Useful for combining
#'   multiple \code{run} calls into one dataset for ANOVA. Defaults to
#'   \code{"ctrl"}.
#'
#' @param prev Integer. Offset added to subject IDs in the long-format
#'   output (\code{long}), so participants are numbered starting after
#'   \code{prev} instead of at 1. Prevents subject ID collisions when
#'   combining multiple \code{run} calls (e.g. across groups) into one
#'   dataset for ANOVA. For example, if a control group of 20 is run
#'   first, a subsequent call would use \code{prev = 20} so IDs start at
#'   21. Defaults to \code{0}.
#'
#' @param ITI List of numeric vectors, or \code{NULL}. Timestamps (not
#'   gaps) marking when each trial occurs. Must be a list of length
#'   \code{n} (one vector per participant, even if \code{n = 1}), where
#'   each vector has length \code{ntrials} and is non-decreasing. This
#'   requires specifying a timestamp for every trial, even those with the
#'   default gap of 1; if only a few trials have a non-default gap,
#'   and that gap is the same for every participant, see
#'   \code{ezITI} instead, which only requires listing those exceptions.Gaps between consecutive timestamps partially decay
#'   latent-state beliefs toward uniform (with a decay of
#'   \code{gamma} for each one unit gap), simulating forgetting, and allow more
#'   rumination iterations (\code{chi}) on the preceding trial. Cannot be
#'   used together with \code{ezITI}.
#'
#'   For example, with \code{ntrials = 3} and \code{n = 2}, \code{ITI =
#'   list(c(1, 2, 3), c(1, 2, 10))} gives participant 1 the standard single
#'   \code{gamma} decay between each trial, while participant 2 has a large
#'   gap before trial 3, producing the equivalent of 8 successive
#'   \code{gamma} decays (since the gap between timestamps 2 and 10 is 8).
#'   Defaults to \code{NULL} (trials evenly spaced, one unit apart).
#'
#' @param ezITI List, or \code{NULL}. A simpler alternative to \code{ITI}
#'   for specifying uneven trial timing, shared across all participants
#'   (use \code{ITI} instead if you need per-participant timing). Each
#'   element is a length-2 numeric vector \code{c(trial, gap)}, setting the
#'   gap immediately after that trial to \code{gap} timestamp units
#'   instead of the default gap of 1. All unspecified gaps remain 1.
#'   \code{trial} must be a whole number between 1 and \code{ntrials - 1},
#'   and \code{gap} must be greater than 1. These gaps are used to build
#'   the full \code{ITI} timestamp vector internally, so they affect belief
#'   decay and rumination (\code{chi}) the same way \code{ITI} does.
#'   Cannot be used together with \code{ITI}.
#'
#'   For example, with \code{ntrials = 3}, \code{ezITI = list(c(2, 8))}
#'   sets the gap after trial 2 to 8, producing the same effective
#'   timestamps as \code{ITI = list(c(1, 2, 10))}. This is equivalent to 8
#'   successive \code{gamma} decays before trial 3. Defaults to \code{NULL}.
#'
#'
#' @param chi Character. The number of "rumination" iterations the agent
#'   performs on a single trial's outcome before moving to the next trial.
#'   Higher values let the agent update its associative strengths
#'   repeatedly from the same outcome, simulating extra "thinking time"
#'   between trials. The actual number of iterations used is capped by the
#'   \code{ITI} gap before the next trial, so \code{chi} only has an
#'   effect on trials where that gap exceeds 1; if \code{ITI} is
#'   \code{NULL}, \code{chi} has no effect at all. Passed as a string, so
#'   it can specify either a fixed value (e.g. \code{chi = "1"}) or a distribution
#'   to sample from independently for each participant (e.g. chi =
#'   \code{"rpois(1, 2) + 1"}). Must be a whole number of at least 1.
#'   Defaults to \code{"1"} (no extra rumination).
#'
#' @param contextShift Numeric vector, or \code{NULL}. Trial numbers
#'   marking the first trial of a new context, at which latent-state
#'   beliefs are fully reset to uniform right before that trial's update,
#'   simulating a change in context (e.g. a new room). Unlike
#'   \code{ITI}, which gradually decays beliefs based on timestamp gaps,
#'   this resets them instantly. Only belief
#'   proportions (\code{lsb}) are affected -- latent states, \code{V},
#'   \code{B}, and \code{sigmasq} are untouched.
#'
#'   For example, \code{contextShift = c(50, 100)} resets beliefs right
#'   before trials 50 and 100. Defaults to \code{NULL} (no forced resets).
#'
#' @param my_world List, or \code{NULL}. Input a fixed, non-randomized
#'   order of cues and rewards. This will override the random order that is
#'   usually created from \code{phase_def}. This allows complete control over
#'   the trial sequence of the simulated experiment. The input should be built with the \code{make_world()}
#'   function, which
#'   returns the required \code{c_vec}(cue order array), \code{win} (reward order vector),
#'   and \code{type_seq} (flattened cue order vector) elements
#'   in the correct format. \code{phase_def} is still needed to determine
#'   trial counts, phase boundaries, and reward validation, but no new
#'   trial order or reward assignment is generated. Defaults to
#'   \code{NULL} (world is generated randomly).
#'
#' @param initialex Numeric array, or \code{NULL}. Sets the agent's initial
#'   associative strengths (\code{V}), overriding the default of 0. Must be
#'   an array of dimension \code{D x C x L} (dimensions x cues x latent
#'   states). \code{initialex[d, c, l]} is cue \code{c}'s starting value on
#'   dimension \code{d} in state \code{l}. Only state 1 is active initially,
#'   so setting all \code{L} slices the same applies that value to every
#'   future state as it's created, while setting only slice 1 (rest at 0)
#'   leaves new states starting fresh at 0.
#'   Values must be in \code{[-0.5, 0.5]} (the centered reward scale). Defaults to
#'   \code{NULL} (every cue starts at 0 in every state).
#'
#' @param plot Logical. If \code{TRUE}, produces a plot of overall
#'   associative strength (\code{V}) across trials, with one line per cue.
#'   Phase boundaries are marked with dashed vertical lines.
#'   By default, only participant 1 is plotted; to plot every participant
#'   instead, use \code{plot_all}. Combine with \code{plot_label} to show
#'   trial-type labels (colored by reward) on the x-axis, and with
#'   \code{plot_arm} to also show which arm was chosen on each trial (when there's
#'   more than one option).
#'   Defaults to \code{FALSE}.
#'
#' @param plot_all Logical. Same as \code{plot}, but produces the V plot for
#'   every participant instead of only participant 1. Defaults to
#'   \code{FALSE}.
#'
#' @param plot_label Logical. If \code{TRUE}, adds trial-type labels along
#'   the x-axis of the relevant plots (\code{plot}/\code{plot_all},
#'   \code{plot_raw}/\code{plot_raw_all}, \code{plot_mu}/\code{plot_mu_all},
#'   \code{plot_q}/\code{plot_q_all}, \code{plot_sigma}/\code{plot_sigma_all}),
#'   colored green if that trial was rewarded and red if it wasn't.
#'   Defaults to \code{FALSE}.
#'
#' @param plot_arm Logical. If \code{TRUE}, adds the chosen arm number
#'   beneath the trial-type labels on the x-axis, for any of the following plots:
#'   \code{plot}/\code{plot_all},
#'   \code{plot_raw}/\code{plot_raw_all}, \code{plot_mu}/\code{plot_mu_all},
#'   \code{plot_q}/\code{plot_q_all}, \code{plot_sigma}/\code{plot_sigma_all}.
#'   Has no effect unless \code{plot_label = TRUE}. Defaults to \code{FALSE}.
#'
#' @param plot_mu Logical. If \code{TRUE}, plots the expected value
#'   (\code{mu}) of the chosen arm across trials, one plot per reward
#'   dimension, for participant 1. Combine with \code{plot_label} to show
#'   trial-type labels (colored by reward) on the x-axis, and
#'   \code{plot_arm} to also show the chosen arm beneath each label.
#'   Note: this compound's true value can fall below -0.5 under some conditions
#'   (see Supplementary Materials S1.8.1). When it does, the plotted value is
#'   clamped to -0.5, which is the value of a non-rewarded trial and thus
#'   the minimum theoretically meaningful associative value.
#'   Defaults to \code{FALSE}.
#'
#' @param plot_mu_all Logical. Same as \code{plot_mu}, but produces the mu
#'   plot for every participant instead of only participant 1. Note: this
#'   compound's true value can fall below -0.5 under some conditions
#'   (see Supplementary Materials S1.8.1). When it does, the plotted value is
#'   clamped to -0.5, which is the value of a non-rewarded trial and thus
#'   the minimum theoretically meaningful associative value. Defaults to
#'   \code{FALSE}.
#'
#' @param plot_shown Logical. If \code{TRUE}, plots overall V by
#'   presentation number (rather than trial number) for each distinct
#'   trial-type cue combination (e.g. \code{"A"}, \code{"AB"}), averaged
#'   across all participants. Phase boundaries are marked based on when
#'   that cue combination was actually chosen. Combine with \code{plot_sd}
#'   to add a shaded standard-deviation ribbon when \code{n > 1}. Defaults
#'   to \code{FALSE}.
#'
#' @param plot_cue Logical. If \code{TRUE}, plots the agent's V by
#'   presentation number for each individual base cue, averaged across all
#'   participants, pooling every trial type that cue appears in (e.g.
#'   \code{"A"} pools presentations of \code{"A"} and \code{"AB"} together).
#'   Note that a base cue's own value, as plotted here, is not the
#'   associative value the agent actually experiences on a given trial.
#'   This can be confusing when \code{"AB"} is itself a base cue under the
#'   default encoding: \code{"AB"} as a base cue refers only to that
#'   configural feature's own isolated value, whereas \code{"AB"} as a
#'   trial type refers to the whole compound experience, whose true value
#'   is the sum of every contributing cue (\code{"A"} + \code{"B"} +
#'   \code{"AB"}). This plot shows each base cue -- including a configural
#'   cue like \code{"AB"} -- as its own separate, individual line; to see
#'   the trial-level total instead, use \code{plot_shown}. Each base cue
#'   gets its own plot. Combine with \code{plot_sd} to add a shaded
#'   standard-deviation ribbon when \code{n > 1}. Defaults to \code{FALSE}.
#'
#' @param plot_sd Logical. If \code{TRUE} and \code{n > 1}, adds a shaded
#'   ribbon showing +/- 1 SD around the mean line on presentation-based
#'   plots (\code{plot_cue}, \code{plot_shown}, \code{plot_appears}). Has no
#'   effect with a single participant. Defaults to \code{FALSE}.
#'
#' @param plot_raw Logical. If \code{TRUE}, plots raw (unweighted) V for
#'   each active latent state and reward dimension, across trials, for
#'   participant 1. Belief proportion for that state is overlaid as a
#'   dashed line on a second y-axis. V is not shown until the trial after a
#'   state's creation, as that is the first trial when the latent state will
#'   actually affect the overall V. Combine with \code{plot_label} to show trial-type
#'   labels (colored by reward) on the x-axis, and \code{plot_arm} to also
#'   show the chosen arm beneath each label. Defaults to \code{FALSE}.
#'
#' @param plot_raw_all Logical. Same as \code{plot_raw}, but produces the
#'   raw V plot for every participant instead of only participant 1.
#'   Defaults to \code{FALSE}.
#'
#' @param plot_q Logical. If \code{TRUE}, plots the change-point statistic
#'   (\code{q}) across trials for participant 1, with a dashed horizontal
#'   line marking the \code{eta} threshold at which a new latent state is
#'   created. Trials where \code{q} was \code{Inf} (an immediate,
#'   unambiguous state change) are marked with a red triangle and
#'   \eqn{\infty} label, capped at \eqn{2 \times \code{eta}} for display
#'   purposes only. Combine with \code{plot_label} to show trial-type
#'   labels (colored by reward) on the x-axis, and \code{plot_arm} to also
#'   show the chosen arm beneath each label. Defaults to \code{FALSE}.
#'
#' @param plot_q_all Logical. Same as \code{plot_q}, but produces the q
#'   plot for every participant instead of only participant 1. Defaults to
#'   \code{FALSE}.
#'
#' @param plot_sigma Logical. If \code{TRUE}, plots the variance estimate
#'   (\code{sigma}) at the start of each trial, across trials, for
#'   participant 1. Combine with \code{plot_label} to show trial-type
#'   labels (colored by reward) on the x-axis, and \code{plot_arm} to also
#'   show the chosen arm beneath each label. Defaults to \code{FALSE}.
#'
#' @param plot_sigma_all Logical. Same as \code{plot_sigma}, but produces
#'   the sigma plot for every participant instead of only participant 1.
#'   Defaults to \code{FALSE}.
#'
#' @param plot_appears Logical. If \code{TRUE}, plots the total value of
#'   the chosen arm by presentation number for each base cue, averaged
#'   across all participants, whenever that cue was part of the chosen arm
#'   (alone or combined with others). Unlike \code{plot_cue}, which tracks
#'   a cue's own isolated value, this reflects the combined value of the
#'   whole arm on every presentation that cue contributed to. In other words,
#'   this plots the overall V of trials where a certain cue "appears". Combine with
#'   \code{plot_sd} to add a shaded standard-deviation ribbon when
#'   \code{n > 1}.
#'   Note: this compound's true value can fall below -0.5 under some conditions
#'   (see Supplementary Materials S1.8.1). When it does, the plotted value is
#'   clamped to -0.5, which is the value of a non-rewarded trial and thus
#'   the minimum theoretically meaningful associative value.
#'   Defaults to \code{FALSE}.
#'
#' @param fixed_axis Logical. If \code{TRUE}, y-axes are fixed to a
#'   consistent range across plots and participants (\code{[-0.5, 0.5]} for
#'   V and mu plots, and a shared data-driven range for \code{q} and
#'   \code{sigma} plots), making plots directly comparable. If
#'   \code{FALSE}, each plot's y-axis is scaled to its own data range.
#'   Defaults to \code{TRUE}.
#'
#' @return A list of length \code{n}, one element per participant. Each
#'   element is itself a list containing:
#'   \describe{
#'     \item{\code{arm_chosen}}{Integer vector of length \code{ntrials}
#'       giving the arm chosen by the agent on each trial.}
#'     \item{\code{win_received}}{Matrix (\code{ntrials x D}) of the reward
#'       outcome received on each trial, for each reward dimension.}
#'     \item{\code{lsb}}{Matrix (\code{ntrials x L}) of latent-state
#'       beliefs at the start of each trial, where \code{ncop} is the total
#'       number of latent states.}
#'     \item{\code{Lmax}}{Integer vector of length \code{ntrials} giving
#'       the number of active latent states at each trial.}
#'     \item{\code{q}}{Numeric vector of length \code{ntrials} giving the
#'       change-point statistic after that trial's reward but before any
#'       new-state creation/reset (the value used for plotting).}
#'     \item{\code{sigma}}{Numeric vector of length \code{ntrials} giving
#'       the agent's variance estimate at the start of each trial.}
#'     \item{\code{mu}}{Array (\code{ntrials x D x A}) of the
#'       overall expected value \code{V} of each arm, for each reward
#'       dimension, before the agent's choice on each trial.}
#'     \item{\code{mu_raw}}{Array (\code{ntrials x D x L x A}) of the
#'       unweighted expected value of each arm, shown separately for each latent
#'       state, for each reward dimension.}
#'     \item{\code{V}}{Named list with one entry per base cue, each a
#'       matrix (\code{ntrials x D}) of that cue's overall
#'       associative strength over trials.}
#'     \item{\code{V_cue}}{Named list with one entry per base cue, each a
#'       matrix (presentations x D) of that cue's own overall
#'       value, recorded only on trials where the cue was shown.}
#'     \item{\code{V_raw}}{Named list with one entry per base cue, each an
#'       array (\code{ntrials x D x L}) of that cue's raw (unweighted)
#'       associative strength in every latent state over trials.}
#'     \item{\code{V_shown}}{Named list with one entry per trial type
#'       (e.g. \code{"A"}, \code{"AB"}), each a matrix (presentations x D)
#'       of the agent's total V whenever that exact trial type was the
#'       chosen arm.}
#'     \item{\code{V_appear}}{Named list with one entry per base cue, each
#'       a matrix (presentations x D) of the agent's total V on the chosen
#'       arm, recorded whenever that cue was part of the chosen arm.}
#'     \item{\code{B}}{Array (\code{ntrials x C x C x D x L}) giving the
#'       effort matrix, which governs cue associability, for every cue
#'       pair, reward dimension, and latent state, at every trial.}
#'     \item{\code{par}}{The full parameter and task configuration used to
#'     generate this participant. The most useful pieces are \code{ls}
#'     (the exact parameters used
#'     for this participant, e.g. \code{alpha0}, \code{eta}, \code{delta})
#'     and \code{tau} (the inverse-temperature parameter used). The
#'     remaining fields (\code{D}, \code{C}, \code{A}, \code{ntrials},
#'     \code{base_cues}, \code{ITI}, etc.) describe the task's structure and
#'     are mainly used internally (e.g. by \code{plot_compare()}) to
#'     recover cue counts, trial counts, and timing.}
#'     \item{\code{world}}{List of the two arrays defining what was
#'       presented each trial: \code{c_vec} (\code{D x C x L x A x
#'       ntrials}), which cues were active per dimension, latent state,
#'       and arm; and \code{win} (\code{D x A x ntrials}), the reward
#'       generated for each arm and dimension, regardless of the agent's
#'       choice. Also includes \code{type_seq}, which is a list
#'       showing the order of cues presented to this participant.}
#'     \item{\code{long}}{List (one entry per phase) of data frames
#'       combining every cue type presented in that phase, in long format,
#'       for \code{D = 1} only, intended for running an ANOVA using the
#'       ezANOVA package. Columns are: \code{subject}, the participant ID
#'       (\code{i + prev}, offset so IDs don't collide across combined
#'       \code{run()} calls); \code{group}, the batch label from the
#'       \code{group} argument; \code{cue_type}, the trial type or arm
#'       segment (e.g. \code{"A"}, \code{"AB"}) this row belongs to;
#'       \code{presentation}, the ordinal position of this row among all
#'       presentations of that cue type within this phase; \code{V}, the
#'       agent's overall associative value on that presentation; and, when
#'       \code{block} is specified, \code{block}, the block number that
#'       presentation falls into.}
#'   }
#'
#' @references Cochran, A. L., & Cisler, J. M. (2019). A flexible and
#' generalizable model of online latent-state learning. PLOS Computational
#' Biology, 15(9). \doi{10.1371/journal.pcbi.1007331}
#'
#' @examples
#' phase_def <- list(
#'   phase1 = list(
#'     trials = list(
#'       A_B  = 20,
#'       B_A  = 10,
#'       AB_C = 12,
#'       C_AB = 6,
#'       B_0  = 5
#'     ),
#'     rewards = list(
#'       D1 = c(A = 0.80, B = 0.20, AB = 0.50),
#'       D2 = c(A = 0.60, B = 0.00, AB = 0.00)
#'     )
#'   ),
#'   phase2 = list(
#'     trials = list(
#'       A_B  = 15,
#'       B_A  = 15,
#'       AB_C = 8,
#'       C_AB = 8
#'     ),
#'     rewards = list(
#'       D1 = c(A = 0.10, B = 0.50, C = 0.00, AB = 0.25),
#'       D2 = c(A = 0.00, B = 0.60, AB = 0.00)
#'     )
#'   )
#' )
#' result <- run(phase_def, n = 5)
#'
#' @export
run = function(phase_def,
               n = 1,
               seed = 42,

               alpha0 = "0.2",
               alpha1 = "0.05",
               alpha2 = "0.05",
               gamma  = "0.01",
               eta    = "1.2",
               delta  = "0.6",
               sigma0 = "0.5",
               tau = "10",
               lambda = "1",

               pseudo = NULL,
               block = NULL,
               probabilistic = FALSE,
               features = NULL,

               V_noise_sd = 0,

               ncop   = 10,

               group = "ctrl",
               prev = 0,

               ITI = NULL,
               ezITI = NULL,
               chi    = "1",
               contextShift = NULL,

               my_world = NULL,
               initialex = NULL,

               plot=FALSE,
               plot_all=FALSE,
               plot_label=FALSE,
               plot_arm=FALSE,
               plot_mu=FALSE,
               plot_mu_all=FALSE,
               plot_shown=FALSE,
               plot_cue=FALSE,
               plot_sd=FALSE,
               plot_raw=FALSE,
               plot_raw_all=FALSE,
               plot_q=FALSE,
               plot_q_all=FALSE,
               plot_sigma=FALSE,
               plot_sigma_all=FALSE,
               plot_appears=FALSE,
               fixed_axis=TRUE
) {
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  all_results <- vector("list", n)
  set.seed(seed)

  # ============================================================
  # Parameter sampling helpers
  # ============================================================

  sample_par <- function(x, name) {
    if (!is.character(x)) stop(
      name, " must be passed as a string. ",
      "Use e.g. ", name, " = \"0.05\" for a fixed value or ",
      name, " = \"rnorm(1, 0.05, 0.01)\" for a distribution."
    )
    val <- eval(parse(text = x))[1]
    if (!is.numeric(val) || !is.finite(val)) stop(
      name, " must evaluate to a single finite numeric value, but got: ", deparse(val), "."
    )
    val
  }

  for (i in 1:n) {

    # ============================================================
    # 1. Create participant parameters for this participant
    # ============================================================
    tau_ind    <- sample_par(tau,    "tau")
    alpha0_ind <- sample_par(alpha0, "alpha0")
    alpha1_ind <- sample_par(alpha1, "alpha1")
    alpha2_ind <- sample_par(alpha2, "alpha2")
    gamma_ind  <- sample_par(gamma,  "gamma")
    eta_ind    <- sample_par(eta,    "eta")
    delta_ind  <- sample_par(delta,  "delta")
    sigma0_ind <- sample_par(sigma0, "sigma0")
    chi_ind    <- sample_par(chi, "chi")
    lambda_ind <- sample_par(lambda, "lambda")

    # Parameter range checks
    if (tau_ind <= 0)              stop("tau must be positive, got: ", tau_ind, ".")
    if (alpha0_ind < 0 || alpha0_ind > 1) stop("alpha0 must be in [0, 1], got: ", alpha0_ind, ".")
    if (alpha1_ind < 0 || alpha1_ind > 1) stop("alpha1 must be in [0, 1], got: ", alpha1_ind, ".")
    if (alpha2_ind < 0 || alpha2_ind > 1) stop("alpha2 must be in [0, 1], got: ", alpha2_ind, ".")
    if (gamma_ind  < 0 || gamma_ind  > 1) stop("gamma must be in [0, 1], got: ", gamma_ind,  ".")
    if (eta_ind    < 0)            stop("eta must be positive, got: ", eta_ind, ".")
    if (delta_ind  < 0)            stop("delta must be positive, got: ", delta_ind, ".")
    if (sigma0_ind <= 0)           stop("sigma0 must be positive, got: ", sigma0_ind, ".")
    if (chi_ind    < 1)            stop("chi must be at least 1, got: ", chi_ind, ".")
    if (chi_ind    < 1)            stop("lambda must be positive, got: ", lambda_ind, ".")

    # ============================================================
    # 2. Initialize task
    # ============================================================

    #for simplicity, I've hardcoded par_L to 1 (the argument for initial number
    #of latent states), meaning that the function can
    #only simulate results where the agent begins with a single latent state.
    #However, the original code from Cochran and Cisler (2019) included code to
    #allow agents to begin with more than one latent state. In the original code,
    #latent states increase proportional to your starting number of latent states.
    #For example, if you start with 2 latent states (par_L=2), then when an
    #individual creates a new latent state, they actually create two new states
    #Essentially, agents create par_L new states each time the state creation
    #threshold is reached. I do not understand why the original authors made this
    #decision, so I chose not to allow par_L greater than 1. My code does keep
    #all the necessary code for par_L > 1, so if you understand why this multi-
    #state creation feature exist, then feel free to use it.

    out   <- TaskInit(phase_def = phase_def,
                      pseudo = pseudo,
                      par_L = 1,
                      ncop = ncop,
                      center = 0.5,
                      contextShift = contextShift,
                      block = block,
                      probabilistic = probabilistic,
                      features = features)
    world <- out$world
    par   <- out$par
    all_cue_types <- out$cue_names

    # ============================================================
    # 3. Set model parameters
    # ============================================================

    par$tau <- tau_ind

    par$ls <- list(
      alpha0 = alpha0_ind,
      alpha1 = alpha1_ind,
      alpha2 = alpha2_ind,
      gamma  = gamma_ind,
      ncop   = ncop,
      eta    = eta_ind,
      delta  = delta_ind,
      chi    = chi_ind,
      sigma0 = sigma0_ind,
      lambda = lambda_ind
    )

    # Make sure ITI is in the correct form
    ntrials <- sum(sapply(phase_def, function(ph) sum(unlist(ph$trials))))


    # --- 1. Disallow using both ITI and ezITI ---
    if (!is.null(ITI) && !is.null(ezITI)) {
      stop("Cannot use both ITI and ezITI at the same time. Please specify only one.")
    }

    # --- 2. Validate ITI if provided directly ---
    if (!is.null(ITI)) {
      if (!is.list(ITI) || length(ITI) != n) stop(paste0(
        "ITI must be a list with exactly n (", n, ") vectors, ",
        "but has length ", length(ITI), "."
      ))
      for (j in seq_along(ITI)) {
        if (length(ITI[[j]]) != ntrials) stop(paste0(
          "ITI[[", j, "]] must have length equal to the number of trials (", ntrials, "), ",
          "but has length ", length(ITI[[j]]), "."
        ))
      }
    }

    # --- 3. Validate ezITI if used ---
    if (!is.null(ezITI)) {
      if (!is.list(ezITI) || length(ezITI) == 0) {
        stop("ezITI must be a non-empty list of numeric vectors c(trial, ITI).")
      }
      for (j in seq_along(ezITI)) {
        v <- ezITI[[j]]
        if (!is.numeric(v) || length(v) != 2) {
          stop(paste0(
            "Each ezITI[[", j, "]] must be a numeric vector of length 2: c(trial, ITI)."
          ))
        }
        trial_raw <- v[1]
        iti_gap   <- v[2]

        if (!is.numeric(trial_raw) || length(trial_raw) != 1 ||
            trial_raw != round(trial_raw)) {
          stop("In ezITI[[", j, "]], trial index ", trial_raw,
               " is not an integer. Use whole trial numbers, e.g. 56, not 56.5.")
        }
        trial_idx <- as.integer(trial_raw)
        if (trial_idx < 1 || trial_idx >= ntrials) {
          stop("In ezITI[[", j, "]], trial index ", trial_idx,
               " must be between 1 and ", ntrials - 1, ".")
        }
        if (!is.finite(iti_gap) || iti_gap <= 1) {
          stop("In ezITI[[", j, "]], the ITI gap ", iti_gap,
               " must be greater than 1 (the default gap is 1).")
        }
      }
    }

    # --- 4. Construct par$ITI for this participant ---
    if (!is.null(ITI)) {
      # Use direct per-participant ITI
      par$ITI <- ITI[[i]]

    } else if (!is.null(ezITI)) {
      # Build from ezITI (global schedule, same for all participants)
      if (ntrials == 1) {
        par$ITI <- 1
      } else {
        gaps <- rep(1, ntrials - 1)  # default gaps

        for (spec in ezITI) {
          trial_idx <- as.integer(spec[1])
          iti_gap   <- spec[2]
          gaps[trial_idx] <- iti_gap
        }

        par$ITI <- numeric(ntrials)
        par$ITI[1] <- 1
        for (t in 2:ntrials) {
          par$ITI[t] <- par$ITI[t - 1] + gaps[t - 1]
        }
      }

    } else {
      par$ITI <- NULL
    }

    par$initialex <- initialex

    #lsb0 sets the initial belief in latent states for cases where the agent
    #begins with more than one latent state. I've hardcoded lsb0 to NULL, as
    #the code currently does not allow the agent to begin with more than one
    #latent state. I've left the code in for lsb0 in case a future user figures
    #out the code for starting with more than one latent state, and wants to
    #use this lsb0 feature.
    par$lsb0      <- NULL

    L <- par$L * par$ls$ncop

    # Cannot use both ITI and ezITI
    if (!is.null(ITI) && !is.null(ezITI)) {
      stop("Cannot use both ITI and ezITI at the same time. Please specify only one.")
    }

    # Validate ITI (if provided directly)
    if (!is.null(ITI)) {
      if (length(ITI) != n) stop(paste0(
        "ITI must be a list with exactly n (", n, ") vectors, ",
        "but has length ", length(ITI), "."
      ))
      for (j in seq_along(ITI)) {
        if (length(ITI[[j]]) != ntrials) stop(paste0(
          "ITI[[", j, "]] must have length equal to the number of trials (", ntrials, "), ",
          "but has length ", length(ITI[[j]]), "."
        ))
      }
    }

    # ezITI: one global schedule, same for all participants.
    # It must be a non-empty list; each element is c(trial, ITI_gap).
    if (!is.null(ezITI)) {
      if (!is.list(ezITI) || length(ezITI) == 0) {
        stop("ezITI must be a non-empty list of numeric vectors c(trial, ITI).")
      }
      for (j in seq_along(ezITI)) {
        v <- ezITI[[j]]
        if (!is.numeric(v) || length(v) != 2) {
          stop(paste0(
            "Each ezITI[[", j, "]] must be a numeric vector of length 2: c(trial, ITI)."
          ))
        }
      }
    }

    #replace generated world with a custom one inserted by the user
    if (!is.null(my_world)) {

      # Must be a list
      if (!is.list(my_world)) stop(
        "my_world must be a list with elements $c_vec and $win."
      )

      # Must contain both required elements
      if (is.null(my_world$c_vec)) stop(
        "my_world must contain element $c_vec."
      )
      if (is.null(my_world$win)) stop(
        "my_world must contain element $win."
      )

      # c_vec must be a 5-dimensional array of the correct size
      if (!is.array(my_world$c_vec) || length(dim(my_world$c_vec)) != 5) stop(
        "my_world$c_vec must be a 5-dimensional array (D x C x L x A x ntrials)."
      )
      if (!all(dim(my_world$c_vec) == c(par$D, par$C, L, par$A, par$ntrials))) stop(paste0(
        "my_world$c_vec has wrong dimensions. ",
        "Expected (", paste(c(par$D, par$C, L, par$A, par$ntrials), collapse = " x "), "), ",
        "got (", paste(dim(my_world$c_vec), collapse = " x "), ")."
      ))

      # win must be a 3-dimensional array of the correct size
      if (!is.array(my_world$win) || length(dim(my_world$win)) != 3) stop(
        "my_world$win must be a 3-dimensional array (D x A x ntrials)."
      )
      if (!all(dim(my_world$win) == c(par$D, par$A, par$ntrials))) stop(paste0(
        "my_world$win has wrong dimensions. ",
        "Expected (", paste(c(par$D, par$A, par$ntrials), collapse = " x "), "), ",
        "got (", paste(dim(my_world$win), collapse = " x "), ")."
      ))

      world <- my_world
      # Rebuild type_seq with per-phase structure matching phase_def
      flat_seq <- unlist(my_world$type_seq)
      phase_lengths <- sapply(phase_def, function(ph) sum(unlist(ph$trials)))
      phase_ends_idx <- cumsum(phase_lengths)
      phase_starts_idx <- c(1, phase_ends_idx[-length(phase_ends_idx)] + 1)
      out$type_seq <- setNames(lapply(seq_along(phase_def), function(ph) {
        as.list(flat_seq[phase_starts_idx[ph]:phase_ends_idx[ph]])
      }), names(phase_def))
    }

    # Validate chi if non-null (must be an integer)
    if (!is.null(par$ls$chi)) {
      if (!is.numeric(par$ls$chi) || length(par$ls$chi) != 1) stop("chi must be a single numeric value.")
      if (par$ls$chi < 1) stop("chi must be at least 1.")
      if (par$ls$chi != round(par$ls$chi)) stop("chi must be a whole number.")
      if (is.null(par$ITI) && chi != "1") warning(
        "chi has no effect when ITI is NULL. ",
        "Rumination iterations are only applied when inter-trial intervals are provided via the ITI argument."
      )
    }

    # Validate pre-specified initial V
    if (!is.null(par$initialex)) {
      if (!is.array(initialex) || !all(dim(initialex) == c(par$D, par$C, L))) {
        stop(paste0(
          "initialex must be a numeric array of dimension D x C x L (",
          par$D, " x ", par$C, " x ", L, "), ",
          "but has dimensions ", paste(dim(initialex), collapse = " x "), "."
        ))
      }
      if (any(abs(initialex) > 0.5)) warning(
        "initialex contains values outside [-0.5, 0.5]. Values are on the centered reward scale ",
        "and should lie within this range unless rewards are non-binary."
      )
    }
    # ============================================================
    # 4. Storage objects
    # ============================================================

    arm_chosen   = numeric(par$ntrials)
    win_received = matrix(NA, nrow = par$ntrials, ncol = par$D)  # D outcomes per trial
    lsb          = matrix(NA, par$ntrials, L)
    Lmax         = numeric(par$ntrials)
    q            = numeric(par$ntrials)
    mu = array(NA, dim = c(par$ntrials, par$D, par$A))
    mu_raw = array(NA, dim = c(par$ntrials, par$D, L, par$A))
    B_history <- array(NA, dim = c(par$ntrials, par$C, par$C, par$D, L))
    sigma       = numeric(par$ntrials)

    # Belief-weighted V: trial x D x cue
    V_weighted_history <- array(NA, dim = c(par$ntrials, par$D, par$C))
    dimnames(V_weighted_history) <- list(
      trial = NULL,
      dim   = paste0("D", 1:par$D),
      cue   = paste0("cue", 1:par$C)
    )

    # Belief-weighted V for shown cues only: NA for cues not presented on that trial
    V_weighted_shown <- array(NA, dim = c(par$ntrials, par$D, par$C))
    dimnames(V_weighted_shown) <- list(
      trial = NULL,
      dim   = paste0("D", 1:par$D),
      cue   = paste0("cue", 1:par$C)
    )

    # Raw V for all latent states: trial x D x cue x state
    V_raw_history <- array(NA, dim = c(par$ntrials, par$D, par$C, L))
    dimnames(V_raw_history) <- list(
      trial = NULL,
      dim   = paste0("D", 1:par$D),
      cue   = paste0("cue", 1:par$C),
      state = paste0("state", 1:L)
    )

    #used to create V_shown
    type_seq_flat <- unlist(out$type_seq)
    all_trial_types <- unique(Filter(function(s) s != "0",
                                     unlist(lapply(type_seq_flat, function(nm) strsplit(nm, "_")[[1]]))
    ))

    # V for each cue type (e.g. A, AX)
    V_shown <- setNames(vector("list", length(all_trial_types)), all_trial_types)
    for (ct in all_trial_types) {
      V_shown[[ct]] <- matrix(NA_real_, nrow = 0, ncol = par$D)
      colnames(V_shown[[ct]]) <- paste0("D", 1:par$D)
    }

    V_appear <- setNames(vector("list", length(par$base_cues)), par$base_cues)
    for (bc in par$base_cues) {
      V_appear[[bc]] <- matrix(NA_real_, nrow = 0, ncol = par$D)
      colnames(V_appear[[bc]]) <- paste0("D", 1:par$D)
    }

    # For each feature cue type, find all feature indices that are subsets of it,
    # so that later code can sum them to get the total value for that cue type.
    # ct = "A"  -> indices for ["A"]
    # ct = "AB" -> indices for ["A", "B", "AB"] (all features whose letters ⊆ {A,B})
    cue_type_cue_idx <- setNames(lapply(all_trial_types, function(ct) {
      ct_letters <- strsplit(ct, "")[[1]]
      which(sapply(par$base_cues, function(f) {
        f_letters <- strsplit(f, "")[[1]]
        all(f_letters %in% ct_letters)
      }))
    }), all_trial_types)

    # For each base cue, find all cue types (arm segments) whose letters are a
    # superset of that base cue's letters, i.e. where that base cue "appears".
    # e.g. base cue "A" appears in cue types "AX" and "AY". This is used for
    # plot_appears
    appear_ct_map <- setNames(lapply(seq_along(par$base_cues), function(bc_idx) {
      all_trial_types[sapply(all_trial_types, function(ct) bc_idx %in% cue_type_cue_idx[[ct]])]
    }), par$base_cues)

    # ============================================================
    # 5. Run simulation
    # ============================================================

    agent <- LatentState(par, NULL, NULL)

    for (t in 1:par$ntrials) {
      par$t <- t

      world_this <- list(
        win   = array(world$win[, , t],       dim = c(par$D, par$A)),
        c_vec = array(world$c_vec[, , , , t], dim = c(par$D, par$C, L, par$A))
      )

      agent <- LatentState(par, agent, world_this)

      # Add noise to this output, such that the "true" value is used for future
      # computations, and the noise only affects this specific trial
      V_noisy <- agent$V_pre + array(
        rnorm(length(agent$V_pre), mean = 0, sd = V_noise_sd),
        dim = dim(agent$V_pre)
      )
      # Make sure V stays within its range
      V_noisy <- pmax(-0.5, pmin(0.5, V_noisy))
      dim(V_noisy) <- dim(agent$V_pre)

      # Store trial-level results
      arm_chosen[t]      <- agent$arm
      win_received[t, ]  <- agent$win
      lsb[t, ]           <- agent$lsb
      Lmax[t]            <- agent$Lmax
      q[t]               <- agent$q_plot
      mu[t, , ] <- agent$mu
      mu_raw[t, , , ] <- agent$mu_arr
      B_history[t, , , , ] <- agent$B
      sigma[t] <- agent$sigma

      # Belief-weighted V: loop over D and C
      for (d in 1:par$D) {
        for (c in 1:par$C) {
          V_weighted_history[t, d, c] <- sum(V_noisy[d, c, ] * agent$lsb_pre)
        }
      }

      c_trial <- world$c_vec[1, , 1, , t]
      if (is.null(dim(c_trial))) dim(c_trial) <- c(par$C, par$A)
      cue_shown <- rowSums(c_trial) > 0
      for (d in 1:par$D) {
        V_weighted_shown[t, d, cue_shown] <- V_weighted_history[t, d, cue_shown]
      }

      V_cue <- setNames(lapply(1:par$C, function(c) {
        vals <- V_weighted_shown[, , c]  # ntrials x D, may drop to vector if D == 1
        dim(vals) <- c(par$ntrials, par$D)  # always enforce 2D shape
        shown_idx <- !is.na(vals[, 1])
        out <- vals[shown_idx, , drop = FALSE]  # prevent drop at subsetting
        if (length(dim(out)) < 2) dim(out) <- c(sum(shown_idx), par$D)  # fallback guard
        out
      }), par$base_cues)

      # Raw V: loop over D and L
      for (d in 1:par$D) {
        for (l in 1:L) {
          V_raw_history[t, d, , l] <- V_noisy[d, , l]  # C-length vector
        }
      }

      # V_shown: identify the chosen arm's cue segment, then append summed V per dimension
      chosen_arm_seg <- strsplit(type_seq_flat[t], "_")[[1]][agent$arm]
      if (chosen_arm_seg != "0") {
        cue_idx <- cue_type_cue_idx[[chosen_arm_seg]]
        row_d   <- sapply(1:par$D, function(d) sum(V_weighted_history[t, d, cue_idx]))

        # Clamp V_shown values to [-0.5, 0.5]
        row_d <- pmax(-0.5, pmin(0.5, row_d))

        V_shown[[chosen_arm_seg]] <- rbind(V_shown[[chosen_arm_seg]], row_d, deparse.level = 0)
      }

      # V_appear: append this trial's overall value to every base cue whose
      # letters are a subset of the chosen arm's cue segment
      if (chosen_arm_seg != "0") {
        for (bc in par$base_cues) {
          if (chosen_arm_seg %in% appear_ct_map[[bc]]) {
            V_appear[[bc]] <- rbind(V_appear[[bc]], row_d, deparse.level = 0)
          }
        }
      }
    }

    # ============================================================
    # 5.5 Create long format output for ANOVA use and general output
    # ============================================================

    phase_ends_trial <- cumsum(sapply(phase_def, function(ph) sum(unlist(ph$trials))))
    phase_starts_trial <- c(1, phase_ends_trial[-length(phase_ends_trial)] + 1)

    # Build long-format data for this participant using D = 1 only
    long <- vector("list", length(phase_def))
    for (phase_i in seq_along(phase_def)) {

      phase_rows <- do.call(rbind, lapply(all_trial_types, function(ct) {
        v_mat <- V_shown[[ct]]
        if (nrow(v_mat) == 0) return(NULL)

        # Find which trials in type_seq_flat contain this cue type as a chosen arm
        cue_trial_idx <- which(sapply(seq_along(type_seq_flat), function(t_idx) {
          chosen_seg <- strsplit(type_seq_flat[t_idx], "_")[[1]][arm_chosen[t_idx]]
          chosen_seg == ct
        }))

        # Presentation index of the last presentation before this phase starts
        pres_start <- if (phase_i == 1) 1 else
          sum(cue_trial_idx <= phase_starts_trial[phase_i] - 1) + 1

        # Presentation index of the last presentation within this phase
        pres_end <- sum(cue_trial_idx <= phase_ends_trial[phase_i])

        if (pres_start > pres_end) {
          return(NULL)
        }

        v_phase <- v_mat[pres_start:pres_end, 1] # D1 only

        #create the appropriate block vector based on block input
        block_vec <- if (!is.null(block)) {
          if (length(block) == 1) rep(block, length(phase_def)) else block
        } else NULL

        n_pres  <- length(v_phase)

        #(cue trial/total trials) * block size
        pres_per_block <- if (!is.null(block_vec)) {
          (phase_def[[phase_i]]$trials[[ct]] /
             sum(unlist(phase_def[[phase_i]]$trials))) * block_vec[phase_i]
        } else NULL

        block_col <- if (!is.null(block_vec)) {
          factor(rep(seq_len(n_pres / pres_per_block), each = pres_per_block))
        } else NULL

        long_df <- data.frame(
          subject      = factor(i + prev),
          group        = factor(group),
          cue_type     = factor(ct),
          presentation = factor(seq_len(length(v_phase))),
          V            = v_phase
        )

        if (!is.null(block_col)) {
          long_df$block <- block_col
        }

        long_df

      }))

      long[[phase_i]] <- phase_rows
    }


    #Make V into a list format which is more user friendly
    V_list <- setNames(lapply(1:par$C, function(c) {
      mat <- V_weighted_history[, , c]
      dim(mat) <- c(par$ntrials, par$D)
      mat
    }), par$base_cues)

    #Make V_raw into a list format to be more user friendly
    V_raw_list <- setNames(lapply(1:par$C, function(c) {
      mat <- V_raw_history[, , c, ]
      dim(mat) <- c(par$ntrials, par$D, L)
      mat
    }), par$base_cues)

    all_results[[i]] <- list(
      arm_chosen   = arm_chosen,
      win_received = win_received,
      lsb          = lsb,
      Lmax         = Lmax,
      q            = q,
      sigma = sigma,
      mu = mu,
      mu_raw = mu_raw,
      V = V_weighted_history,
      V_list=V_list,
      V_raw_list=V_raw_list,
      V_cue = V_cue,
      V_raw      = V_raw_history,
      V_shown        = V_shown,
      V_appear     = V_appear,
      B            = B_history,
      par                = par,
      world              = world,
      type_seq     = out$type_seq,
      long = long
    )
  }

  # ============================================================
  # 6. Plot V_cue per cue (across participants if n > 1)
  # ============================================================
  base_cues <- par$base_cues
  if (isTRUE(plot) || isTRUE(plot_all)) {
    n_cues <- par$C

    plot_ids <- if (isTRUE(plot_all)) seq_len(n) else 1L

    for (i in plot_ids) {
      for (d in 1:par$D) {

        x_list <- lapply(1:n_cues, function(c) {
          1:all_results[[i]]$par$ntrials
        })

        y_list <- lapply(1:n_cues, function(c) {
          all_results[[i]]$V[, d, c]
        })

        y_all <- unlist(y_list)

        par(mar = c(5, 4, 4, 10))

        plot(NA,
             xlim = c(1, all_results[[i]]$par$ntrials),
             ylim = if (isTRUE(fixed_axis)) c(-0.5, 0.5) else range(y_all, na.rm = TRUE),             xlab = "Trial",
             ylab = "Associative Value (V)",
             xaxt = "n",
             main = paste0("V - D", d, paste0(" - Participant ", i)))

        abline(v = 1:all_results[[i]]$par$ntrials, col = "gray90", lty = 1)

        for (c in 1:n_cues) {
          lines(x_list[[c]], y_list[[c]], col = c)
        }

        # Vertical lines at phase boundaries
        phase_ends <- cumsum(sapply(phase_def, function(ph) sum(unlist(ph$trials))))
        abline(v = phase_ends[-length(phase_ends)] + 0.5, col = "gray40", lty = 2)

        legend(x = par("usr")[2],
               y = par("usr")[4],
               legend = par$base_cues,
               col    = 1:n_cues,
               lty    = 1,
               bty    = "n",
               xpd    = TRUE)

        if (isTRUE(plot_label)) {
          type_seq_flat <- unlist(all_results[[i]]$type_seq)
          win_flat <- all_results[[i]]$win_received  # ntrials x D
          arm_flat <- all_results[[i]]$arm_chosen

          # Color each label by reward on chosen arm
          label_cols <- sapply(1:all_results[[i]]$par$ntrials, function(t) {
            reward <- sum(win_flat[t, ])
            if (reward > 0) "darkgreen" else "red"
          })

          axis(1,
               at       = 1:all_results[[i]]$par$ntrials,
               labels   = type_seq_flat,
               las      = 2,
               cex.axis = 0.6,
               col.axis = "black",
               tick     = TRUE)

          # Recolor each label individually
          for (t in 1:all_results[[i]]$par$ntrials) {
            axis(1,
                 at       = t,
                 labels   = type_seq_flat[t],
                 las      = 2,
                 cex.axis = 0.6,
                 col.axis = label_cols[t],
                 tick     = FALSE)
          }

          if (isTRUE(plot_arm)) {
            mtext(arm_flat,
                  side = 1,
                  at   = 1:all_results[[i]]$par$ntrials,
                  line = 4.5,
                  las  = 2,
                  cex  = 0.5,
                  col  = label_cols)
          }

        } else {
          axis(1, at = seq(0, all_results[[i]]$par$ntrials, by = 10))
          axis(1, at = 1:all_results[[i]]$par$ntrials, labels = FALSE, tcl = -0.3)
        }
      }
    }
  }


  # ============================================================
  # 7. Plot V_cue: per-cue per-dimension, mean across participants
  # ============================================================
  if (isTRUE(plot_cue)) {

    # For each cue, find which presentation indices correspond to phase boundaries
    type_seq_flat <- unlist(all_results[[1]]$type_seq)
    phase_lengths <- sapply(phase_def, function(ph) sum(unlist(ph$trials)))
    phase_ends_trial <- cumsum(phase_lengths)  # trial indices where each phase ends

    for (d in 1:par$D) {
      for (c in 1:par$C) {

        # Collect V_cue[[c]][, d] across all participants, then average
        mat <- sapply(1:n, function(i) all_results[[i]]$V_cue[[c]][, d])
        if (is.null(dim(mat))) dim(mat) <- c(nrow(as.matrix(mat)), n)
        y_mean <- rowMeans(mat, na.rm = TRUE)
        y_sd   <- apply(mat, 1, function(x) sd(x, na.rm = TRUE))

        # Find which presentations of this cue fall at phase boundaries
        # by checking which trial indices (from type_seq) contain this cue
        cue_trial_idx <- which(sapply(type_seq_flat, function(nm) {
          segs <- strsplit(nm, "_")[[1]]
          cue_letters <- strsplit(base_cues[c], "")[[1]]
          any(sapply(segs, function(seg) {
            seg_letters <- strsplit(seg, "")[[1]]
            all(cue_letters %in% seg_letters)
          }))
        }))


        # Phase boundary presentation indices: last presentation before each phase end
        phase_boundary_pres <- sapply(phase_ends_trial[-length(phase_ends_trial)], function(end_t) {
          sum(cue_trial_idx <= end_t)
        })
        phase_boundary_pres <- phase_boundary_pres[phase_boundary_pres > 0 &
                                                     phase_boundary_pres < nrow(mat)]

        y_lo <- y_mean - y_sd
        y_hi <- y_mean + y_sd
        ylim_range <- if (isTRUE(plot_sd) && n > 1) {
          sd_range <- range(c(y_lo, y_hi), na.rm = TRUE)
          if (isTRUE(fixed_axis)) c(min(-0.5, sd_range[1]), max(0.5, sd_range[2])) else sd_range
        } else if (isTRUE(fixed_axis)) c(-0.5, 0.5) else range(mat, na.rm = TRUE)
        plot(NA,
             xlim = c(1, nrow(mat)),
             ylim = ylim_range,
             xlab = "Presentation number",
             ylab = "Associative Value (V)",
             xaxt = "n",
             main = paste0("V cue - D", d, " - Cue ", par$base_cues[c],
                           if (n > 1) paste0(" (mean of ", n, " participants)") else ""))

        # Faint vertical grid lines
        abline(v = 1:nrow(mat), col = "gray90", lty = 1)

        # SE ribbon
        if (isTRUE(plot_sd) && n > 1) {
          xs <- c(1:nrow(mat), nrow(mat):1)
          ys <- c(y_hi, rev(y_lo))
          polygon(xs, ys, col = adjustcolor("black", alpha.f = 0.15), border = NA)
        }

        # Mean line
        lines(1:nrow(mat), y_mean, col = "black", lwd = 2)

        # Phase boundary lines
        if (length(phase_boundary_pres) > 0) {
          abline(v = phase_boundary_pres + 0.5, col = "gray40", lty = 2)
        }

        axis(1, at = seq(0, nrow(mat), by = 10))
        axis(1, at = 1:nrow(mat), labels = FALSE, tcl = -0.3)
      }
    }
  }
  # ============================================================
  # 8. Plot V_raw_history: per-latent-state
  # ============================================================
  if (isTRUE(plot_raw) || isTRUE(plot_raw_all)) {

    plot_ids_raw <- if (isTRUE(plot_raw_all)) seq_len(n) else 1L

    for (i in plot_ids_raw) {

      res      <- all_results[[i]]
      n_states <- max(res$Lmax)
      V_raw    <- res$V_raw      # ntrials x D x C x L

      for (l in 1:n_states) {
        for (d in 1:res$par$D) {

          # Extract V for this state and dimension: ntrials x C
          v_mat <- V_raw[, d, , l]
          if (is.null(dim(v_mat))) dim(v_mat) <- c(res$par$ntrials, res$par$C)

          # Belief proportion for this state across trials
          lsb_state <- res$lsb[, l]

          # Second trial where belief > 0 for this state. V has no effect until
          # the trial after creation, and on the initial trial it still holds
          # its "baseline" value of 0, which is misleading.We therefore do not
          # plot V until the second trial of the state
          first_active <- which(lsb_state > 0)[1] + 1

          y_all <- as.vector(v_mat)

          par(mar = if (isTRUE(plot_label)) c(7, 4, 4, 12) else c(5, 4, 4, 12))

          plot(NA,
               xlim = c(1, res$par$ntrials),
               ylim = if (isTRUE(fixed_axis)) c(-0.5, 0.5) else range(y_all, na.rm = TRUE),               xlab = "Trial",
               ylab = "V (raw)",
               xaxt = "n",
               main = paste0("Latent State ", l, " - D", d, " - Participant ", i))

          abline(v = 1:res$par$ntrials, col = "gray90", lty = 1)

          for (c in 1:res$par$C) {
            idx <- first_active:res$par$ntrials
            lines(idx, v_mat[idx, c], col = c)
          }

          # Belief proportion line on second y-axis
          lsb_scaled <- range(y_all, na.rm = TRUE)[1] +
            lsb_state * diff(range(y_all, na.rm = TRUE))
          lines(1:res$par$ntrials, lsb_scaled, col = "black", lty = 2, lwd = 1.5)
          axis(4, at = range(y_all, na.rm = TRUE)[1] +
                 seq(0, 1, by = 0.25) * diff(range(y_all, na.rm = TRUE)),
               labels = paste0(seq(0, 100, by = 25), "%"),
               las = 2)
          mtext("Belief (%)", side = 4, line = 10, xpd = NA)

          # Vertical lines at phase boundaries
          phase_ends <- cumsum(sapply(phase_def, function(ph) sum(unlist(ph$trials))))
          abline(v = phase_ends[-length(phase_ends)] + 0.5, col = "gray40", lty = 2)

          legend(x = par("usr")[2],
                 y = par("usr")[4],
                 legend = c(res$par$base_cues, "Belief"),
                 col    = c(1:res$par$C, "black"),
                 lty    = c(rep(1, res$par$C), 2),
                 bty    = "n",
                 xpd    = TRUE)

          if (isTRUE(plot_label)) {
            type_seq_flat_raw <- unlist(res$type_seq)
            win_flat <- res$win_received
            arm_flat_raw <- res$arm_chosen

            label_cols_raw <- sapply(1:res$par$ntrials, function(t) {
              reward <- sum(win_flat[t, ])
              if (reward > 0) "darkgreen" else "red"
            })

            axis(1,
                 at       = 1:res$par$ntrials,
                 labels   = type_seq_flat_raw,
                 las      = 2,
                 cex.axis = 0.6,
                 col.axis = "black",
                 tick     = TRUE)

            for (t in 1:res$par$ntrials) {
              axis(1,
                   at       = t,
                   labels   = type_seq_flat_raw[t],
                   las      = 2,
                   cex.axis = 0.6,
                   col.axis = label_cols_raw[t],
                   tick     = FALSE)
            }

            if (isTRUE(plot_arm)) {
              mtext(arm_flat_raw,
                    side = 1,
                    at   = 1:res$par$ntrials,
                    line = 4.5,
                    las  = 2,
                    cex  = 0.5,
                    col  = label_cols_raw)
            }

          } else {
            axis(1, at = seq(0, res$par$ntrials, by = 10))
            axis(1, at = 1:res$par$ntrials, labels = FALSE, tcl = -0.3)
          }
        }
      }
    }
  }

  # ============================================================
  # 9. Plot V_shown: per-cue-type, mean across participants
  # ============================================================
  if (isTRUE(plot_shown)) {

    type_seq_flat <- unlist(all_results[[1]]$type_seq)

    for (ct in all_trial_types) {
      for (d in 1:par$D) {

        # Collect V_shown[[ct]][, d] across all participants
        raw_list <- lapply(1:n, function(i) all_results[[i]]$V_shown[[ct]][, d])

        # Find the expected number of rows (max across participants)
        n_rows <- max(sapply(raw_list, length))

        if (n_rows == 0) {
          warning(paste0("Cue type '", ct, "' is not plotted because it was never chosen."))
          next
        }

        # Pad shorter participants with NA so matrix is rectangular
        mat <- sapply(raw_list, function(x) {
          if (length(x) < n_rows) c(x, rep(NA_real_, n_rows - length(x))) else x
        })
        if (is.null(dim(mat))) dim(mat) <- c(length(mat) / n, n)

        y_mean <- rowMeans(mat, na.rm = TRUE)
        y_sd   <- apply(mat, 1, function(x) sd(x, na.rm = TRUE))

        # Number of participants contributing to each datapoint
        n_contrib <- apply(mat, 1, function(x) sum(!is.na(x)))

        y_lo <- y_mean - y_sd
        y_hi <- y_mean + y_sd
        ylim_range <- if (isTRUE(plot_sd) && n > 1) {
          sd_range <- range(c(y_lo, y_hi), na.rm = TRUE)
          if (isTRUE(fixed_axis)) c(min(-0.5, sd_range[1]), max(0.5, sd_range[2])) else sd_range
        } else if (isTRUE(fixed_axis)) c(-0.5, 0.5) else range(mat, na.rm = TRUE)

        # Add bottom margin for participant counts if multi-arm
        if (par$A > 1) {
          par(mar = c(7, 4, 4, 2))
        } else {
          par(mar = c(5, 4, 4, 2))
        }

        plot(NA,
             xlim = c(1, nrow(mat)),
             ylim = ylim_range,
             xlab = "Presentation number",
             ylab = "Associative Value (V)",
             xaxt = "n",
             main = paste0("V shown - D", d, " - Cue type ", ct,
                           if (n > 1) paste0(" (mean of ", n, " participants)") else ""))

        # Faint vertical grid lines
        abline(v = 1:nrow(mat), col = "gray90", lty = 1)

        # SE ribbon - trim to non-NA range before drawing
        if (isTRUE(plot_sd) && n > 1) {
          valid <- which(!is.na(y_lo) & !is.na(y_hi))
          if (length(valid) > 0) {
            xs <- c(valid, rev(valid))
            ys <- c(y_hi[valid], rev(y_lo[valid]))
            polygon(xs, ys, col = adjustcolor("black", alpha.f = 0.15), border = NA)
          }
        }

        # Mean line
        lines(1:nrow(mat), y_mean, col = "black", lwd = 2)

        # Phase boundary lines
        #find which trials each cue type appears in, considering only the chosen arm
        cue_trial_idx <- which(sapply(seq_along(type_seq_flat), function(t_idx) {
          chosen_seg <- strsplit(type_seq_flat[t_idx], "_")[[1]][all_results[[1]]$arm_chosen[t_idx]]
          chosen_seg == ct
        }))

        phase_ends_trial <- cumsum(sapply(phase_def, function(ph) sum(unlist(ph$trials))))
        phase_boundary_pres <- sapply(phase_ends_trial[-length(phase_ends_trial)], function(end_t) {
          sum(cue_trial_idx <= end_t)
        })
        phase_boundary_pres <- phase_boundary_pres[phase_boundary_pres > 0 &
                                                     phase_boundary_pres < nrow(mat)]
        if (length(phase_boundary_pres) > 0) {
          abline(v = phase_boundary_pres + 0.5, col = "gray40", lty = 2)
        }

        axis(1, at = seq(0, nrow(mat), by = 10))
        axis(1, at = 1:nrow(mat), labels = FALSE, tcl = -0.3)

        # Participant count line below x-axis (multi-arm only)
        if (par$A > 1) {
          mtext(n_contrib,
                side = 1,
                at   = 1:nrow(mat),
                line = 4,
                las  = 2,
                cex  = 0.6)
          mtext("N",
                side = 1,
                at   = 0,
                line = 4,
                las  = 2,
                cex  = 0.6,
                font = 2)
        }
      }
    }
  }

  # ============================================================
  # 10. Plot mu of chosen arm
  # ============================================================
  if (isTRUE(plot_mu) || isTRUE(plot_mu_all)) {

    plot_ids_mu <- if (isTRUE(plot_mu_all)) seq_len(n) else 1L

    for (i in plot_ids_mu) {

      res        <- all_results[[i]]
      type_seq_flat_mu <- unlist(res$type_seq)
      arm_flat   <- res$arm_chosen

      for (d in 1:res$par$D) {

        # Extract mu of chosen arm for each trial: ntrials-length vector
        mu_chosen <- sapply(1:res$par$ntrials, function(t) {
          res$mu[t, d, arm_flat[t]]
        })

        # clamp so that only theoretically meaningful values can be plotted
        mu_chosen <- pmax(-0.5, pmin(0.5, mu_chosen))

        y_all <- mu_chosen

        ylim_range <- if (isTRUE(fixed_axis)) c(-0.5, 0.5) else range(y_all, na.rm = TRUE)

        par(mar = c(5, 4, 4, 4))

        plot(NA,
             xlim = c(1, res$par$ntrials),
             ylim = ylim_range,
             xlab = "Trial",
             ylab = "mu (chosen arm)",
             xaxt = "n",
             main = paste0("mu chosen arm - D", d, " - Participant ", i))

        abline(v = 1:res$par$ntrials, col = "gray90", lty = 1)

        lines(1:res$par$ntrials, mu_chosen, col = "black", lwd = 2)

        # Phase boundary lines
        phase_ends <- cumsum(sapply(phase_def, function(ph) sum(unlist(ph$trials))))
        abline(v = phase_ends[-length(phase_ends)] + 0.5, col = "gray40", lty = 2)

        # plot_label: trial type labels on x-axis, colored by reward
        if (isTRUE(plot_label)) {
          win_flat <- res$win_received  # ntrials x D

          label_cols <- sapply(1:res$par$ntrials, function(t) {
            reward <- sum(win_flat[t, ])
            if (reward > 0) "darkgreen" else "red"
          })

          axis(1,
               at       = 1:res$par$ntrials,
               labels   = type_seq_flat_mu,
               las      = 2,
               cex.axis = 0.6,
               col.axis = "black",
               tick     = TRUE)

          for (t in 1:res$par$ntrials) {
            axis(1,
                 at       = t,
                 labels   = type_seq_flat_mu[t],
                 las      = 2,
                 cex.axis = 0.6,
                 col.axis = label_cols[t],
                 tick     = FALSE)
          }

          if (isTRUE(plot_arm)) {
            mtext(arm_flat,
                  side = 1,
                  at   = 1:res$par$ntrials,
                  line = 4.5,
                  las  = 2,
                  cex  = 0.5,
                  col  = label_cols)
          }

        } else {
          axis(1, at = seq(0, res$par$ntrials, by = 10))
          axis(1, at = 1:res$par$ntrials, labels = FALSE, tcl = -0.3)
        }
      }
    }
  }

  # =================================================================================
  # 11. Plot q across trials, q is plotted after reward, before state creation/reset
  # =================================================================================
  if (isTRUE(plot_q) || isTRUE(plot_q_all)) {

    plot_ids_q <- if (isTRUE(plot_q_all)) seq_len(n) else 1L

    for (i in plot_ids_q) {
      res <- all_results[[i]]

      q_vec   <- res$q

      # simple plotting-only cap for infinities
      inf_trials <- which(is.infinite(q_vec))

      if (any(is.infinite(q_vec))) {
        q_vec[is.infinite(q_vec) & q_vec > 0] <-  as.numeric(eta)*2
        q_vec[is.infinite(q_vec) & q_vec < 0] <-  as.numeric(eta)*2
      }

      ntr     <- res$par$ntrials
      eta_val <- res$par$ls$eta

      # y-lim from 0 to max(q, eta), with small headroom
      q_max <- max(q_vec, eta_val, na.rm = TRUE)
      y_pad <- 0.05 * (q_max + 1e-8)
      ylim_range <- c(0, q_max + y_pad)

      type_seq_flat_q <- unlist(res$type_seq)
      win_flat_q      <- res$win_received

      par(mar = if (isTRUE(plot_label)) c(7, 4, 4, 2) else c(5, 4, 4, 2))

      plot(NA,
           xlim = c(1, ntr),
           ylim = ylim_range,
           xlab = "Trial",
           ylab = "q (change-point statistic)",
           xaxt = "n",
           main = paste0("q - Participant ", i))

      # Grid
      abline(v = 1:ntr, col = "gray90", lty = 1)

      # q trajectory
      lines(1:ntr, q_vec, col = "black", lwd = 2)

      # Mark trials where q was Inf (true value arbitrarily large, capped at
      # eta*2 for display)
      if (length(inf_trials) > 0) {
        points(inf_trials, q_vec[inf_trials],
               pch = 17, col = "red", cex = 1.2)
        text(inf_trials, q_vec[inf_trials],
             labels = "\u221e",
             pos = 3, col = "red", cex = 0.9)
        legend("topleft",
               legend = "q = \u221e (capped at 2\u00d7\u03b7 for display)",
               pch = 17, col = "red", bty = "n", cex = 0.8)
      }

      # Threshold eta
      abline(h = eta_val, col = "red", lty = 2, lwd = 1.5)

      # Phase boundaries
      phase_ends <- cumsum(sapply(phase_def, function(ph) sum(unlist(ph$trials))))
      if (length(phase_ends) > 1) {
        abline(v = phase_ends[-length(phase_ends)] + 0.5, col = "gray40", lty = 2)
      }

      # X‑axis labels with trial types / reward coloring, mirroring other plots
      if (isTRUE(plot_label)) {

        label_cols <- sapply(1:ntr, function(t) {
          reward <- sum(win_flat_q[t, ])
          if (reward > 0) "darkgreen" else "red"
        })

        axis(1,
             at       = 1:ntr,
             labels   = type_seq_flat_q,
             las      = 2,
             cex.axis = 0.6,
             col.axis = "black",
             tick     = TRUE)

        # Recolor each label individually
        for (t in 1:ntr) {
          axis(1,
               at       = t,
               labels   = type_seq_flat_q[t],
               las      = 2,
               cex.axis = 0.6,
               col.axis = label_cols[t],
               tick     = FALSE)
        }

        if (isTRUE(plot_arm)) {
          mtext(res$arm_chosen,
                side = 1,
                at   = 1:ntr,
                line = 4.5,
                las  = 2,
                cex  = 0.5,
                col  = label_cols)
        }

      } else {
        axis(1, at = seq(0, ntr, by = 10))
        axis(1, at = 1:ntr, labels = FALSE, tcl = -0.3)
      }
    }
  }

  # ============================================================
  # 12. Plot sigma (at start of each trial)
  # ============================================================
  if (isTRUE(plot_sigma) || isTRUE(plot_sigma_all)) {

    plot_ids_sigma <- if (isTRUE(plot_sigma_all)) seq_len(n) else 1L

    # Common y‑axis if fixed_axis = TRUE
    if (isTRUE(fixed_axis)) {
      sigma_all <- unlist(lapply(plot_ids_sigma, function(i) all_results[[i]]$sigma))
      ylim_sigma <- range(sigma_all, na.rm = TRUE)
      if (!all(is.finite(ylim_sigma))) {
        ylim_sigma <- c(0, 1)
      } else if (diff(ylim_sigma) == 0) {
        pad <- if (ylim_sigma[1] == 0) 1 else abs(ylim_sigma[1]) * 0.1
        ylim_sigma <- ylim_sigma + c(-pad, pad)
      }
    }

    for (i in plot_ids_sigma) {
      res        <- all_results[[i]]
      sigma_vec  <- res$sigma
      ntr        <- res$par$ntrials
      type_seq_flat <- unlist(res$type_seq)
      win_flat   <- res$win_received

      ylim_range <- if (isTRUE(fixed_axis)) {
        ylim_sigma
      } else {
        rng <- range(sigma_vec, na.rm = TRUE)
        if (!all(is.finite(rng)) || diff(rng) == 0) {
          c(0, 1)
        } else rng
      }

      par(mar = if (isTRUE(plot_label)) c(7, 4, 4, 2) else c(5, 4, 4, 2))

      plot(NA,
           xlim = c(1, ntr),
           ylim = ylim_range,
           xlab = "Trial",
           ylab = "sigma (start of trial)",
           xaxt = "n",
           main = paste0("sigma - Participant ", i))

      # Grid
      abline(v = 1:ntr, col = "gray90", lty = 1)

      # sigma trajectory
      lines(1:ntr, sigma_vec, col = "black", lwd = 2)

      # Phase boundaries
      phase_ends <- cumsum(sapply(phase_def, function(ph) sum(unlist(ph$trials))))
      if (length(phase_ends) > 1) {
        abline(v = phase_ends[-length(phase_ends)] + 0.5, col = "gray40", lty = 2)
      }

      # X-axis labels (compatible with plot_label / plot_arm)
      if (isTRUE(plot_label)) {
        label_cols <- sapply(1:ntr, function(t) {
          reward <- sum(win_flat[t, ])
          if (reward > 0) "darkgreen" else "red"
        })

        axis(1,
             at       = 1:ntr,
             labels   = type_seq_flat,
             las      = 2,
             cex.axis = 0.6,
             col.axis = "black",
             tick     = TRUE)

        for (t in 1:ntr) {
          axis(1,
               at       = t,
               labels   = type_seq_flat[t],
               las      = 2,
               cex.axis = 0.6,
               col.axis = label_cols[t],
               tick     = FALSE)
        }

        if (isTRUE(plot_arm)) {
          mtext(res$arm_chosen,
                side = 1,
                at   = 1:ntr,
                line = 4.5,
                las  = 2,
                cex  = 0.5,
                col  = label_cols)
        }
      } else {
        axis(1, at = seq(0, ntr, by = 10))
        axis(1, at = 1:ntr, labels = FALSE, tcl = -0.3)
      }
    }
  }

  # ============================================================
  # 13. Plot V_appear: per base cue, mean across participants
  # ============================================================
  if (isTRUE(plot_appears)) {

    type_seq_flat <- unlist(all_results[[1]]$type_seq)

    for (bc in base_cues) {
      for (d in 1:par$D) {

        # Collect V_appear[[bc]][, d] across all participants
        raw_list <- lapply(1:n, function(i) all_results[[i]]$V_appear[[bc]][, d])

        # Find the expected number of rows (max across participants)
        n_rows <- max(sapply(raw_list, length))

        if (n_rows == 0) {
          warning(paste0("Base cue '", bc, "' is not plotted because it never appeared."))
          next
        }

        # Pad shorter participants with NA so matrix is rectangular
        mat <- sapply(raw_list, function(x) {
          if (length(x) < n_rows) c(x, rep(NA_real_, n_rows - length(x))) else x
        })
        if (is.null(dim(mat))) dim(mat) <- c(length(mat) / n, n)

        y_mean <- rowMeans(mat, na.rm = TRUE)
        y_sd   <- apply(mat, 1, function(x) sd(x, na.rm = TRUE))

        # Number of participants contributing to each datapoint
        n_contrib <- apply(mat, 1, function(x) sum(!is.na(x)))

        y_lo <- y_mean - y_sd
        y_hi <- y_mean + y_sd
        ylim_range <- if (isTRUE(plot_sd) && n > 1) {
          sd_range <- range(c(y_lo, y_hi), na.rm = TRUE)
          if (isTRUE(fixed_axis)) c(min(-0.5, sd_range[1]), max(0.5, sd_range[2])) else sd_range
        } else if (isTRUE(fixed_axis)) c(-0.5, 0.5) else range(mat, na.rm = TRUE)

        # Add bottom margin for participant counts if multi-arm
        if (par$A > 1) {
          par(mar = c(7, 4, 4, 2))
        } else {
          par(mar = c(5, 4, 4, 2))
        }

        plot(NA,
             xlim = c(1, nrow(mat)),
             ylim = ylim_range,
             xlab = "Presentation number",
             ylab = "Associative Value (V)",
             xaxt = "n",
             main = paste0("V appears - D", d, " - Base cue ", bc,
                           if (n > 1) paste0(" (mean of ", n, " participants)") else ""))

        # Faint vertical grid lines
        abline(v = 1:nrow(mat), col = "gray90", lty = 1)

        # SE ribbon - trim to non-NA range before drawing
        if (isTRUE(plot_sd) && n > 1) {
          valid <- which(!is.na(y_lo) & !is.na(y_hi))
          if (length(valid) > 0) {
            xs <- c(valid, rev(valid))
            ys <- c(y_hi[valid], rev(y_lo[valid]))
            polygon(xs, ys, col = adjustcolor("black", alpha.f = 0.15), border = NA)
          }
        }

        # Mean line
        lines(1:nrow(mat), y_mean, col = "black", lwd = 2)

        # Phase boundary lines
        # find which trials this base cue appears in, considering only the chosen arm
        cue_trial_idx <- which(sapply(seq_along(type_seq_flat), function(t_idx) {
          chosen_seg <- strsplit(type_seq_flat[t_idx], "_")[[1]][all_results[[1]]$arm_chosen[t_idx]]
          chosen_seg %in% appear_ct_map[[bc]]
        }))

        phase_ends_trial <- cumsum(sapply(phase_def, function(ph) sum(unlist(ph$trials))))
        phase_boundary_pres <- sapply(phase_ends_trial[-length(phase_ends_trial)], function(end_t) {
          sum(cue_trial_idx <= end_t)
        })
        phase_boundary_pres <- phase_boundary_pres[phase_boundary_pres > 0 &
                                                     phase_boundary_pres < nrow(mat)]
        if (length(phase_boundary_pres) > 0) {
          abline(v = phase_boundary_pres + 0.5, col = "gray40", lty = 2)
        }

        axis(1, at = seq(0, nrow(mat), by = 10))
        axis(1, at = 1:nrow(mat), labels = FALSE, tcl = -0.3)

        # Participant count line below x-axis (multi-arm only)
        if (par$A > 1) {
          mtext(n_contrib,
                side = 1,
                at   = 1:nrow(mat),
                line = 4,
                las  = 2,
                cex  = 0.6)
          mtext("N",
                side = 1,
                at   = 0,
                line = 4,
                las  = 2,
                cex  = 0.6,
                font = 2)
        }
      }
    }
  }

  all_results <- lapply(all_results, function(res) {
    res$V        <- res$V_list
    res$V_raw    <- res$V_raw_list
    res$V_list   <- NULL
    res$V_raw_list <- NULL
    res$type_seq <- NULL
    res
  })

  return(all_results)
}

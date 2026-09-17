#' Compare latent-state learning results across groups
#'
#' Overlays results from multiple \code{run()} calls (e.g. different
#' groups, conditions, or parameter settings) on the same plots, allowing
#' direct visual comparison. Cues are distinguished by color and groups by
#' line type (or by color alone, for the presentation-based plots).
#'
#' @param ... One or more outputs of \code{run()}, i.e. lists of
#'   participant results, passed as named arguments (e.g. \code{ctrl =
#'   result1, treatment = result2}). The names become group labels used in
#'   plot legends; if left unnamed, groups are labeled \code{"group1"},
#'   \code{"group2"}, etc. All groups must share the same task structure
#'   (\code{par$D}, \code{par$C}, \code{par$base_cues}, and trial/phase
#'   layout), since the first group's structure is used as the reference
#'   for all plots.
#'
#' @param plot Logical. If \code{TRUE}, plots V across trials for
#'   participant 1 of each group, overlaid on the same axes -- cues are
#'   colored, groups are distinguished by line type. Combine with
#'   \code{plot_label} to add trial-type labels (colored by reward) on the
#'   x-axis, using the reference group's trial sequence. Defaults to
#'   \code{FALSE}.
#'
#' @param plot_avg Logical. If \code{TRUE}, plots the mean V across
#'   participants within each group, across trials, overlaid on the same
#'   axes -- cues are colored, groups are distinguished by line type.
#'   Combine with \code{plot_sd} to add a shaded standard-deviation ribbon
#'   per group. Defaults to \code{FALSE}.
#'
#' @param fixed_axis Logical. If \code{TRUE}, y-axes are fixed to
#'   \code{[-0.5, 0.5]},
#'   keeping plots comparable across cues and groups. If \code{FALSE}, each
#'   plot's y-axis is scaled to its own data range. Defaults to \code{TRUE}.
#'
#' @param plot_sd Logical. If \code{TRUE}, adds a shaded ribbon showing
#'   +/- 1 SD around each group's mean line, for any group with more than
#'   one participant. Applies to \code{plot_avg}, \code{plot_cue},
#'   \code{plot_shown}, and \code{plot_appear}. Defaults to \code{FALSE}.
#'
#' @param plot_label Logical. If \code{TRUE}, adds trial-type labels
#'   (colored by reward) along the x-axis of the \code{plot} output, taken
#'   from the reference group's trial sequence. Requires every participant
#'   in every group to share an identical trial sequence. Defaults to \code{FALSE}.
#'
#' @param plot_shown Logical. If \code{TRUE}, plots V by presentation
#'   number for each distinct trial-type cue combination (e.g. \code{"A"},
#'   \code{"AB"}), comparing the mean across participants for each group on
#'   the same axes, colored by group. Combine with \code{plot_sd} to add a
#'   shaded standard-deviation ribbon. Defaults to \code{FALSE}.
#'
#' @param plot_cue Logical. If \code{TRUE}, plots each individual base
#'   cue's own value by presentation number, pooling every trial type that
#'   cue appears in, comparing the mean across participants for each group
#'   on the same axes, colored by group. Combine with \code{plot_sd} to add
#'   a shaded standard-deviation ribbon. Defaults to \code{FALSE}.
#'
#' @param plot_appear Logical. If \code{TRUE}, plots the total value of the
#'   chosen arm by presentation number for each base cue, restricted to
#'   presentations where that cue was part of the chosen arm, comparing the
#'   mean across participants for each group on the same axes, colored by
#'   group. Combine with \code{plot_sd} to add a shaded standard-deviation
#'   ribbon. Defaults to \code{FALSE}.
#'
#' @return No return value. Called for its side effect of producing one or
#'   more base R plots comparing latent-state values across the supplied
#'   groups. Invisibly returns \code{NULL}.
#'
#' @examples
#' phase_def <- list(phase1 = list(trials = list(A = 20, B = 20)))
#' ctrl      <- run(phase_def, n = 5, alpha0 = "0.1")
#' treatment <- run(phase_def, n = 5, alpha0 = "0.3")
#' plot_compare(ctrl = ctrl, treatment = treatment, plot_avg = TRUE)
#'
#' @export
plot_compare <- function(...,
                         plot = FALSE,
                         plot_avg = FALSE,
                         fixed_axis = TRUE,
                         plot_sd = FALSE,
                         plot_label = FALSE,
                         plot_shown = FALSE,
                         plot_cue = FALSE,
                         plot_appear = FALSE) {

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  groups     <- list(...)
  group_names <- names(groups)
  if (is.null(group_names)) group_names <- paste0("group", seq_along(groups))

  n_groups <- length(groups)
  colors   <- seq_len(n_groups)

  # Derive shared structure from first group
  first   <- groups[[1]][[1]]
  par     <- first$par
  ntrials <- first$par$ntrials
  phase_ends <- cumsum(sapply(first$type_seq, function(ph) length(unlist(ph))))

  # Recompute phase ends from type_seq
  phase_ends <- cumsum(sapply(first$type_seq, function(ph) length(unlist(ph))))

  # Validate if type sequences match across all groups if using plot_label
  if (isTRUE(plot_label)) {
    ref_seq <- unlist(first$type_seq)
    for (g in seq_len(n_groups)) {
      for (j in seq_along(groups[[g]])) {
        cmp_seq <- unlist(groups[[g]][[j]]$type_seq)
        if (!identical(ref_seq, cmp_seq)) stop(paste0(
          "plot_label = TRUE requires all groups to have identical type sequences, ",
          "but group '", group_names[g], "', participant ", j, " differs from the reference."
        ))
      }
    }
  }

  # Helper: compute mean V across participants in one run result list
  mean_V <- function(res_list, d, c) {
    mat <- sapply(res_list, function(res) res$V[[c]][, d])
    if (is.null(dim(mat))) dim(mat) <- c(length(mat), 1)
    list(
      mean = rowMeans(mat, na.rm = TRUE),
      sd   = apply(mat, 1, function(x) sd(x, na.rm = TRUE)),
      n    = length(res_list)
    )
  }

  # ============================================================
  # plot=TRUE equivalent: V of participant 1 per group
  # ============================================================
  if (plot == TRUE) {
    for (d in 1:par$D) {

      all_vals <- unlist(lapply(seq_len(n_groups), function(g) {
        lapply(1:par$C, function(c) groups[[g]][[1]]$V[[c]][, d])
      }))

      ylim_range <- if (isTRUE(fixed_axis)) c(-0.5, 0.5) else range(all_vals, na.rm = TRUE)

      par(mar = c(5, 4, 4, 12))

      plot(NA,
           xlim = c(1, ntrials),
           ylim = ylim_range,
           xlab = "Trial",
           ylab = "V",
           xaxt = "n",
           main = paste0("V - D", d, " - Participant 1"))

      abline(v = 1:ntrials, col = "gray90", lty = 1)
      abline(v = phase_ends[-length(phase_ends)] + 0.5, col = "gray40", lty = 2)

      for (c in 1:par$C) {
        for (g in seq_len(n_groups)) {
          lines(1:ntrials, groups[[g]][[1]]$V[[c]][, d], col = c, lty = g, lwd = 2)
        }
      }

      legend(x = par("usr")[2],
             y = par("usr")[4],
             legend = c(first$par$base_cues, group_names),
             col    = c(1:par$C, rep("black", n_groups)),
             lty    = c(rep(1, par$C), seq_len(n_groups)),
             lwd    = 2,
             bty    = "n",
             xpd    = TRUE)

      if (isTRUE(plot_label)) {
        type_seq_flat <- unlist(first$type_seq)
        win_flat      <- first$win_received
        arm_flat      <- first$arm_chosen

        label_cols <- sapply(1:ntrials, function(t) {
          reward <- sum(win_flat[t, ])
          if (reward > 0) "darkgreen" else "red"
        })

        axis(1,
             at       = 1:ntrials,
             labels   = type_seq_flat,
             las      = 2,
             cex.axis = 0.6,
             col.axis = "black",
             tick     = TRUE)

        for (t in 1:ntrials) {
          axis(1,
               at       = t,
               labels   = type_seq_flat[t],
               las      = 2,
               cex.axis = 0.6,
               col.axis = label_cols[t],
               tick     = FALSE)
        }
      } else {
        axis(1, at = seq(0, ntrials, by = 10))
        axis(1, at = 1:ntrials, labels = FALSE, tcl = -0.3)
      }
    }
  }

  # ============================================================
  # plot_avg=TRUE: mean V across participants per group
  # ============================================================
  if (isTRUE(plot_avg)) {
    for (d in 1:par$D) {

      all_means <- lapply(seq_len(par$C), function(c) {
        lapply(seq_len(n_groups), function(g) mean_V(groups[[g]], d, c))
      })

      if (isTRUE(fixed_axis)) {
        ylim_range <- c(-0.5, 0.5)
        if (isTRUE(plot_sd)) {
          all_vals <- unlist(lapply(all_means, function(cue) lapply(cue, function(m) c(m$mean + m$sd, m$mean - m$sd))))
          sd_range <- range(all_vals, na.rm = TRUE)
          ylim_range <- c(min(ylim_range[1], sd_range[1]), max(ylim_range[2], sd_range[2]))
        }
      } else {
        all_vals <- unlist(lapply(all_means, function(cue) lapply(cue, function(m) {
          if (isTRUE(plot_sd) && m$n > 1) c(m$mean + m$sd, m$mean - m$sd) else m$mean
        })))
        ylim_range <- range(all_vals, na.rm = TRUE)
      }

      par(mar = c(5, 4, 4, 12))

      plot(NA,
           xlim = c(1, ntrials),
           ylim = ylim_range,
           xlab = "Trial",
           ylab = "V",
           xaxt = "n",
           main = paste0("V - D", d, " (group means)"))

      abline(v = 1:ntrials, col = "gray90", lty = 1)
      abline(v = phase_ends[-length(phase_ends)] + 0.5, col = "gray40", lty = 2)

      for (c in 1:par$C) {
        for (g in seq_len(n_groups)) {
          m <- all_means[[c]][[g]]

          if (isTRUE(plot_sd) && m$n > 1) {
            xs <- c(1:ntrials, ntrials:1)
            ys <- c(m$mean + m$sd, rev(m$mean - m$sd))
            polygon(xs, ys, col = adjustcolor(colors[g], alpha.f = 0.15), border = NA)
          }

          lines(1:ntrials, m$mean, col = c, lty = g, lwd = 2)
        }
      }

      legend(x = par("usr")[2],
             y = par("usr")[4],
             legend = c(first$par$base_cues, group_names),
             col    = c(1:par$C, rep("black", n_groups)),
             lty    = c(rep(1, par$C), seq_len(n_groups)),
             lwd    = 2,
             bty    = "n",
             xpd    = TRUE)

      axis(1, at = seq(0, ntrials, by = 10))
      axis(1, at = 1:ntrials, labels = FALSE, tcl = -0.3)
    }
  }


  # ============================================================
  # plot_cue equivalent: V per presentation of each cue
  # ============================================================
  if (isTRUE(plot_cue)) {

    type_seq_flat <- unlist(first$type_seq)
    base_cues     <- first$par$base_cues

    for (d in 1:par$D) {
      for (c in 1:par$C) {

        all_shown <- lapply(seq_len(n_groups), function(g) {
          mat <- sapply(groups[[g]], function(res) res$V_cue[[c]][, d])
          if (is.null(dim(mat))) dim(mat) <- c(length(mat), 1)
          list(
            mean = rowMeans(mat, na.rm = TRUE),
            sd   = apply(mat, 1, function(x) sd(x, na.rm = TRUE)),
            n    = length(groups[[g]])
          )
        })

        n_pres <- length(all_shown[[1]]$mean)

        if (isTRUE(fixed_axis)) {
          ylim_range <- c(-0.5, 0.5)
          if (isTRUE(plot_sd)) {
            all_vals <- unlist(lapply(all_shown, function(m) c(m$mean + m$sd, m$mean - m$sd)))
            sd_range <- range(all_vals, na.rm = TRUE)
            ylim_range <- c(min(ylim_range[1], sd_range[1]), max(ylim_range[2], sd_range[2]))
          }
        } else {
          all_vals <- unlist(lapply(all_shown, function(m) {
            if (isTRUE(plot_sd) && m$n > 1) c(m$mean + m$sd, m$mean - m$sd) else m$mean
          }))
          ylim_range <- range(all_vals, na.rm = TRUE)
        }

        par(mar = c(5, 4, 4, 12))

        plot(NA,
             xlim = c(1, n_pres),
             ylim = ylim_range,
             xlab = "Presentation number",
             ylab = "V",
             xaxt = "n",
             main = paste0("V cue - D", d, " - Cue ", first$par$base_cues[c]))

        abline(v = 1:n_pres, col = "gray90", lty = 1)

        # Phase boundary presentations for this cue
        cue_trial_idx <- which(sapply(type_seq_flat, function(nm) {
          segs <- strsplit(nm, "_")[[1]]
          cue_letters <- strsplit(base_cues[c], "")[[1]]
          any(sapply(segs, function(seg) {
            seg_letters <- strsplit(seg, "")[[1]]
            all(cue_letters %in% seg_letters)
          }))
        }))

        phase_ends_trial <- cumsum(sapply(first$type_seq, function(ph) length(unlist(ph))))
        phase_boundary_pres <- cumsum(sapply(seq_along(phase_ends_trial), function(ph) {
          t_start <- if (ph == 1) 1 else phase_ends_trial[ph - 1] + 1
          t_end   <- phase_ends_trial[ph]
          sum(cue_trial_idx >= t_start & cue_trial_idx <= t_end)
        }))
        phase_boundary_pres <- phase_boundary_pres[-length(phase_boundary_pres)]
        phase_boundary_pres <- phase_boundary_pres[phase_boundary_pres > 0 &
                                                     phase_boundary_pres < n_pres]

        if (length(phase_boundary_pres) > 0) {
          abline(v = phase_boundary_pres + 0.5, col = "gray40", lty = 2)
        }

        for (g in seq_len(n_groups)) {
          m <- all_shown[[g]]

          if (isTRUE(plot_sd) && m$n > 1) {
            xs <- c(1:n_pres, n_pres:1)
            ys <- c(m$mean + m$sd, rev(m$mean - m$sd))
            polygon(xs, ys, col = adjustcolor(colors[g], alpha.f = 0.15), border = NA)
          }

          lines(1:n_pres, m$mean, col = colors[g], lwd = 2)
        }

        legend(x = par("usr")[2],
               y = par("usr")[4],
               legend = group_names,
               col    = colors,
               lty    = 1,
               lwd    = 2,
               bty    = "n",
               xpd    = TRUE)

        axis(1, at = seq(0, n_pres, by = 10))
        axis(1, at = 1:n_pres, labels = FALSE, tcl = -0.3)
      }
    }
  }

  # ============================================================
  # plot_shown equivalent: V_shown per presentation of each cue type
  # ============================================================
  if (isTRUE(plot_shown)) {

    all_cue_types <- names(first$V_shown)

    for (ct in all_cue_types) {
      for (d in 1:par$D) {

        all_cue <- lapply(seq_len(n_groups), function(g) {
          raw_list <- lapply(groups[[g]], function(res) res$V_shown[[ct]][, d])
          n_rows   <- max(sapply(raw_list, length))
          if (n_rows == 0) return(NULL)
          mat <- sapply(raw_list, function(x) {
            if (length(x) < n_rows) c(x, rep(NA_real_, n_rows - length(x))) else x
          })
          if (is.null(dim(mat))) dim(mat) <- c(n_rows, 1)
          list(
            mean = rowMeans(mat, na.rm = TRUE),
            sd   = apply(mat, 1, function(x) sd(x, na.rm = TRUE)),
            n    = length(groups[[g]])
          )
        })

        if (any(sapply(all_cue, is.null))) next

        n_pres <- length(all_cue[[1]]$mean)

        if (isTRUE(fixed_axis)) {
          ylim_range <- c(-0.5, 0.5)
          if (isTRUE(plot_sd)) {
            all_vals <- unlist(lapply(all_cue, function(m) {
              if (m$n > 1) c(m$mean + m$sd, m$mean - m$sd) else m$mean
            }))
            sd_range <- range(all_vals, na.rm = TRUE)
            ylim_range <- c(min(-0.5, sd_range[1]), max(0.5, sd_range[2]))
          }
        } else {
          all_vals <- unlist(lapply(all_cue, function(m) {
            if (isTRUE(plot_sd) && m$n > 1) c(m$mean + m$sd, m$mean - m$sd) else m$mean
          }))
          ylim_range <- range(all_vals, na.rm = TRUE)
        }

        par(mar = c(5, 4, 4, 12))

        plot(NA,
             xlim = c(1, n_pres),
             ylim = ylim_range,
             xlab = "Presentation number",
             ylab = "V",
             xaxt = "n",
             main = paste0("V shown - D", d, " - Cue type ", ct))

        type_seq_flat <- unlist(first$type_seq)
        cue_trial_idx <- which(sapply(seq_along(type_seq_flat), function(t_idx) {
          chosen_seg <- strsplit(type_seq_flat[t_idx], "_")[[1]][first$arm_chosen[t_idx]]
          chosen_seg == ct
        }))
        phase_ends_trial <- cumsum(sapply(first$type_seq, function(ph) length(unlist(ph))))
        phase_boundary_pres <- cumsum(sapply(seq_along(phase_ends_trial), function(ph) {
          t_start <- if (ph == 1) 1 else phase_ends_trial[ph - 1] + 1
          t_end   <- phase_ends_trial[ph]
          sum(cue_trial_idx >= t_start & cue_trial_idx <= t_end)
        }))
        phase_boundary_pres <- phase_boundary_pres[-length(phase_boundary_pres)]
        phase_boundary_pres <- phase_boundary_pres[phase_boundary_pres > 0 &
                                                     phase_boundary_pres < n_pres]
        if (length(phase_boundary_pres) > 0) {
          abline(v = phase_boundary_pres + 0.5, col = "gray40", lty = 2)
        }

        abline(v = 1:n_pres, col = "gray90", lty = 1)

        for (g in seq_len(n_groups)) {
          m <- all_cue[[g]]

          if (isTRUE(plot_sd) && m$n > 1) {
            valid <- which(!is.na(m$mean + m$sd))
            if (length(valid) > 0) {
              xs <- c(valid, rev(valid))
              ys <- c((m$mean + m$sd)[valid], rev((m$mean - m$sd)[valid]))
              polygon(xs, ys, col = adjustcolor(colors[g], alpha.f = 0.15), border = NA)
            }
          }

          lines(1:n_pres, m$mean, col = colors[g], lwd = 2)
        }

        legend(x = par("usr")[2],
               y = par("usr")[4],
               legend = group_names,
               col    = colors,
               lty    = 1,
               lwd    = 2,
               bty    = "n",
               xpd    = TRUE)

        axis(1, at = seq(0, n_pres, by = 10))
        axis(1, at = 1:n_pres, labels = FALSE, tcl = -0.3)
      }
    }
  }

  # ============================================================
  # plot_appear equivalent: V_appear per presentation of each base cue
  # ============================================================
  if (isTRUE(plot_appear)) {

    all_base_cues <- names(first$V_appear)

    # A base cue "appears" in a chosen arm segment whenever all of the
    # base cue's letters are present in that segment's letters (same rule
    # used to build V_appear during run()).
    contains_bc <- function(bc, seg) {
      bc_letters  <- strsplit(bc,  "")[[1]]
      seg_letters <- strsplit(seg, "")[[1]]
      all(bc_letters %in% seg_letters)
    }

    for (bc in all_base_cues) {
      for (d in 1:par$D) {

        all_appear <- lapply(seq_len(n_groups), function(g) {
          raw_list <- lapply(groups[[g]], function(res) res$V_appear[[bc]][, d])
          n_rows   <- max(sapply(raw_list, length))
          if (n_rows == 0) return(NULL)
          mat <- sapply(raw_list, function(x) {
            if (length(x) < n_rows) c(x, rep(NA_real_, n_rows - length(x))) else x
          })
          if (is.null(dim(mat))) dim(mat) <- c(n_rows, 1)
          list(
            mean = rowMeans(mat, na.rm = TRUE),
            sd   = apply(mat, 1, function(x) sd(x, na.rm = TRUE)),
            n    = length(groups[[g]])
          )
        })

        if (any(sapply(all_appear, is.null))) next

        n_pres <- length(all_appear[[1]]$mean)

        if (isTRUE(fixed_axis)) {
          ylim_range <- c(-0.5, 0.5)
          if (isTRUE(plot_sd)) {
            all_vals <- unlist(lapply(all_appear, function(m) {
              if (m$n > 1) c(m$mean + m$sd, m$mean - m$sd) else m$mean
            }))
            sd_range <- range(all_vals, na.rm = TRUE)
            ylim_range <- c(min(-0.5, sd_range[1]), max(0.5, sd_range[2]))
          }
        } else {
          all_vals <- unlist(lapply(all_appear, function(m) {
            if (isTRUE(plot_sd) && m$n > 1) c(m$mean + m$sd, m$mean - m$sd) else m$mean
          }))
          ylim_range <- range(all_vals, na.rm = TRUE)
        }

        par(mar = c(5, 4, 4, 12))

        plot(NA,
             xlim = c(1, n_pres),
             ylim = ylim_range,
             xlab = "Presentation number",
             ylab = "V",
             xaxt = "n",
             main = paste0("V appear - D", d, " - Base cue ", bc))

        type_seq_flat <- unlist(first$type_seq)
        cue_trial_idx <- which(sapply(seq_along(type_seq_flat), function(t_idx) {
          chosen_seg <- strsplit(type_seq_flat[t_idx], "_")[[1]][first$arm_chosen[t_idx]]
          chosen_seg != "0" && contains_bc(bc, chosen_seg)
        }))
        phase_ends_trial <- cumsum(sapply(first$type_seq, function(ph) length(unlist(ph))))
        phase_boundary_pres <- cumsum(sapply(seq_along(phase_ends_trial), function(ph) {
          t_start <- if (ph == 1) 1 else phase_ends_trial[ph - 1] + 1
          t_end   <- phase_ends_trial[ph]
          sum(cue_trial_idx >= t_start & cue_trial_idx <= t_end)
        }))
        phase_boundary_pres <- phase_boundary_pres[-length(phase_boundary_pres)]
        phase_boundary_pres <- phase_boundary_pres[phase_boundary_pres > 0 &
                                                     phase_boundary_pres < n_pres]
        if (length(phase_boundary_pres) > 0) {
          abline(v = phase_boundary_pres + 0.5, col = "gray40", lty = 2)
        }

        abline(v = 1:n_pres, col = "gray90", lty = 1)

        for (g in seq_len(n_groups)) {
          m <- all_appear[[g]]

          if (isTRUE(plot_sd) && m$n > 1) {
            valid <- which(!is.na(m$mean + m$sd))
            if (length(valid) > 0) {
              xs <- c(valid, rev(valid))
              ys <- c((m$mean + m$sd)[valid], rev((m$mean - m$sd)[valid]))
              polygon(xs, ys, col = adjustcolor(colors[g], alpha.f = 0.15), border = NA)
            }
          }

          lines(1:n_pres, m$mean, col = colors[g], lwd = 2)
        }

        legend(x = par("usr")[2],
               y = par("usr")[4],
               legend = group_names,
               col    = colors,
               lty    = 1,
               lwd    = 2,
               bty    = "n",
               xpd    = TRUE)

        axis(1, at = seq(0, n_pres, by = 10))
        axis(1, at = 1:n_pres, labels = FALSE, tcl = -0.3)
      }
    }
  }

  invisible(NULL)
}

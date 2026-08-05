#' Print a discriminant_projector
#'
#' Compact, uniform display for any `discriminant_projector` returned by the
#' discrimination analysis methods in `discursive`.
#'
#' @param x A `discriminant_projector` object.
#' @param ... Currently ignored.
#'
#' @return The input `x`, invisibly.
#' @method print discriminant_projector
#' @export
print.discriminant_projector <- function(x, ...) {
  cat(sprintf("<discriminant_projector: %s>\n", .projector_method(x)))
  cat(sprintf("  Input dim:  %s features\n",   .fmt_int(.dim_in(x))))
  cat(sprintf("  Output dim: %s components\n", .fmt_int(.dim_out(x))))
  cat(sprintf("  Samples:    %s\n",            .fmt_int(.n_samples(x))))
  nc <- .projector_nclasses(x)
  if (!is.na(nc)) cat(sprintf("  Classes:    %d\n", nc))
  invisible(x)
}

#' Summarise a discriminant_projector
#'
#' Builds a structured summary that mirrors [print.discriminant_projector()] and
#' adds per-component standard deviations and the original call when stored.
#'
#' @param object A `discriminant_projector` object.
#' @param ... Currently ignored.
#'
#' @return An object of class `"summary.discriminant_projector"`.
#' @method summary discriminant_projector
#' @export
summary.discriminant_projector <- function(object, ...) {
  out <- list(
    method   = .projector_method(object),
    n        = .n_samples(object),
    d_in     = .dim_in(object),
    d_out    = .dim_out(object),
    nclasses = .projector_nclasses(object),
    sdev     = object$sdev,
    call     = object$call
  )
  class(out) <- "summary.discriminant_projector"
  out
}

#' Print a discriminant projector summary
#'
#' @param x A `summary.discriminant_projector` object.
#' @param ... Currently ignored.
#'
#' @return The input `x`, invisibly.
#' @method print summary.discriminant_projector
#' @export
print.summary.discriminant_projector <- function(x, ...) {
  cat(sprintf("Discriminant projector summary (%s)\n", x$method))
  cat(sprintf("  Input dim:  %s features\n",   .fmt_int(x$d_in)))
  cat(sprintf("  Output dim: %s components\n", .fmt_int(x$d_out)))
  cat(sprintf("  Samples:    %s\n",            .fmt_int(x$n)))
  if (!is.na(x$nclasses)) cat(sprintf("  Classes:    %d\n", x$nclasses))
  if (!is.null(x$sdev) && length(x$sdev)) {
    cat("\nComponent sdev:\n")
    sd_vec <- as.numeric(x$sdev)
    names(sd_vec) <- paste0("Comp", seq_along(sd_vec))
    print(round(sd_vec, 4))
  }
  if (!is.null(x$call)) {
    cat("\nCall:\n")
    print(x$call)
  }
  invisible(x)
}

# ---------------------------------------------------------------------------
# Internal helpers (not exported)
# ---------------------------------------------------------------------------

.projector_method <- function(x) {
  if (!is.null(x$method)) return(as.character(x$method)[1])
  drop <- c("discriminant_projector", "bi_projector", "projector", "list")
  cls  <- setdiff(class(x), drop)
  if (length(cls)) cls[1] else "discriminant_projector"
}

.projector_nclasses <- function(x) {
  labs <- x$labels
  if (is.null(labs)) return(NA_integer_)
  if (is.matrix(labs)) ncol(labs) else length(unique(labs))
}

.dim_in    <- function(x) if (is.null(x$v)) NA_integer_ else nrow(x$v)
.dim_out   <- function(x) if (is.null(x$v)) NA_integer_ else ncol(x$v)
.n_samples <- function(x) if (is.null(x$s)) NA_integer_ else nrow(x$s)
.fmt_int   <- function(x) if (is.na(x)) "?" else format(x)

`%||%` <- function(a, b) if (is.null(a)) b else a

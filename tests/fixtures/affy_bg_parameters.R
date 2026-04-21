# bg.parameters() from the Bioconductor affy package, vendored here so
# fixture regeneration does not require installing affy or a clone of
# its source tree.
#
# Upstream source: affy::bg.parameters() in affy/R/bg.R
# Upstream project: https://bioconductor.org/packages/affy
# Reference commit: b2e5a6675a642f692a54fcee94e7485bac03390c (affy 1.89.0)
# Licence: LGPL (>= 2.0)
# Copyright: Rafael A. Irizarry, Laurent Gautier, Benjamin Milo Bolstad,
#            Crispin Miller, and contributors.
#
# pylimma's normexp_fit(method="rma") ports this function to Python in
# pylimma/normalize.py::_bg_parameters. It is vendored verbatim here so
# the R-side parity fixtures in tests/fixtures/generate_all_fixtures.R
# can call the original implementation without external dependencies.

bg.parameters <- function(pm, n.pts = 2 ^ 14) {

  max.density <- function(x, n.pts) {
    aux <- density(x, kernel = "epanechnikov", n = n.pts, na.rm = TRUE)
    aux$x[order(-aux$y)[1]]
  }

  pmbg <- max.density(pm, n.pts)           ## Log helps detect mode
  bg.data <- pm[pm < pmbg]
  ## do it again to really get the mode
  pmbg <- max.density(bg.data, n.pts)
  bg.data <- pm[pm < pmbg]
  bg.data <- bg.data - pmbg

  bgsd <- sqrt(sum(bg.data ^ 2) / (length(bg.data) - 1)) * sqrt(2)

  sig.data <- pm[pm > pmbg]
  sig.data <- sig.data - pmbg

  expmean <- max.density(sig.data, n.pts)
  alpha <- 1 / expmean
  mubg <- pmbg
  list(alpha = alpha, mu = mubg, sigma = bgsd)
}

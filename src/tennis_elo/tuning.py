"""Fitting the rating hyper-parameters on the training period only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .elo import EloParams, run_elo
from .evaluate import log_loss

#: which forecast column of :class:`EloResult` each variant is scored on
VARIANT_PROB = {"overall": "p_overall", "surface": "p_surface", "blend": "p_blend"}


def _unpack(theta, fit_w: bool, fit_mov: bool, base: EloParams) -> EloParams:
    c, o, s = np.exp(theta[0]), np.exp(theta[1]), 1 / (1 + np.exp(-theta[2]))
    i = 3
    w = base.surface_weight
    if fit_w:
        w = 1 / (1 + np.exp(-theta[i]))
        i += 1
    beta = base.mov_beta
    if fit_mov:
        beta = float(np.exp(theta[i]))
    return EloParams(c=float(c), o=float(o), s=float(s), surface_weight=float(w),
                     mov_beta=float(beta), bo5_adjust=base.bo5_adjust)


def tune(matches: pd.DataFrame, train_mask: np.ndarray, variant: str = "blend",
         base: EloParams = EloParams(), fit_mov: bool = False, maxiter: int = 250,
         verbose: bool = False) -> tuple[EloParams, float]:
    """Minimise training-period log loss of the chosen forecast over ``(c, o, s[, w, beta])``.

    Parameters are optimised on unconstrained transforms (log for ``c, o, beta``, logit
    for ``s, w``) with Nelder-Mead. Ratings still run through the whole history (they
    must, to be warm by the test period) but only ``train_mask`` rows enter the loss, so
    nothing after the training window influences the parameters.
    """
    col = VARIANT_PROB[variant]
    fit_w = variant == "blend"
    x0 = [np.log(base.c), np.log(base.o), np.log(base.s / (1 - base.s))]
    if fit_w:
        x0.append(np.log(base.surface_weight / (1 - base.surface_weight)))
    if fit_mov:
        x0.append(np.log(max(base.mov_beta, 0.5)))
    # only the training rows (and the burn-in before them) are needed for the loss
    last = int(np.flatnonzero(train_mask)[-1]) + 1
    sub = matches.iloc[:last]
    mask = train_mask[:last]

    def objective(theta):
        params = _unpack(theta, fit_w, fit_mov, base)
        p = getattr(run_elo(sub, params), col)[mask]
        loss = log_loss(p)
        if verbose:
            print(f"{loss:.5f} {params}")
        return loss

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-6})
    return _unpack(res.x, fit_w, fit_mov, base), float(res.fun)

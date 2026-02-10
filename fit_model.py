# fit_model.py
import numpy as np
from scipy.optimize import curve_fit

def first_order_step(t, Ka, tau, y0, t0):
    """
    First-order step response with Ka treated as ONE parameter:

        y(t) = y0 + Ka * (1 - exp(-(t - t0)/tau)), for t >= t0
        y(t) = y0, for t < t0

    Parameters
    ----------
    t : array-like
    Ka : float
        Single gain*step term (Ka)
    tau : float
        Time constant (>0)
    y0 : float
        Baseline output
    t0 : float
        Step time
    """
    t = np.asarray(t, dtype=float)
    t_shift = np.maximum(t - t0, 0.0)
    return y0 + Ka * (1.0 - np.exp(-t_shift / tau))

def clean_sort(t, y):
    """Remove NaNs/infs and sort by time."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    order = np.argsort(t)
    return t[order], y[order]

def initial_guesses(t, y, t0):
    """
    Fast guesses for Ka, tau, y0 using baseline + 63.2% rule.
    """
    t, y = clean_sort(t, y)

    # y0 guess: average before step if available, else first few points
    pre = y[t <= t0]
    y0 = float(np.mean(pre)) if pre.size >= 2 else float(np.mean(y[: min(3, len(y))]))

    # steady-state estimate from last 20% of points
    n_tail = max(3, int(0.2 * len(y)))
    y_inf = float(np.mean(y[-n_tail:]))

    # Ka guess = (y_inf - y0)
    Ka0 = y_inf - y0

    # tau guess via 63.2% crossing after step:
    # y(tau) = y0 + 0.632*(y_inf - y0)
    target = y0 + 0.632 * (y_inf - y0)

    after = t >= t0
    t_after = t[after]
    y_after = y[after]

    tau0 = (t[-1] - t[0]) / 3.0 if len(t) > 1 else 1.0
    idx = np.where(y_after >= target)[0]
    if idx.size > 0:
        i = int(idx[0])
        if i == 0:
            tau0 = max(float(t_after[0] - t0), 1e-6)
        else:
            t1, t2 = float(t_after[i - 1]), float(t_after[i])
            y1, y2 = float(y_after[i - 1]), float(y_after[i])
            # linear interpolation to find crossing time
            t_cross = t1 + (target - y1) * (t2 - t1) / (y2 - y1 + 1e-12)
            tau0 = max(t_cross - t0, 1e-6)

    return float(Ka0), float(max(tau0, 1e-6)), float(y0)

def fit_first_order(t, y, t0=0.0, fit_y0=True):
    """
    Fit Ka and tau (and optionally y0) by nonlinear least squares.

    Returns dict with:
      Ka, tau, y0, SSE, R2, y_fit, residuals, plus initial guesses.
    """
    t, y = clean_sort(t, y)
    if t.size < 4:
        raise ValueError("Need at least 4 valid data points.")

    Ka0, tau0, y0_guess = initial_guesses(t, y, t0)

    if fit_y0:
        popt, _ = curve_fit(
            lambda tt, Ka, tau, y0: first_order_step(tt, Ka, tau, y0, t0),
            t, y,
            p0=[Ka0, tau0, y0_guess],
            bounds=([-np.inf, 1e-9, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=20000
        )
        Ka_hat, tau_hat, y0_hat = popt
    else:
        popt, _ = curve_fit(
            lambda tt, Ka, tau: first_order_step(tt, Ka, tau, y0_guess, t0),
            t, y,
            p0=[Ka0, tau0],
            bounds=([-np.inf, 1e-9], [np.inf, np.inf]),
            maxfev=20000
        )
        Ka_hat, tau_hat = popt
        y0_hat = y0_guess

    y_fit = first_order_step(t, float(Ka_hat), float(tau_hat), float(y0_hat), t0)
    residuals = y - y_fit

    SSE = float(np.sum(residuals ** 2))
    ybar = float(np.mean(y))
    SStot = float(np.sum((y - ybar) ** 2))
    R2 = float(1.0 - SSE / SStot) if SStot > 0 else float("nan")

    return {
        "t": t,
        "y": y,
        "Ka": float(Ka_hat),
        "tau": float(tau_hat),
        "y0": float(y0_hat),
        "SSE": SSE,
        "R2": R2,
        "y_fit": y_fit,
        "residuals": residuals,
        "Ka0": Ka0,
        "tau0": tau0,
        "y0_guess": y0_guess
    }
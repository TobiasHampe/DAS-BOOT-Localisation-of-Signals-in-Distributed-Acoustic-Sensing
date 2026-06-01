"""
tt_stacking.py — TT stacking algorithm and post-processing.

    moveout_stack              semblance-based moveout inversion for a single window
    run_stacking_loop          sliding-window loop with 2D Kalman tracker
    assign_y_sign_stacking     assign y-sign from cable crossing time
    filter_stacking            filter df_stack by semblance, spread, and geometry
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from tqdm.auto import tqdm
import matplotlib.pyplot as plt



# 1. MOVEOUT STACK INVERSION


def moveout_stack(windowed_amp, distances, dt, c_w,
                  x_lo, x_hi, y_lo, y_hi,
                  depth_interp=None, cable_z=0.0,
                  dx_grid=50.0, dy_grid=20.0,
                  ch_step=5, t_margin_pct=15,
                  n_candidates=5):
    """
    Semblance-based moveout stacking inversion for a single time window.

    For each (x_s, y_s) in the search grid, expected arrival times are computed
    per channel using the 3-D travel-time model:

        t(x) = sqrt((x - x_s)^2 + y_s^2 + z(x)^2) / c_w

    Data is NMO-corrected and semblance is computed with a strain-sensitivity
    weighting w_k ∝ (x_k - x_s)^2 / r_k^3:

        S(t0) = [Σ_k a_k(t0+Δt_k) w_k]^2  /  [Σ_k a_k^2  ·  Σ_k w_k^2]

    Parameters
    ----------
    windowed_amp : ndarray (nt, nch)   DAS data for this window (signed)
    distances    : ndarray (nch,)      channel positions in km
    dt           : float               sample interval (s)
    c_w          : float               water sound speed (m/s)
    x_lo, x_hi   : float               along-cable search bounds (m)
    y_lo, y_hi   : float               crosstrack search bounds (m)
    depth_interp : callable or None    km → cable depth (m); uses cable_z if None
    cable_z      : float               constant depth when depth_interp is None
    dx_grid, dy_grid : float           grid spacing (m)
    ch_step      : int                 spatial subsampling (every ch_step-th channel)
    t_margin_pct : int                 % of window excluded at edges for apex search
    n_candidates : int                 number of top candidates returned

    Returns
    -------
    dict or None
        x_m, y_m           best-semblance grid point (m)
        t0_sample          apex time index within window
        stack_amplitude    semblance value of best candidate
        pick_idx           NMO-corrected sample indices for each channel
        ch_idx             channel indices used (after subsampling)
        x_spread, y_spread std of top-N candidate positions (low = coherent)
        x_median, y_median median of top-N candidate positions (m)
        x_weighted, y_weighted semblance-weighted mean of top-N (m)
        candidates         list of (semblance, x, y, t0) for top-N candidates
    """
    nt, nx_full = windowed_amp.shape

    ch_idx = np.arange(0, nx_full, ch_step)
    data   = windowed_amp[:, ch_idx]
    dist_m = distances[ch_idx] * 1000.0
    n_ch   = len(ch_idx)

    if depth_interp is not None:
        z_arr = np.abs(depth_interp(distances[ch_idx]))
    else:
        z_arr = np.full(n_ch, cable_z)

    t_margin = max(1, int(nt * t_margin_pct / 100))
    t0_lo    = t_margin
    t0_hi    = nt - t_margin

    x_grid = np.arange(x_lo, x_hi + dx_grid, dx_grid)
    y_grid = np.arange(y_lo, y_hi + dy_grid, dy_grid)

    candidates = []
    ch_range   = np.arange(n_ch)

    for x_s in x_grid:
        for y_s in y_grid:
            t_mod     = np.sqrt((dist_m - x_s)**2 + y_s**2 + z_arr**2) / c_w
            t_mod_rel = t_mod - t_mod.min()
            t_samp    = np.round(t_mod_rel / dt).astype(int)
            max_off   = t_samp.max()

            t0_end = min(t0_hi, nt - max_off)
            if t0_end <= t0_lo:
                continue

            t0_arr  = np.arange(t0_lo, t0_end)
            idx     = t0_arr[:, None] + t_samp[None, :]
            valid   = (idx >= 0) & (idx < nt)
            n_valid = valid.sum(axis=1)

            if n_valid.max() < max(5, n_ch // 3):
                continue

            idx_safe = np.where(valid, idx, 0)
            traces   = np.where(valid, data[idx_safe, ch_range[None, :]], 0.0)

            x_rel    = dist_m - x_s
            r        = np.sqrt(x_rel**2 + y_s**2 + z_arr**2)
            w        = x_rel**2 / r**3
            w       /= w.max()
            w_sq_sum = (w**2).sum()

            stack_sum  = (traces * w[None, :]).sum(axis=1)
            amp_sq_sum = (traces**2).sum(axis=1)

            semblance = np.where(
                amp_sq_sum > 0,
                stack_sum**2 / (amp_sq_sum * w_sq_sum + 1e-30),
                0.0,
            )

            best_row = semblance.argmax()
            best_idx = idx_safe[best_row]
            peak_val = float(semblance.max())
            peak_t0  = int(t0_arr[best_row])
            candidates.append((peak_val, x_s, y_s, peak_t0, best_idx))

    if not candidates:
        return None

    candidates.sort(key=lambda r: r[0], reverse=True)
    top_n = candidates[:n_candidates]

    best_val, best_x, best_y, best_t0, best_idx = top_n[0]

    xs = np.array([r[1] for r in top_n])
    ys = np.array([r[2] for r in top_n])

    weights  = np.array([r[0] for r in top_n])
    weights /= weights.sum()
    x_wmean  = float(np.dot(weights, xs))
    y_wmean  = float(np.dot(weights, ys))

    return {
        "x_m":             best_x,
        "y_m":             best_y,
        "t0_sample":       best_t0,
        "stack_amplitude": best_val,
        "pick_idx":        best_idx,
        "ch_idx":          ch_idx,
        "x_spread":        float(np.std(xs)),
        "y_spread":        float(np.std(ys)),
        "x_median":        float(np.median(xs)),
        "y_median":        float(np.median(ys)),
        "x_weighted":      x_wmean,
        "y_weighted":      y_wmean,
        "candidates":      [(float(s), float(x), float(y), int(t0))
                            for s, x, y, t0, _ in top_n],
    }


# 2. SLIDING-WINDOW LOOP WITH KALMAN TRACKER


def run_stacking_loop(inv_data, time_axis, distances, dt,
                      start_time_str=None, end_time_str=None,
                      window_ms=2000, step_ms=500,
                      c_w=1500.0,
                      y_lower=10.0, y_upper=500.0,
                      dx_grid=50.0, dy_grid=20.0,
                      ch_step=5,
                      depth_interp=None, cable_z=0.0,
                      min_stack_snr=None,
                      kalman_sigma_ax=0.05, kalman_sigma_ay=0.05,
                      kalman_sigma_rx=150.0, kalman_sigma_ry=80.0,
                      x_max_drift=300.0,
                      y_max_drift=100.0,
                      v_max=6.17,
                      x_fairway_lo=None, x_fairway_hi=None,
                      n_candidates=10,
                      use_mahal_gate=True,
                      disable_tqdm=False):
    """
    Sliding-window moveout stacking with 2D Kalman filter.

    Kalman state: [x_s, vx, y_s, vy]. Measurements are the semblance-weighted
    position averages (x_weighted, y_weighted) from moveout_stack.
    The Mahalanobis gate (chi-sq threshold 13.82, df=2, p=0.999) rejects
    outlier measurements when use_mahal_gate=True.

    Parameters
    ----------
    inv_data        : ndarray (nt, nch)   preprocessed DAS data (signed)
    time_axis       : list of datetime    UTC timestamps, length nt
    distances       : ndarray (nch,)      channel positions in km
    dt              : float               sample interval (s)
    start_time_str  : str or None         "HHMMSS" window start (None = beginning)
    end_time_str    : str or None         "HHMMSS" window end   (None = end)
    window_ms       : int                 stacking window length (ms)
    step_ms         : int                 window step (ms)
    c_w             : float               water sound speed (m/s)
    y_lower, y_upper: float               crosstrack search bounds (m)
    dx_grid, dy_grid: float               grid spacing (m)
    ch_step         : int                 spatial subsampling
    depth_interp    : callable or None    km → cable depth (m)
    cable_z         : float               constant depth when depth_interp is None
    min_stack_snr   : float or None       skip windows below this SNR
    kalman_sigma_ax, _ay : float          process noise acceleration std (m/s²)
    kalman_sigma_rx, _ry : float          measurement noise std (m)
    x_max_drift     : float               max x search radius around prediction (m)
    y_max_drift     : float               max y search radius around prediction (m)
    v_max           : float               max ship speed for velocity clipping (m/s)
    x_fairway_lo, _hi : float or None     hard x bounds (m); None = full cable extent
    n_candidates    : int                 top candidates per window
    use_mahal_gate  : bool                reject outlier measurements
    disable_tqdm    : bool                suppress progress bar

    Returns
    -------
    df : DataFrame sorted by time_center with columns:
         x_m, y_m, t0_sample, stack_amplitude, pick_idx, ch_idx,
         x_spread, y_spread, x_median, y_median, x_weighted, y_weighted,
         candidates, x_meas, y_meas,
         mahal_sq, mahal_accepted,
         x_kalman, vx_kalman, y_kalman, vy_kalman,
         time_center, window_start_idx, window_end_idx, snr
    """
    dt_ms        = dt * 1000.0
    time_seconds = np.array([(t - time_axis[0]).total_seconds() for t in time_axis])

    if start_time_str is None and end_time_str is None:
        start_idx = 0
        end_idx   = len(time_axis)
    else:
        ref_date = time_axis[0].date()
        t0_ref   = time_axis[0].time()

        def _to_sec(s):
            return (datetime.combine(ref_date, datetime.strptime(s, "%H%M%S").time())
                    - datetime.combine(ref_date, t0_ref)).total_seconds()

        start_s   = _to_sec(start_time_str) if start_time_str else 0.0
        end_s     = _to_sec(end_time_str)   if end_time_str   else time_seconds[-1]
        start_idx = np.searchsorted(time_seconds, start_s)
        end_idx   = np.searchsorted(time_seconds, end_s)

    window_samples = int(window_ms / dt_ms)
    step_samples   = int(step_ms   / dt_ms)

    dist_m    = distances * 1000.0
    x_lo_full = x_fairway_lo if x_fairway_lo is not None else dist_m.min()
    x_hi_full = x_fairway_hi if x_fairway_hi is not None else dist_m.max()

    H_xy = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]], dtype=float)
    R_xy         = np.diag([kalman_sigma_rx**2, kalman_sigma_ry**2])
    I4           = np.eye(4)
    mahal_thresh = 13.82

    kal_x = None
    kal_P = None
    kal_t = None

    results = []

    for i in tqdm(range(start_idx, end_idx - window_samples, step_samples),
                  desc="Stacking windows", disable=disable_tqdm):
        w_data   = inv_data[i:i + window_samples, :]
        w_amp    = np.abs(w_data)
        w_time   = time_axis[i:i + window_samples]
        t_center = time_seconds[i + window_samples // 2]

        snr = np.percentile(w_amp, 99) / (np.median(w_amp) + 1e-12)
        if min_stack_snr is not None and snr < min_stack_snr:
            continue

        # Kalman predict → narrow search bounds
        if kal_x is not None:
            dt_k = t_center - kal_t
            if dt_k > 0:
                F = np.array([[1, dt_k, 0, 0   ],
                              [0, 1,    0, 0   ],
                              [0, 0,    1, dt_k],
                              [0, 0,    0, 1   ]], dtype=float)
                qx = kalman_sigma_ax**2
                qy = kalman_sigma_ay**2
                d2, d3, d4 = dt_k**2, dt_k**3, dt_k**4
                Q = np.array([
                    [0.25*d4*qx, 0.5*d3*qx, 0,          0         ],
                    [0.5*d3*qx,  d2*qx,     0,          0         ],
                    [0,          0,          0.25*d4*qy, 0.5*d3*qy],
                    [0,          0,          0.5*d3*qy,  d2*qy    ]], dtype=float)
                kal_x = F @ kal_x
                kal_P = F @ kal_P @ F.T + Q
            kal_t = t_center

            x_lo = max(x_lo_full, kal_x[0] - x_max_drift)
            x_hi = min(x_hi_full, kal_x[0] + x_max_drift)
            y_lo = max(y_lower,   abs(kal_x[2]) - y_max_drift)
            y_hi = min(y_upper,   abs(kal_x[2]) + y_max_drift)
        else:
            x_lo, x_hi = x_lo_full, x_hi_full
            y_lo, y_hi = y_lower, y_upper

        out = moveout_stack(
            w_data, distances, dt, c_w,
            x_lo=x_lo, x_hi=x_hi,
            y_lo=y_lo, y_hi=y_hi,
            depth_interp=depth_interp, cable_z=cable_z,
            dx_grid=dx_grid, dy_grid=dy_grid,
            ch_step=ch_step,
            n_candidates=n_candidates,
        )

        if out is None:
            continue

        x_meas = out["x_weighted"]
        y_meas = out["y_weighted"]
        out["x_meas"] = x_meas
        out["y_meas"] = y_meas

        # Kalman update
        z              = np.array([x_meas, y_meas])
        mahal_sq       = np.nan
        mahal_accepted = True

        if kal_x is None:
            kal_x = np.array([x_meas, 0.0, y_meas, 0.0])
            kal_P = np.diag([kalman_sigma_rx**2, (v_max / 2)**2,
                             kalman_sigma_ry**2, (v_max / 2)**2])
            kal_t = t_center
        else:
            innov          = z - H_xy @ kal_x
            S              = H_xy @ kal_P @ H_xy.T + R_xy
            mahal_sq       = float(innov @ np.linalg.inv(S) @ innov)
            mahal_accepted = not use_mahal_gate or mahal_sq <= mahal_thresh

            if mahal_accepted:
                K     = kal_P @ H_xy.T @ np.linalg.inv(S)
                kal_x = kal_x + K @ innov
                kal_P = (I4 - K @ H_xy) @ kal_P
                kal_x[1] = np.clip(kal_x[1], -v_max, v_max)
                kal_x[3] = np.clip(kal_x[3], -v_max, v_max)

        out["mahal_sq"]         = mahal_sq
        out["mahal_accepted"]   = mahal_accepted
        out["x_kalman"]         = kal_x[0]
        out["vx_kalman"]        = kal_x[1]
        out["y_kalman"]         = kal_x[2]
        out["vy_kalman"]        = kal_x[3]
        out["time_center"]      = w_time[len(w_time) // 2]
        out["window_start_idx"] = i
        out["window_end_idx"]   = i + window_samples
        out["snr"]              = snr
        results.append(out)

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("No windows produced results.")
        return df

    return df.sort_values("time_center").reset_index(drop=True)



# 3. POST-PROCESSING

def assign_y_sign_stacking(df, crossing_override=None, smooth_window=5):
    """
    Assign y_sign based on minimum y_m (closest approach = cable crossing).

    Parameters
    ----------
    df                : df_stack from run_stacking_loop
    crossing_override : str or None   "HHMMSS" to force crossing time
    smooth_window     : int           rolling median window for automatic detection

    Returns
    -------
    df_signed, crossing_time
        df_signed has added columns y_sign (+1 before crossing, -1 after)
        and y_signed = y_m * y_sign.
    """
    df = df.copy().sort_values("time_center").reset_index(drop=True)
    df["time_center"] = pd.to_datetime(df["time_center"], utc=True)

    if crossing_override is not None:
        ct_date = df["time_center"].iloc[0].date()
        crossing_time = pd.Timestamp(
            datetime.combine(
                ct_date,
                datetime.strptime(crossing_override, "%H%M%S").time(),
                tzinfo=timezone.utc,
            )
        )
    else:
        y_col    = "y_kalman" if "y_kalman" in df.columns else "y_m"
        y_smooth = df[y_col].rolling(smooth_window, center=True, min_periods=1).median()
        idx_min  = y_smooth.idxmin()
        crossing_time = df["time_center"].iloc[idx_min]

    df["y_sign"]   = (df["time_center"] <= crossing_time).map({True: 1.0, False: -1.0})
    df["y_signed"] = df["y_m"] * df["y_sign"]

    return df, crossing_time



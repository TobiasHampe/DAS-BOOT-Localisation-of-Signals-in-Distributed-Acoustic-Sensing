"""
config.py — Per-ship parameters for the Storebælt TT-stacking pipeline.

Usage
-----
    from src.config import get_ship

    ship = get_ship("TERN SEA")

    # preprocess
    data, time_axis, distances = preprocess_ship(
        ship_cfg       = ship,
        file_paths_npy = npy_files,
        noise_data     = noise,
        dmin_km        = ship["dmin_km"],
        dmax_km        = ship["dmax_km"],
        dx=dx, dt=dt, offset_m=offset_m,
    )

    # stacking
    df_stack = run_stacking_loop(
        inv_data       = data,
        time_axis      = time_axis,
        distances      = distances,
        dt             = dt,
        start_time_str = ship["start_time_str"],
        end_time_str   = ship["end_time_str"],
        **ship["stacking_cfg"],
    )

    # y-sign assignment
    df_signed, crossing_time = assign_y_sign_stacking(
        df_stack,
        crossing_override = ship["crossing_time_override"],
    )
"""

DEFAULT_STACKING_CFG = dict(
    window_ms       = 2000,
    step_ms         = 1000,
    c_w             = 1500.0,
    y_lower         = 1.0,
    y_upper         = 1000.0,
    dx_grid         = 10.0,
    dy_grid         = 10.0,
    ch_step         = 5,
    min_stack_snr   = None,
    kalman_sigma_ax = 0.05,
    kalman_sigma_ay = 1.5,
    kalman_sigma_rx = 50.0,
    kalman_sigma_ry = 5.0,
    x_max_drift     = 100.0,
    y_max_drift     = 300.0,
    v_max           = 10,
    x_fairway_lo    = None,
    x_fairway_hi    = None,
    n_candidates    = 5,
)



# SHIP CONFIGURATIONS
#
# Required fields per ship:
#   start_time_str          "HHMMSS"  data window start
#   end_time_str            "HHMMSS"  data window end
#   crossing_time_override  "HHMMSS"  cable-crossing time for y-sign assignment
#   bound                   "north" | "south"
#   dmin_km                 float     calibrated near distance (km)
#   dmax_km                 float     calibrated far  distance (km)
#   stacking_cfg            dict      overrides for DEFAULT_STACKING_CFG (can be {})


SHIPS = {
    "TERN SEA": dict(
        start_time_str         = "090532",
        end_time_str           = "090932",
        crossing_time_override = "090730",
        bound                  = "south",
        dmin_km                = 4.75,   
        dmax_km                = 8.5,  
        stacking_cfg           = dict(y_upper=1200.0, x_fairway_lo=6250, x_fairway_hi=6600),
    ),
    "STAVFJORD": dict(
        start_time_str         = "090910",
        end_time_str           = "091710",
        crossing_time_override = "091310",
        bound                  = "south",
        dmin_km                = 4.5,   
        dmax_km                = 8.5,  
        stacking_cfg           = dict(y_upper=2000.0, x_fairway_lo=6400, x_fairway_hi=6700),
    ),
    "ARC ENDEAVOR": dict(
        start_time_str         = "092251",
        end_time_str           = "093041",
        crossing_time_override = "092801",
        bound                  = "south",
        dmin_km                = 5,   
        dmax_km                = 8.5,   
        stacking_cfg           = dict(y_upper=1000.0, x_fairway_lo=6500, x_fairway_hi=6900),
    ),
    "RESOLUTE BAY": dict(
        start_time_str         = "092615",
        end_time_str           = "093325",
        crossing_time_override = "092111",
        bound                  = "north",
        dmin_km                = 5,
        dmax_km                = 8.5,
        stacking_cfg           = dict(y_upper=2000.0, x_fairway_lo=6500, x_fairway_hi=7700),
    ),
    "DUSK": dict(
        start_time_str         = "101419",
        end_time_str           = "102329",
        crossing_time_override = "101756",
        bound                  = "north",
        dmin_km                = 4.5,
        dmax_km                = 8.5,
        stacking_cfg           = dict(y_upper=1000.0, x_fairway_lo=6500, x_fairway_hi=7700),
    ),
    "FEDERAL SUTTON": dict(
        start_time_str         = "101639",
        end_time_str           = "102410",
        crossing_time_override = "101949",
        bound                  = "north",
        dmin_km                = 4.5,
        dmax_km                = 8.5,
        stacking_cfg           = dict(y_upper=1000.0, x_fairway_lo=6500, x_fairway_hi=7700),
    ),
}


def get_ship(name: str) -> dict:
    """
    Return the full config for a ship: timing + distance window + merged stacking config.

    stacking_cfg in the returned dict is DEFAULT_STACKING_CFG with any ship-specific
    overrides applied, so it can be passed directly as **ship["stacking_cfg"].
    """
    if name not in SHIPS:
        raise KeyError(f"Unknown ship {name!r}. Available: {list(SHIPS)}")
    entry = dict(SHIPS[name])
    entry["stacking_cfg"] = {**DEFAULT_STACKING_CFG, **entry.get("stacking_cfg", {})}
    return entry

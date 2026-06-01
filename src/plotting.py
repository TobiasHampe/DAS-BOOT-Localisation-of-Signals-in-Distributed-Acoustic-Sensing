"""
plotting.py — Visualisation helpers for TT-stacking output vs AIS ground truth.

    project_ais_to_cable    project AIS lat/lon onto cable → along-cable distance (m)
    plot_stacking_vs_ais    two-panel x/y comparison: stacking + Kalman + AIS
    plot_stacking_map       satellite map: cable + AIS track + DAS estimates coloured by time
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyproj import Geod


# CABLE PROJECTION

def _load_cable_polyline(cable_path):
    """
    Load cable vertices from GeoJSON.
    Handles both formats:
      - Storebælt: top-level "positions" dict keyed by integer strings
      - Faroe:     top-level "features" array with properties {lon, lat, depth}
    Returns (lats, lons, along_cable_m) as numpy arrays.
    """
    with open(cable_path) as f:
        geo = json.load(f)

    if "positions" in geo:
        pos  = geo["positions"]
        keys = sorted(pos.keys(), key=int)
        lons = np.array([pos[k]["lon"] for k in keys])
        lats = np.array([pos[k]["lat"] for k in keys])
    else:
        rows = [feat["properties"] for feat in reversed(geo["features"])]
        lons = np.array([r["lon"] for r in rows])
        lats = np.array([r["lat"] for r in rows])

    geod  = Geod(ellps="WGS84")
    dists = [0.0]
    for i in range(1, len(lats)):
        _, _, d = geod.inv(lons[i - 1], lats[i - 1], lons[i], lats[i])
        dists.append(dists[-1] + d)

    return lats, lons, np.array(dists)


def project_ais_to_cable(df_ais, cable_path):
    """
    Project each AIS row onto the cable polyline via perpendicular segment projection.

    Works in a local Cartesian frame (metres) centred on the cable midpoint so
    that the foot of the perpendicular is found correctly even when the ship is
    far from the cable. Nearest-vertex approaches fail in that case because they
    snap to whichever vertex happens to be closest rather than the true
    perpendicular foot on the segment.

    Parameters
    ----------
    df_ais      : DataFrame with 'lat' and 'lon' columns
    cable_path  : path to cable GeoJSON

    Returns
    -------
    x_ais : ndarray  along-cable distance in metres for each row
    """
    cable_lats, cable_lons, cable_dists = _load_cable_polyline(cable_path)

    # Local flat-Earth projection centred at cable midpoint
    lat0 = np.mean(cable_lats)
    lon0 = np.mean(cable_lons)
    R    = 6_371_000.0
    k    = np.pi / 180.0

    def to_xy(lat, lon):
        x = (lon - lon0) * np.cos(lat0 * k) * R * k
        y = (lat - lat0) * R * k
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    cx, cy = to_xy(cable_lats, cable_lons)

    x_ais = np.empty(len(df_ais))
    for i, (_, row) in enumerate(df_ais.iterrows()):
        px, py = to_xy(row["lat"], row["lon"])
        px, py = float(px), float(py)

        best_dist_sq = np.inf
        best_s       = 0.0

        for j in range(len(cx) - 1):
            dx, dy     = cx[j+1] - cx[j], cy[j+1] - cy[j]
            seg_len_sq = dx*dx + dy*dy
            if seg_len_sq < 1e-6:
                continue

            # Parameter t of the perpendicular foot, clamped to [0, 1]
            t  = np.clip(((px - cx[j])*dx + (py - cy[j])*dy) / seg_len_sq, 0.0, 1.0)
            qx = cx[j] + t*dx
            qy = cy[j] + t*dy

            dist_sq = (px - qx)**2 + (py - qy)**2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_s       = cable_dists[j] + t * (cable_dists[j+1] - cable_dists[j])

        x_ais[i] = best_s

    return x_ais



# MAIN PLOT

def plot_stacking_vs_ais(df_signed, df_ais, crossing_time,
                         cable_geojson_path=None,
                         ship_name=""):
    """
    Two-panel figure comparing TT-stacking output with AIS ground truth.

    Panel 1 — x (along-cable, m)
        · blue dots   x_weighted  raw semblance-weighted stacking estimate
        · orange line x_kalman    Kalman-smoothed position
        · green line  AIS         lat/lon projected onto the cable
        · red dashed  apex        cable-crossing time

    Panel 2 — y (crosstrack, m, signed: + approaching, − departing)
        · blue dots   y_signed    signed raw stacking estimate
        · orange line y_kalman    signed Kalman position
        · green line  AIS dist    perpendicular distance, signed by crossing time
        · red dashed  apex        cable-crossing time

    Parameters
    ----------
    df_signed          : DataFrame from assign_y_sign_stacking
                         (columns: time_center, x_weighted, x_kalman,
                          y_signed, y_kalman, y_sign)
    df_ais             : AIS DataFrame pre-filtered to this ship
                         (columns: time, lat, lon, dist)
    crossing_time      : pd.Timestamp  cable-crossing time
    cable_geojson_path : str or None   needed for AIS x projection
    ship_name          : str           figure title

    Returns
    -------
    fig : matplotlib Figure
    """
    df = df_signed.copy()
    df["time_center"] = pd.to_datetime(df["time_center"], utc=True)
    crossing_time     = pd.to_datetime(crossing_time, utc=True)

    t0  = df["time_center"].min()
    t1  = df["time_center"].max()
    ais = df_ais[(df_ais["time"] >= t0) & (df_ais["time"] <= t1)].copy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw=dict(hspace=0.06),
    )

    # Panel 1: x (along-cable) 
    ax1.scatter(df["time_center"], df["x_weighted"],
                s=5, alpha=0.45, zorder=2, label="x_weighted (raw stacking)")
    ax1.plot(df["time_center"], df["x_kalman"],
             color="orange", lw=2, zorder=3, label="x_kalman")

    if cable_geojson_path is not None and {"lat", "lon"}.issubset(ais.columns) and len(ais):
        x_ais = project_ais_to_cable(ais, cable_geojson_path)
        ax1.scatter(ais["time"], x_ais,
                    marker="x", color="green", s=20, lw=1.2, zorder=3,
                    label="AIS (projected to cable)")

    ax1.axvline(crossing_time, color="red", linestyle="--", lw=1.5, label="crossing")
    ax1.set_ylabel("x (m)")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.7)
    if ship_name:
        ax1.set_title(ship_name, fontsize=10, loc="left")

    # Panel 2: y (crosstrack, signed) 
    y_kalman_signed = df["y_kalman"] * df["y_sign"]

    ax2.scatter(df["time_center"], df["y_signed"],
                s=5, alpha=0.45, zorder=2, label="y_signed (raw stacking)")
    ax2.plot(df["time_center"], y_kalman_signed,
             color="orange", lw=2, zorder=3, label="y_kalman")

    if "dist" in ais.columns and len(ais):
        ais_sign = np.where(ais["time"] <= crossing_time, 1.0, -1.0)
        ax2.scatter(ais["time"], ais["dist"] * ais_sign,
                    marker="x", color="green", s=20, lw=1.2, zorder=3,
                    label="AIS dist")

    ax2.axvline(crossing_time, color="red", linestyle="--", lw=1.5, label="crossing")
    ax2.set_ylabel("y (m)")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.7)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=20, ha="right")
    plt.tight_layout()

    return fig


# ============================================================
# MAP PLOT HELPERS
# ============================================================

def _get_perp_sign(ais_before, cable_lats, cable_lons):
    """
    Returns +1 if AIS points before crossing are to the LEFT of the cable
    (in the direction of increasing cable distance), -1 if to the RIGHT.

    This determines the sign convention for _cable_to_latlon: which side of
    the cable is "positive y_signed" (= approaching side).
    """
    if len(ais_before) == 0:
        return 1.0

    lat0 = np.mean(cable_lats)
    lon0 = np.mean(cable_lons)
    R    = 6_371_000.0
    k    = np.pi / 180.0
    cos0 = np.cos(lat0 * k)

    cx = (cable_lons - lon0) * cos0 * R * k
    cy = (cable_lats - lat0) * R * k

    crosses = []
    for _, row in ais_before.iterrows():
        px = float((row["lon"] - lon0) * cos0 * R * k)
        py = float((row["lat"] - lat0) * R * k)

        best_d2 = np.inf
        best_cr = 0.0
        for j in range(len(cx) - 1):
            dx, dy     = cx[j+1] - cx[j], cy[j+1] - cy[j]
            seg_len_sq = dx*dx + dy*dy
            if seg_len_sq < 1e-6:
                continue
            t  = np.clip(((px - cx[j])*dx + (py - cy[j])*dy) / seg_len_sq, 0.0, 1.0)
            qx = cx[j] + t*dx
            qy = cy[j] + t*dy
            d2 = (px - qx)**2 + (py - qy)**2
            if d2 < best_d2:
                best_d2 = d2
                best_cr = dx*(py - qy) - dy*(px - qx)   # 2D cross product
        crosses.append(best_cr)

    return 1.0 if np.mean(crosses) >= 0 else -1.0


def _cable_to_latlon(x_m, y_m, cable_lats, cable_lons, cable_dists, perp_sign=1.0):
    """
    Convert along-cable distance (m) + signed crosstrack offset (m) to lat/lon.

    Vectorised over array inputs.
    perp_sign : +1 → positive y is to the LEFT of cable (CCW)
                -1 → positive y is to the RIGHT (CW)
    """
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)

    lat_c = np.interp(x_m, cable_dists, cable_lats)
    lon_c = np.interp(x_m, cable_dists, cable_lons)

    idx = np.searchsorted(cable_dists, x_m, side="right").clip(1, len(cable_dists) - 1)

    R = 6_371_000.0
    k = np.pi / 180.0

    # Cable tangent in local metric (east, north) components
    dlat_m = (cable_lats[idx] - cable_lats[idx - 1]) * R * k
    dlon_m = (cable_lons[idx] - cable_lons[idx - 1]) * np.cos(lat_c * k) * R * k
    mag    = np.where(np.sqrt(dlat_m**2 + dlon_m**2) < 1e-10, 1.0,
                      np.sqrt(dlat_m**2 + dlon_m**2))

    tngnt_n = dlat_m / mag   # north component of unit cable tangent
    tngnt_e = dlon_m / mag   # east  component

    # Left-perpendicular (CCW 90°): perp_e = -tngnt_n, perp_n = tngnt_e
    north_m = perp_sign * y_m * tngnt_e
    east_m  = perp_sign * y_m * (-tngnt_n)

    lat_out = lat_c + north_m / (R * k)
    lon_out = lon_c + east_m  / (np.cos(lat_c * k) * R * k)

    return lat_out, lon_out


# ============================================================
# MAP PLOT
# ============================================================

def plot_stacking_map(df_signed, df_ais, crossing_time,
                      cable_geojson_path,
                      ship_name="", bound="",
                      margin=0.006,
                      show_kalman=False):
    """
    Satellite-background map comparing DAS stacking estimates with AIS ground truth.

    Layers (bottom → top):
        satellite tiles   ESRI World Imagery
        Cable             white line, alpha 0.8
        AIS outside win   grey dots (context)
        AIS in window     x-markers coloured by relative time [s]
        DAS estimates     filled circles coloured by same relative time scale

    The shared RdYlGn colormap means matching colours = same time instant,
    making it easy to see whether the DAS estimate tracks the AIS position.

    Parameters
    ----------
    df_signed          : output of assign_y_sign_stacking
                         (columns: time_center, x_kalman, y_signed, y_sign)
    df_ais             : AIS DataFrame with columns time, lat, lon
    crossing_time      : pd.Timestamp
    cable_geojson_path : str
    ship_name          : str   used in title
    bound              : str   "north" or "south"
    margin             : float degrees of padding around the visible region
    show_kalman        : bool  False → raw stacking (x_weighted, y_signed);
                               True  → Kalman-smoothed (x_kalman, y_kalman * y_sign)

    Returns
    -------
    fig : matplotlib Figure
    """
    try:
        import contextily as ctx
    except ImportError:
        raise ImportError("pip install contextily  (needed for satellite tiles)")

    df = df_signed.copy()
    df["time_center"] = pd.to_datetime(df["time_center"], utc=True)
    crossing_time     = pd.to_datetime(crossing_time, utc=True)

    t0       = df["time_center"].min()
    t1       = df["time_center"].max()
    duration = (t1 - t0).total_seconds()
    t0_ts    = t0.timestamp()

    ais_in  = df_ais[(df_ais["time"] >= t0) & (df_ais["time"] <= t1)].copy()
    ais_out = df_ais[(df_ais["time"] < t0)  | (df_ais["time"] > t1)].copy()

    cable_lats, cable_lons, cable_dists = _load_cable_polyline(cable_geojson_path)

    # Convert DAS (x, y) → lat/lon using perpendicular direction inferred from AIS
    ais_before = ais_in[ais_in["time"] <= crossing_time]
    perp_sign  = _get_perp_sign(ais_before, cable_lats, cable_lons)

    if show_kalman:
        das_x = df["x_kalman"].values
        das_y = (df["y_kalman"] * df["y_sign"]).values
        das_label = "Stacking estimate (Kalman)"
    else:
        das_x = df["x_weighted"].values
        das_y = df["y_signed"].values
        das_label = "Stacking estimate (raw)"

    das_lats, das_lons = _cable_to_latlon(
        das_x, das_y,
        cable_lats, cable_lons, cable_dists, perp_sign,
    )

    # Map extent: cover all AIS-in-window + DAS estimates + margin
    all_lats = np.concatenate([ais_in["lat"].values, das_lats])
    all_lons = np.concatenate([ais_in["lon"].values, das_lons])
    lat_min, lat_max = all_lats.min() - margin, all_lats.max() + margin
    lon_min, lon_max = all_lons.min() - margin, all_lons.max() + margin

    # Cable slice visible in this extent
    buf = 0.02
    vis = np.where(
        (cable_lats >= lat_min - buf) & (cable_lats <= lat_max + buf) &
        (cable_lons >= lon_min - buf) & (cable_lons <= lon_max + buf)
    )[0]
    i0 = max(0, vis[0] - 1) if len(vis) else 0
    i1 = min(len(cable_lats), vis[-1] + 2) if len(vis) else 0

    # Relative times (seconds from t0)
    cmap     = plt.cm.RdYlGn
    norm     = plt.Normalize(vmin=0, vmax=duration)
    das_trel = np.array([tc.timestamp() - t0_ts for tc in df["time_center"]])
    ais_trel = np.array([t.timestamp()  - t0_ts for t in ais_in["time"]])

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ctx.add_basemap(ax, crs="EPSG:4326",
                    source=ctx.providers.Esri.WorldImagery,
                    attribution=False)

    # Cable
    if i1 > i0:
        ax.plot(cable_lons[i0:i1], cable_lats[i0:i1],
                color="white", lw=2, alpha=0.8, zorder=3, label="Cable")

    # AIS outside window (grey context dots)
    if len(ais_out):
        ais_out_vis = ais_out[
            ais_out["lat"].between(lat_min, lat_max) &
            ais_out["lon"].between(lon_min, lon_max)
        ]
        if len(ais_out_vis):
            ax.scatter(ais_out_vis["lon"], ais_out_vis["lat"],
                       s=18, c="grey", alpha=0.6, zorder=4,
                       label="AIS (outside window)")

    # AIS in window — x markers, coloured by t_rel
    if len(ais_in):
        ax.scatter(ais_in["lon"], ais_in["lat"],
                   s=40, c=ais_trel, cmap=cmap, norm=norm,
                   marker="x", linewidths=1.8, zorder=5,
                   label="AIS (in window)")

    # DAS stacking estimates — filled circles, same colour scale
    if len(df):
        ax.scatter(das_lons, das_lats,
                   s=22, c=das_trel, cmap=cmap, norm=norm,
                   zorder=6, label=das_label)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.55)
    cbar.set_label(f"Relative time [s]  (t₀ = {t0.strftime('%H:%M:%S')})",
                   fontsize=9)

    # Scale bar (100 m, white)
    lat_mid   = (lat_min + lat_max) / 2
    scale_deg = 100.0 / (111_320.0 * np.cos(np.radians(lat_mid)))
    sb_x0     = lon_min + 0.06 * (lon_max - lon_min)
    sb_y      = lat_min + 0.04 * (lat_max - lat_min)
    ax.plot([sb_x0, sb_x0 + scale_deg], [sb_y, sb_y],
            color="white", lw=3, solid_capstyle="butt", zorder=7)
    ax.text(sb_x0 + scale_deg / 2, sb_y + 0.004 * (lat_max - lat_min),
            "100 m", color="white", ha="center", va="bottom",
            fontsize=9, fontweight="bold", zorder=7)

    # Axes, grid, title, legend
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3, color="white", linewidth=0.5)

    parts = ["Stacking"]
    if ship_name:
        parts.append(f"— {ship_name}")
    if bound:
        parts.append(f"({bound})")
    parts.append(f"|  {t0.strftime('%H:%M:%S')}–{t1.strftime('%H:%M:%S')}")
    ax.set_title("  ".join(parts), fontsize=10, loc="left", pad=6)

    ax.legend(loc="upper right", fontsize=8, framealpha=0.75)

    plt.tight_layout()
    return fig

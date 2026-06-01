"""
preprocessing.py — DAS data loading and preprocessing for Storebælt (NPY) and Faroe (HDF5).

Shared signal processing
    butter_bandpass         Butterworth bandpass filter
    channel_normalisation   per-channel RMS normalisation
    fk_filter               F-K velocity filter

Storebælt (NPY)
    extract_das_data_npy    load .npy files for a time/distance window
    cable_depth_storebælt   depth interpolator from cable GeoJSON (positions dict)
    load_ais                load processed AIS CSV
    preprocess_ship         full preprocessing pipeline for one ship

Faroe (HDF5)
    extract_metadata        read HDF5 file header
    extract_das_data        load HDF5 files for a file-index/distance window
    cable_depth_faroe       depth interpolator from cable GeoJSON (features array)
    load_ais_faroe          load AIS CSV
    preprocess_hdf5         full preprocessing pipeline
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import h5py
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.interpolate import interp1d
from scipy import ndimage
from pyproj import Geod



# SIGNAL PROCESSING


def butter_bandpass(data, lowcut, highcut, fs, order=8, axis=0):
    """Butterworth bandpass filter via sosfiltfilt (zero-phase)."""
    nyq = 0.5 * fs
    if highcut >= nyq:
        highcut = nyq * 0.99
    sos = butter(order, [lowcut, highcut], btype="bandpass", output="sos", fs=fs)
    return sosfiltfilt(sos, data, axis=axis)


def channel_normalisation(data, reference_data=None):
    """Normalise each channel by its RMS, optionally using a separate noise reference."""
    ref = reference_data if reference_data is not None else data
    rms = np.sqrt(np.mean(ref ** 2, axis=0, keepdims=True))
    rms = np.where(rms == 0, 1.0, rms)
    return data / rms


def fk_filter(data, dt, dx, fs, vmin, vmax, sigma=30):
    """
    F-K velocity filter. Keeps energy with apparent velocity in [vmin, vmax] m/s.
    Gaussian smoothing (sigma) reduces Gibbs artefacts at the filter edges.
    """
    nt, nx = data.shape
    freq  = np.fft.fftshift(np.fft.fftfreq(nt, d=1 / fs))
    k_num = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))

    filt = np.zeros((len(k_num), len(freq)))
    for i, f in enumerate(freq):
        if abs(f) < 1e-10:
            continue
        col = np.zeros_like(k_num)
        col[(k_num > f / vmax) & (k_num < f / vmin)] = 1
        filt[:, i] = col

    sub = filt[len(k_num) // 2:, len(freq) // 2:]
    sub = ndimage.gaussian_filter(sub, sigma)
    filt[len(k_num) // 2:, len(freq) // 2:] = sub
    filt += np.fliplr(filt)
    filt += np.flipud(filt)

    D = np.fft.fftshift(np.fft.fft2(data))
    D *= filt.T
    return np.real(np.fft.ifft2(np.fft.ifftshift(D)))



# STOREBÆLT (NPY files)


def extract_das_data_npy(file_paths, start_time, end_time,
                          dmin_km, dmax_km,
                          dx=2.035, dt=0.0025,
                          sample_step=1, channel_step=1,
                          offset_m=1235):
    """
    Load DAS data from Storebælt .npy files for a time and distance window.

    Parameters
    ----------
    file_paths  : list   sorted .npy file paths
    start_time  : str    "HHMMSS" window start (inclusive)
    end_time    : str    "HHMMSS" window end   (exclusive)
    dmin_km, dmax_km : float  distance range in km (before offset correction)
    dx          : float  spatial sampling (m)
    dt          : float  temporal sampling (s)
    offset_m    : float  calibration offset added to distances (m)

    Returns
    -------
    data, time_axis, distances, fs, dx, dt
    """
    fs = 1 / dt
    n_channels        = 3917
    distance_array_km = np.arange(n_channels) * dx / 1000.0
    dist_mask         = (distance_array_km >= dmin_km) & (distance_array_km <= dmax_km)
    distances         = distance_array_km[dist_mask][::channel_step]

    selected = sorted(
        fp for fp in file_paths
        if start_time <= os.path.basename(fp).replace(".npy", "") < end_time
    )
    if not selected:
        raise ValueError(f"No files found between {start_time} and {end_time}")

    data_list, time_axis = [], []
    for fp in selected:
        s = os.path.basename(fp).replace(".npy", "")
        h, m, sec  = int(s[:2]), int(s[2:4]), int(s[4:6])
        file_start = datetime(2025, 11, 22, h, m, sec, tzinfo=timezone.utc)

        arr = np.load(fp)
        arr = arr[::sample_step, :][:, dist_mask][:, ::channel_step]

        n_samples = arr.shape[0]
        time_axis.extend(
            file_start + timedelta(seconds=i * dt * sample_step)
            for i in range(n_samples)
        )
        data_list.append(arr)

    data      = np.vstack(data_list)
    distances = distances + offset_m / 1000.0

    print(f"Loaded {len(selected)} files ({start_time} → {end_time}),  "
          f"shape {data.shape},  "
          f"duration {(time_axis[-1] - time_axis[0]).total_seconds():.1f} s")
    return data, time_axis, distances, fs, dx, dt


def cable_depth_storebælt(path, geod_crs="WGS84"):
    """Return interp1d(km → depth_m) from Storebælt cable GeoJSON (positions dict)."""
    with open(path, "r") as f:
        geo = json.load(f)
    pos  = geo["positions"]
    keys = sorted(pos.keys(), key=int)
    rows = [[pos[k]["lon"], pos[k]["lat"], pos[k]["depth"]] for k in keys]
    df   = pd.DataFrame(rows, columns=["lon", "lat", "depth"])

    geod  = Geod(ellps=geod_crs)
    dists = [0]
    for i in range(1, len(df)):
        _, _, d = geod.inv(df.lon.iloc[i-1], df.lat.iloc[i-1],
                           df.lon.iloc[i],   df.lat.iloc[i])
        dists.append(dists[-1] + d)
    df["distance_km"] = np.array(dists) / 1000.0

    return interp1d(df["distance_km"], df["depth"].abs(),
                    bounds_error=False, fill_value="extrapolate")


def load_ais(csv_path):
    """Load processed Storebælt AIS CSV. Returns DataFrame with 'time' column (UTC)."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.rename(columns={"timestamp": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    print(f"AIS: {len(df)} points,  "
          f"{df['time'].iloc[0].strftime('%H:%M:%S')}–"
          f"{df['time'].iloc[-1].strftime('%H:%M:%S')} UTC")
    return df


def preprocess_ship(ship_cfg, file_paths_npy, noise_data,
                    dmin_km, dmax_km, dx, dt, offset_m,
                    apply_detrend=True, apply_channel_norm=True,
                    apply_bp=True, bp_low=31, bp_high=100,
                    apply_fk=True, fk_params=None):
    """
    Load and preprocess DAS data for one Storebælt ship.

    Pipeline: detrend → bandpass → channel normalisation → F-K filter.

    Parameters
    ----------
    ship_cfg        : dict with 'start_time_str' and 'end_time_str'
    file_paths_npy  : list of sorted .npy paths
    noise_data      : ndarray (nt, nch) from a quiet window, for channel normalisation
    dmin_km, dmax_km: float  calibrated distance window (km)
    dx, dt          : float  acquisition geometry
    offset_m        : float  calibration offset (m)
    fk_params       : dict with keys vmin, vmax, sigma (passed to fk_filter)

    Returns
    -------
    data_proc, time_axis, distances
    """
    dmin_raw = dmin_km - offset_m / 1000.0
    dmax_raw = dmax_km - offset_m / 1000.0

    raw, time_axis, distances, _, _, _ = extract_das_data_npy(
        file_paths_npy,
        ship_cfg["start_time_str"],
        ship_cfg["end_time_str"],
        dmin_km=dmin_raw, dmax_km=dmax_raw,
        dx=dx, dt=dt, offset_m=offset_m,
    )

    data = raw.copy().astype(np.float32)

    if apply_detrend:
        data = detrend(data, axis=0).astype(np.float32)

    if apply_bp:
        data = butter_bandpass(data, bp_low, bp_high, fs=1 / dt, order=4).astype(np.float32)

    if apply_channel_norm and noise_data is not None:
        data = channel_normalisation(data, noise_data)

    if apply_fk and fk_params is not None:
        data = fk_filter(data, dt=dt, dx=dx, fs=1 / dt, **fk_params)

    return data, time_axis, distances


# FAROE (HDF5 files)


def extract_metadata(file_path):
    """Read header fields from a single Faroe HDF5 file."""
    with h5py.File(file_path, "r") as f:
        start_time  = f["header/time"][()]
        dt          = f["header/dt"][()]
        dx          = f["header/dx"][()]
        channels    = f["header/channels"][()]
        num_samples = f["data"].shape[0]
    return start_time, dt, dx, channels, num_samples


def extract_das_data(file_paths, start_file_idx, end_file_idx,
                     dmin_km, dmax_km, sample_step=1, channel_step=1):
    """
    Load DAS data from a range of Faroe HDF5 files.

    Parameters
    ----------
    file_paths                    : list  sorted HDF5 paths
    start_file_idx, end_file_idx  : int   file index range (end exclusive)
    dmin_km, dmax_km              : float distance range in km

    Returns
    -------
    data, time_axis, distances, fs, dx, dt
    """
    _, dt, dx, channels, _ = extract_metadata(file_paths[0])
    fs = 1 / dt

    distance_m  = channels * dx
    distance_km = distance_m / 1000.0
    dist_mask   = (distance_km >= dmin_km) & (distance_km <= dmax_km)
    distances   = distance_km[dist_mask][::channel_step]

    data_list, time_axis = [], []
    for idx in range(start_file_idx, min(end_file_idx, len(file_paths))):
        file_start, file_dt, *_ = extract_metadata(file_paths[idx])
        with h5py.File(file_paths[idx], "r") as f:
            chunk = f["data"][::sample_step, dist_mask][:, ::channel_step]

        n  = chunk.shape[0]
        t0 = datetime.fromtimestamp(file_start, tz=timezone.utc)
        deltas = np.arange(n) * (file_dt * sample_step)
        time_axis.extend(t0 + timedelta(seconds=float(d)) for d in deltas)
        data_list.append(chunk)

    data    = np.vstack(data_list) if len(data_list) > 1 else data_list[0]
    n_files = min(end_file_idx, len(file_paths)) - start_file_idx
    dur     = (time_axis[-1] - time_axis[0]).total_seconds()
    print(f"Loaded files {start_file_idx}–{min(end_file_idx, len(file_paths))-1} "
          f"({n_files} files) → shape {data.shape},  {dur:.1f} s")
    return data, time_axis, distances, fs, dx, dt


def cable_depth_faroe(path, geod_crs="WGS84"):
    """Return interp1d(km → depth_m) from Faroe cable GeoJSON (features array)."""
    with open(path, "r") as f:
        geo = json.load(f)
    rows = []
    for feat in reversed(geo["features"]):
        p = feat["properties"]
        rows.append([p["lon"], p["lat"], p["depth"]])
    df = pd.DataFrame(rows, columns=["lon", "lat", "depth"])

    geod  = Geod(ellps=geod_crs)
    dists = [0]
    for i in range(1, len(df)):
        _, _, d = geod.inv(df.lon.iloc[i-1], df.lat.iloc[i-1],
                           df.lon.iloc[i],   df.lat.iloc[i])
        dists.append(dists[-1] + d)
    df["distance_km"] = np.array(dists) / 1000.0

    return interp1d(df["distance_km"], df["depth"].abs(),
                    bounds_error=False, fill_value="extrapolate")


def load_ais_faroe(file):
    """Load Faroe AIS CSV. Returns DataFrame with 'time' column (UTC)."""
    df = pd.read_csv(file)
    df = df.rename(columns={"timestamp": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    print(f"AIS: {len(df)} points,  "
          f"{df['time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')} – "
          f"{df['time'].iloc[-1].strftime('%H:%M:%S')} UTC")
    return df


def preprocess_hdf5(data, noise_data, fs, dx, dt,
                    apply_detrend=True,
                    apply_channel_norm=True,
                    apply_bp=True, bp_low=4, bp_high=100,
                    apply_fk=True, fk_params=None):
    """
    Full preprocessing pipeline for Faroe HDF5 DAS data.

    Pipeline: detrend → channel normalisation → bandpass → F-K filter.

    Parameters
    ----------
    data        : ndarray (n_samples, n_channels)  ship data
    noise_data  : ndarray (n_samples, n_channels) or None  noise reference
    fs, dx, dt  : float  acquisition parameters
    fk_params   : dict with keys vmin, vmax, sigma

    Returns
    -------
    processed : ndarray same shape as data
    """
    if fk_params is None:
        fk_params = dict(vmin=1400, vmax=2000, sigma=30)

    out = data.copy().astype(float)

    if apply_detrend:
        out = detrend(out, axis=0, type="linear")
        if noise_data is not None:
            noise_data = detrend(noise_data.astype(float), axis=0, type="linear")

    if apply_channel_norm:
        out = channel_normalisation(out, reference_data=noise_data)

    if apply_bp:
        out = butter_bandpass(out, bp_low, bp_high, fs)

    if apply_fk:
        out = fk_filter(out, dt, dx, fs, **fk_params)

    return out

# DAS BOOT: Localisation of Signals in Distributed Acoustic Sensing

Source code for the bachelor's thesis *DAS BOOT: Localisation of Signals in
Distributed Acoustic Sensing* (Hampe & Tranegaard, DTU, 2026).

Two passive vessel localisation algorithms are implemented and evaluated on
DAS recordings from two submarine fibre-optic cables: the Great Belt (Denmark)
and the Faroe–Shetland corridor. Both methods estimate vessel position from
acoustic strain induced in the cable as a vessel transits overhead, validated
against AIS ground truth.

---

## Methods

**TT-stacking (TDOA)**  
Weighted semblance grid search over candidate source positions using traveltime
differences across cable channels. Per-window estimates are linked into a
continuous track by a 2D Kalman filter with Mahalanobis gating. Logic lives in
`src/` and is imported by the notebooks.

**MUSIC (DOA)**  
Narrowband MUSIC beamformer that decomposes the spatial covariance matrix into
signal and noise subspaces. The MUSIC spectrum is evaluated over a 2D search
grid; the peak gives the vessel position. All code is self-contained inside
the notebooks as a step-by-step guide.

---

## Repository structure

```
├── README.md
├── environment.yml
├── .gitignore
│
├── notebooks/
│   ├── tt_stacking/
│   │   ├── greatbelt_tt_stacking.ipynb   # Great Belt single-ship runner
│   │   └── faroe_tt_stacking.ipynb       # Faroe–Shetland runner (HDF5)
│   └── music/
│       ├── greatbelt_music.ipynb         # Great Belt step-by-step MUSIC
│       └── faroe_music.ipynb             # Faroe–Shetland step-by-step MUSIC
│
└── src/                                  # Shared modules for TT-stacking only
    ├── preprocessing.py  # Butterworth bandpass, FK filter, RMS normalisation,
    │                     # data loading for .npy (Great Belt) and .hdf5 (Faroe)
    ├── tt_stacking.py    # Grid search, semblance, Kalman filter, Mahalanobis gate,
    │                     # y-sign assignment
    ├── config.py         # Per-ship configuration: SHIPS dict, DEFAULT_STACKING_CFG,
    │                     # get_ship() lookup helper
    └── plotting.py       # Visualisation: x/y time-series vs AIS, satellite map
                          # with time-coloured DAS + AIS tracks
```

The `data/` directory is excluded from this repository (see `.gitignore`).

---

## Data availability

The raw DAS recordings were provided by DTU, FMI and GlobalConnect under a data
agreement and cannot be redistributed. Acquisition parameters for both datasets
are reported in the thesis (Table 1.2).

AIS data were obtained from the Danish Maritime Authority.

---

## Installation

```bash
git clone https://github.com/TobiasHampe/DAS-BOOT-Localisation-of-Signals-in-Distributed-Acoustic-Sensing.git
cd DAS-BOOT-Localisation-of-Signals-in-Distributed-Acoustic-Sensing
conda env create -f environment.yml
conda activate das-env
```

---

## TT-stacking quickstart

```bash
jupyter notebook notebooks/tt_stacking/greatbelt_tt_stacking.ipynb
```

Each TT-stacking notebook follows the same five-step flow:

1. **Imports** — load `src` modules via `REPO_ROOT` path injection
2. **Config / paths** — set data directories, acquisition constants, preprocessing parameters
3. **One-time setup** — load file list, AIS, cable depth interpolator
4. **Run ship** — change one line (`SHIP_NAME = "..."`) and re-run to process any ship
5. **Visualise** — `plot_stacking_vs_ais` (x/y time-series) and `plot_stacking_map` (satellite map)

### Ship configuration (`src/config.py`)

All per-ship parameters for the Great Belt dataset live in `src/config.py`:

```python
from src.config import get_ship

ship = get_ship("STAVFJORD")
# Returns timing (start/end/crossing), distance window (dmin_km, dmax_km),
# and a fully-merged stacking config dict ready for **ship["stacking_cfg"]
```

`DEFAULT_STACKING_CFG` holds the baseline stacking parameters. Individual ships
override only the keys that differ (e.g. `y_upper`, `x_fairway_lo/hi`).

### Visualisation (`src/plotting.py`)

Two functions are available after running `assign_y_sign_stacking`:

**`plot_stacking_vs_ais`** — two-panel time-series  
Compares along-cable (x) and crosstrack (y) estimates against AIS ground truth.
Stacking scatter, Kalman-smoothed track, and AIS are plotted on the same time
axis; a red dashed line marks the cable crossing.

**`plot_stacking_map`** — satellite map  
Overlays the DAS estimates and AIS track on ESRI World Imagery tiles. Both are
coloured by the same `RdYlGn` relative-time colormap so matching colours =
matching time — a single-glance validation of localisation accuracy.

```python
from src.plotting import plot_stacking_vs_ais, plot_stacking_map

plot_stacking_vs_ais(df_signed, df_ais_ship, crossing_time, CABLE_GEOJSON, ship_name=SHIP_NAME)
plot_stacking_map(df_signed, df_ais_ship, crossing_time, CABLE_GEOJSON,
                  ship_name=SHIP_NAME, bound=ship["bound"])
# show_kalman=True switches the map from raw scatter to Kalman-smoothed track
```

---

## MUSIC quickstart

```bash
jupyter notebook notebooks/music/faroe_music.ipynb
```

The MUSIC notebooks are self-contained and require no imports from `src/`.
Each notebook walks through the full pipeline step by step:

1. **Parameters** — paths, ship metadata, array window, grid extent, frequency
2. **Load data** — HDF5 files trimmed to the spatial and time window of interest
3. **Load cable** — GeoJSON/JSON positions mapped to the masked channels
4. **Load AIS** — ground-truth pings and interpolators for continuous comparison
5. **Build grid** — rotated 2D UTM search grid projected to lat/lon
6. **Steering delays** — per-grid-point travel-time differences relative to a reference channel
7. **MUSIC loop** — windowed bandpass → analytic signal → `arlpy.bf.music` spectrum → peak extraction
8. **Results** — per-window estimates saved to CSV/JSON with RMSE and median error statistics

---

## Citation

```bibtex
@thesis{hampetranegaard2026dasboot,
  author  = {Hampe, Tobias Lollike Bay and Tranegaard, Emil},
  title   = {{DAS BOOT}: Localisation of Signals in Distributed Acoustic Sensing},
  school  = {Technical University of Denmark},
  year    = {2026},
  type    = {Bachelor's Thesis},
  url     = {https://github.com/TobiasHampe/DAS-BOOT-Localisation-of-Signals-in-Distributed-Acoustic-Sensing}
}
```

---

## Licence

MIT

## Acknowledgements

Data provided by DTU, FMI and GlobalConnect. Supervised by Henning Heiselberg and
Hasse Bulow Pedersen, Center for Security DTU.

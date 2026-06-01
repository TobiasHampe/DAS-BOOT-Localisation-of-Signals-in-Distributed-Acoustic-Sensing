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

**TT-stacking (TDOA)** — `src/tt_stacking.py`  
Weighted semblance grid search over candidate source positions using traveltime
differences across cable channels. Per-window estimates are linked into a
continuous track by a 2D Kalman filter with Mahalanobis gating.

**MUSIC (DOA)** — `src/music.py`  
Narrowband MUSIC beamformer that decomposes the spatial covariance matrix into
signal and noise subspaces. The MUSIC spectrum is evaluated over a 2D search
grid; the peak gives the vessel position.

---

## Repository structure

```
├── README.md
├── environment.yml
├── .gitignore
│
├── notebooks/
│   ├── tt_stacking/
│   │   ├── storebelt_tt_stacking.ipynb
│   │   └── faroe_tt_stacking.ipynb
│   └── music/
│       ├── storebelt_music.ipynb
│       └── faroe_music.ipynb
│
└── src/
    ├── tt_stacking.py    # Grid search, semblance, Kalman filter, Mahalanobis gate
    ├── music.py          # Covariance matrix, eigendecomposition, MUSIC spectrum
    └── preprocessing.py  # Butterworth bandpass, FK filter, RMS normalisation,
                          # data loading for both .npy and .hdf5 formats
```

The `data/` directory is excluded from this repository (see `.gitignore`).

---

## Data availability

The raw DAS recordings were provided by FMI and GlobalConnect under a data
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

## Quickstart

Open a notebook to run the full pipeline, e.g.:

```bash
jupyter notebook notebooks/tt_stacking/storebelt_tt_stacking.ipynb
```

Each notebook covers one method and one dataset: load data → preprocess →
localise → plot tracks against AIS ground truth.

---

## Citation

```bibtex
@thesis{hampe2026dasboot,
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

Data provided by FMI and GlobalConnect. Supervised by Henning Heiselberg and
Hasse Bulow Pedersen, Center for Security DTU.

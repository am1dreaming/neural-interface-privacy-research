# Data

Data is **not stored in git**. This folder contains instructions only. Always read the license and terms of use from the dataset owners and cite them in your work.

| Dataset | Contents | Role in the project | Where to get it |
|---|---|---|---|
| ZuCo 1.0 / 2.0 | EEG plus eye tracking while reading sentences | main dataset for retrieval | OSF, search "ZuCo" (authors: Hollenstein et al., Zurich) |
| Broderick / Di Liberto (Natural Speech) | EEG while listening to continuous speech | second modality: listening | Dryad / the authors' publication |
| MEG-MASC | MEG while listening to speech | comparison with EEG | OpenNeuro |
| PhysioNet EEG Motor Movement/Imagery (EEGMMIDB) | 109 people, 64 channels, 160 Hz, 14 runs each; NO text | second dataset for the leakage side only (more people) | PhysioNet, search "eegmmidb"; `python data/download_eegmmidb.py` |
| TUH EEG Corpus | many hours of clinical EEG without text | encoder pre-training | requires registration at Temple University |

Links are intentionally not hard-coded: addresses and access terms change. Check the current dataset pages yourself.

## Expected layout

```
data/raw/<dataset>/...        downloaded as is (in .gitignore)
data/processed/<dataset>/...  preprocessing output (in .gitignore)
```

## Loader status

- [x] ZuCo 2.0, normal reading (task 1 NR): `neurotext.data.load_zuco2_nr` (HDF5 via h5py, quality filter, 6.4 s window, cache in `data/processed/`)
- [x] EEGMMIDB: `neurotext.data.load_eegmmidb` (pure-numpy EDF reader, windows per run, split by run, pre-registered held-out people excluded by default)
- [ ] ZuCo: word-level alignment, task 2 (TSR), ZuCo 1.0
- [ ] Broderick: reading, slicing by sentence
- [ ] TUH: subsample for pre-training

The pipeline can be checked on synthetic data (`neurotext.data.make_synthetic`); this only tests the code.

## Placing ZuCo 2.0

The large archive `2urht-osfstorage-archive.zip` takes a long time to download and sometimes breaks off. A truncated archive cannot be opened by a regular `unzip` or an archive manager (the central directory is missing at the end). Complete files can still be extracted from it in streaming mode, or you can download only what is needed from OSF: the folder `task1 - NR/Matlab files`.

You need the files `results<CODE>_NR.mat` anywhere inside `data/raw/zuco/`. Unfinished `.part` files are ignored.

## One interface for all datasets

Every loader returns `neurotext.data.EEGData` (recordings, subject ids, sentence ids, text embeddings, texts, subject codes, sampling rate, a report of what was discarded). Everything else (splits, training, baselines, probes) is independent of the dataset: a new dataset needs one reader module and a branch in `neurotext.experiment.load_data`.

## EEGMMIDB (second dataset, leakage only)

```bash
python data/download_eegmmidb.py --subjects 1-20      # quick look first, then without --subjects (about 2 GB)
python data/download_eegmmidb.py --check              # what is missing or damaged
```

Layout: `data/raw/eegmmidb/S001/S001R01.edf ... S109/S109R14.edf`. There is no text in this dataset, so decoding
cannot be tested here and is not claimed: the encoder is trained to align EEG with an embedding of the instructed
run type (eyes open / closed, fist or fists-and-feet, executed or imagined) plus a fixed run-specific component.
Splits are by RUN, never by window. 24 of the 109 people are held out (22 usable: S088 and S100 have most runs at another sampling rate; S092 in the development pool likewise, leaving 84 there) by the rule fixed in
`PREREGISTRATION.md`; the loader excludes them (`holdout: exclude`) and only `configs/eegmmidb_final.yaml` loads them.
Channels (64), sampling rate (160 Hz, decimated to 80 Hz) and the task differ from ZuCo, so absolute numbers are not
compared between the datasets, only the qualitative picture. A person with fewer than 12 usable runs is dropped and
listed in `meta["dropped_subjects"]` (a few of the 109 are known to have irregular recordings).

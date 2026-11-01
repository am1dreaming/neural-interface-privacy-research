# Data

Data is **not stored in git**. This folder contains instructions only. Always read the license and terms of use from the dataset owners and cite them in your work.

| Dataset | Contents | Role in the project | Where to get it |
|---|---|---|---|
| ZuCo 1.0 / 2.0 | EEG plus eye tracking while reading sentences | main dataset for retrieval | OSF, search "ZuCo" (authors: Hollenstein et al., Zurich) |
| Broderick / Di Liberto (Natural Speech) | EEG while listening to continuous speech | second modality: listening | Dryad / the authors' publication |
| MEG-MASC | MEG while listening to speech | comparison with EEG | OpenNeuro |
| TUH EEG Corpus | many hours of clinical EEG without text | encoder pre-training | requires registration at Temple University |

Links are intentionally not hard-coded: addresses and access terms change. Check the current dataset pages yourself.

## Expected layout

```
data/raw/<dataset>/...        downloaded as is (in .gitignore)
data/processed/<dataset>/...  preprocessing output (in .gitignore)
```

## Loader status

- [x] ZuCo 2.0, normal reading (task 1 NR): `neurotext.data.load_zuco2_nr` (HDF5 via h5py, quality filter, 6.4 s window, cache in `data/processed/`)
- [ ] ZuCo: word-level alignment, task 2 (TSR), ZuCo 1.0
- [ ] Broderick: reading, slicing by sentence
- [ ] TUH: subsample for pre-training

The pipeline can be checked on synthetic data (`neurotext.data.make_synthetic`); this only tests the code.

## Placing ZuCo 2.0

The large archive `2urht-osfstorage-archive.zip` takes a long time to download and sometimes breaks off. A truncated archive cannot be opened by a regular `unzip` or an archive manager (the central directory is missing at the end). Complete files can still be extracted from it in streaming mode, or you can download only what is needed from OSF: the folder `task1 - NR/Matlab files`.

You need the files `results<CODE>_NR.mat` anywhere inside `data/raw/zuco/`. Unfinished `.part` files are ignored.

## One interface for all datasets

Every loader returns `neurotext.data.EEGData` (recordings, subject ids, sentence ids, text embeddings, texts, subject codes, sampling rate, a report of what was discarded). Everything else (splits, training, baselines, probes) is independent of the dataset: a new dataset needs one reader module and a branch in `neurotext.experiment.load_data`.

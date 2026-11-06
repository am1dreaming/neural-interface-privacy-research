# Pre-registration

Written on 2026-11-05, **before any of the experiments listed below were run**. The commit that adds
this file (see `git log`) is the timestamp. Anything that changes after that commit goes into
"Deviations" at the bottom, with the date and the reason, never silently into the text above.

## 0. What is already known and therefore NOT confirmatory

These results exist from before this file and were looked at. They can motivate hypotheses but cannot
confirm them:

- ZuCo 2.0 (16 people): decoding EEG -> sentence is at chance; spectral features and the transformer
  encoder leak identity to a closed-set linear and MLP probe; PCA to 32-d does not remove it.
- Defence results on synthetic data (noise, PCA, identity-nulling, adversarial training).
- Everything in the current README, including the numbers.

ZuCo has only 16 people, so a held-out group of people cannot be set aside there. Experiments on ZuCo
that use **new encoders** (section 3, H4) are confirmatory for the encoders but not for the data: the
data was seen.

## 1. Data and the held-out group

- **Second dataset:** PhysioNet EEG Motor Movement/Imagery (EEGMMIDB), 109 people, 64 channels,
  160 Hz, 14 runs per person. It has no text, so only the leakage side can be tested. "Content" for the
  encoder is the instructed run type (baseline eyes open / closed, fist execution / imagery,
  fists-and-feet execution / imagery) plus a fixed run-specific component. Decoding quality is not
  measured and not claimed.
- **Preprocessing (fixed):** decimate 160 -> 80 Hz, 4 s windows (320 samples), at most 6 evenly spaced
  windows per run, runs missing or at another sampling rate are dropped and reported. Config:
  `configs/eegmmidb.yaml` (development) and `configs/eegmmidb_final.yaml` (identical except `holdout: include`).
- **Held-out group:** 24 of the 109 people, chosen by `numpy.random.default_rng(20261105).permutation`
  of the list `S001 ... S109` (the resulting list is fixed in `neurotext.data.eegmmidb.HOLDOUT` and checked by a test:
  S002 S005 S006 S012 S017 S019 S022 S024 S029 S032 S038 S040 S066 S067 S079 S080 S081 S083 S088 S096 S099 S100 S102 S108;
  S088 and S100 turned out to be dropped by the quality screen, see the data-availability check below). The loader excludes them
  by default. They are not used to choose hyper-parameters, features, defences or thresholds. They are
  evaluated **once**, with `experiments/final_holdout.py` (the script refuses to overwrite its output).
- **Development pool:** the other 85 people.
- **Data-availability check (done on 2026-11-06, before any experiment; only the quality screen of the loader was
  run, no model, no probe):** the loader drops a person with fewer than 12 usable runs. S088, S100 and S092 have
  12 of 14 runs recorded at another sampling rate and are dropped. So **22 of the 24 held-out people and 84 of the
  85 development people are usable**. This is why the held-out chance level is 1/22 and not 1/24, and why the
  largest development n is 64.
- **Split inside a person:** by run (14 runs, all windows of a run stay together), never by window.
- **Comparing datasets:** channels, sampling rate and task differ. Absolute numbers are not compared
  between ZuCo and EEGMMIDB, only the qualitative picture (above chance or not, the direction of effects).

## 2. Attacker and metrics

- **Closed-set identification:** linear and MLP probe on embeddings (or features), trained on some runs
  / sentence blocks of each person and tested on others. Metric: accuracy, compared with chance 1/n.
- **Open-set verification** ("same person or not"): pairs of recordings from *different* blocks/runs;
  scores are cosine similarity and cosine after LDA fitted on *other* people. Metrics: EER (chance 0.5),
  AUC (chance 0.5).
- **New people** = people the encoder never saw in training. Seeds change initialisation, the choice
  of people and the fold assignment, not the data, so intervals over seeds are a lower bound on the true
  uncertainty. This is stated next to every result.
- **Paired comparisons** are paired by seed, with 95% t-intervals over seeds.
- Normalising a person's recordings by that person's own statistics is **not** used inside
  cross-validation (it produces accuracy below chance; documented earlier in this project).

## 3. Hypotheses

Each hypothesis says what counts as confirmed and what counts as refuted.

**H1 - leakage replicates on a second dataset (family A, held-out group, `final_holdout.py`).**
For the usable held-out people (22, see section 1) and a trained encoder (transformer, 32-d, config `eegmmidb.yaml`):
 (a) linear-probe accuracy is above 1/n_heldout, (b) MLP-probe accuracy is above 1/n_heldout,
 (c) cosine EER is below 0.5, (d) LDA EER is below 0.5.
Confirmed for a sub-claim if its Holm-adjusted one-sided p < 0.05 (t-test over 10 seeds).
Refuted if the point estimate is not above chance / not below 0.5 in the direction claimed.

**H2 - leakage does not vanish as the number of people grows (family B, development pool, `scaling_curve.py`).**
For n in {5, 10, 20, 40, 64} new people (84 development people minus 20 who train the encoder) (nested samples within a seed), trained encoder:
 (a) linear-probe accuracy minus 1/n is above 0 at every n,
 (b) LDA EER is below 0.5 at every n,
each as a one-sided t-test over the 10 seeds; the 95% intervals are reported next to the p-values.
Closed-set accuracy is *expected to fall* as n grows (more candidates); this is not a refutation. The claim
is only that it stays above 1/n. Whether EER improves or worsens with n is reported, not tested.
Confirmed if (a) and (b) hold at every n with Holm-adjusted p < 0.05 over the 2 x 5 tests.
The n-points are nested samples within a seed, so the points of one curve are not independent.

**H3 - nulling the identity subspace helps against a linear attacker, an adaptive MLP recovers part of it
(family A, held-out group).** Defence `id-null m=32` against none, paired by seed:
 (a) linear-probe accuracy falls (one-sided p < 0.05 after Holm),
 (b) MLP-probe accuracy stays above 1/n_heldout (one-sided p < 0.05 after Holm).
Refuted for (b) if the MLP accuracy under the defence is not above chance.

**H4 - the leakage does not depend on the encoder (family C, ZuCo 2.0 new people, `encoder_robustness.py`).**
For each of four encoders (transformer, linear, small CNN, EEGNet; 32-d): linear-probe accuracy of the
trained encoder is above chance (one-sided t-test over 10 seeds, Holm over 4 encoders x 2 probes).
The paired difference "trained minus untrained" is reported for each encoder but is **not** a
hypothesis: the earlier finding that training reduces leakage is not assumed to generalise.
Embedding size (4, 8, 16, 32, 64) is explored, not tested.

**Exploratory (not confirmatory, said so in the README): feature ablation (`feature_ablation.json`).**
Questions fixed in advance, no thresholds: how much leakage each feature group gives alone and how much
is lost when it is removed (RMS, delta, theta, alpha, beta, gamma band power, individual alpha peak);
whether leakage survives per-recording amplitude normalisation and trimming the window edges;
which channels / electrode regions carry it. Reason: leakage on ZuCo is already known to be near
ceiling for spectral features, so a confirmatory test would be uninformative.

## 4. Multiple comparisons

Holm-Bonferroni within each family (A, B, C), family-wise alpha = 0.05, one-sided where the direction
is stated above. Tests are counted as: A = H1(a-d) + H3(a,b) = 6; B = 10; C = 8. Anything not listed
here is reported as exploratory with unadjusted intervals.

## 5. Procedure that keeps the plan honest

- Seeds 0-9 for every confirmatory run. The number of seeds is not changed after seeing results.
- The development pool may be used to debug code and to look at curves, but the *hypotheses, defences,
  model sizes and thresholds above are fixed*. Defence settings for the held-out run:
  noise 0.5 and 1.0; PCA 4 and 8; id-null 8 and 32; adversarial 1.0 (`configs/eegmmidb.yaml`, same in the final config).
- The held-out run is executed once. If it crashes for a technical reason it may be rerun with
  `--rerun-because "<reason>"`, which is logged in `results/deviations.log` and must be copied into
  "Deviations" below.
- The README separates "confirmatory" (H1-H4, as stated here) from "exploratory" (everything else).

## 6. Looks before this file was committed, and deviations

Things that were run or seen BEFORE this file was committed (no commit existed yet). They are not part of the
confirmatory analysis, and they limit how blind the hypotheses are:

- 2026-11-05/06: a pipeline check on synthetic data (`configs/audit_synthetic.yaml`). Says nothing about real EEG.
- 2026-11-06: the loader's quality screen on EEGMMIDB (section 1). No model, no probe.
- 2026-11-06: a **probe run** of `encoder_robustness.py` on ZuCo 2.0 with 3 seeds and 5 epochs (not the
  pre-registered configuration of 10 seeds and the default 30 epochs). It showed all four encoders above chance
  for the closed-set probes on new people. **H4 was therefore not blind**: its direction was seen. The probe
  is not used; the pre-registered run is repeated from scratch with 10 seeds, and only that run is reported as H4.
  Note also that H4 only asks whether the trained encoder beats chance, which the earlier untrained-encoder control
  already makes very likely; the informative quantity is the paired difference "trained minus untrained", which
  was never a hypothesis here and stays exploratory.

Deviations from the plan after the commit: none so far.

Addendum, 7 November 2026 (written after the held-out run; the text above is unchanged): the list above is
incomplete with respect to the commit. The 10-seed runs for H4 (`encoder_robustness.py`, 5 November, UTC), H2
(`scaling_curve.py`, 6 November, UTC) and the exploratory ablation were made after this file was written but
**before** its first commit (`d6587c0`, 6 November). The hypotheses were not changed after those runs, but
this can only be shown by the local file time. The held-out run (H1, H3) came after the commit and had no reruns.

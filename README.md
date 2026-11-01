# Neural Interface Privacy Lab

*EEG decoding, identity leakage, and the cost of defences.*

Open ML and cybersecurity research: can a sentence be decoded from non-invasive EEG, do the embeddings of such a decoder reveal who the person is, and what do defences cost? Status as of 1 November 2026.

## Key findings

1. **No evidence of EEG → sentence decoding.** On ZuCo 2.0 (natural reading, 16 people, 349 sentences), new people and new sentences give top-1 of 1.2–1.3% against a random level of 1.1–1.3%. A ridge model on spectral features and a neural network on raw EEG are equally at chance. The method is sensitive, though: a synthetic signal at 5% of the noise amplitude is found in 64–73% of cases.
2. **A data trap.** Within one person the result is 4.7% (4× chance), but a baseline that never sees EEG and only knows the sentence's position in the session gets 13.1%. With test blocks of 30 consecutive sentences it falls back to chance. Random splits leak through neighbouring sentences, so real data needs a block split.
3. **Identity leaks.** Spectral features identify a person at 99.6–100%, even after PCA to 32 dimensions. A 32-d encoder keeps much less, and a *trained* encoder leaks less than an *untrained* one with the same initialisation.
4. **Defences have a price.** The best one tested (PCA to 8 directions) is almost free in accuracy. Subtracting identity directions and adversarial training do not protect against a strong attacker.
5. **Per-person adapters** no longer hurt, but a benefit could not be shown.

Everything is reproducible and comes with controls (position oracle, block split, planted signal, untrained encoder). Null results are included.

## Setup

**Data.** ZuCo 2.0, normal reading: 16 people, 349 sentences, 105 EEG channels, 500 Hz. After the quality filter 5264 of 5584 recordings remain (261 without EEG, 59 broken). Put the data in `data/raw/zuco/` (it is not included in the repository). Dataset: Hollenstein et al., "ZuCo 2.0", LREC 2020; its licence is defined by the dataset authors.

**Decoding task.** Given an EEG recording, pick the sentence that was read among ~80–90 unseen ones (random top-1 ≈ 1.1–1.3%). Sentences come from articles and follow each other by topic.

**Attacker model (identity leakage).** Reference-set linkage: the attacker has each person's representations on some sentences and names the person from representations of other sentences. Test blocks of 30 consecutive sentences. Attackers: a linear probe and an MLP.

**Principles of honest evaluation.** The EEG-to-text literature has known methodological problems (split leakage, teacher forcing, results at noise level). So:

- splits are made by person, by sentence and by block, and tests check for the known kinds of leakage;
- every experiment prints baselines by default: random level, noise input, shuffled labels, position oracle;
- the contribution of the signal is measured as the difference from a baseline, not as an absolute metric;
- comparisons are paired across seeds, with 95% confidence intervals.

## Results

### 1. Sentence decoding (ZuCo 2.0, 16 people)

| Setting | New people | Same person |
|---|---|---|
| Ridge, **random** test sentences | 1.3% (position oracle 13.8%) | 4.7% (oracle 13.1%) |
| Ridge, test sentences in **blocks of 30** | 1.2% (oracle 2.2%, random 1.3%) | 1.2% (oracle 1.9%, random 1.1%) |
| Neural network on raw EEG (1 seed, random sentences) | 0.6% (noise input 1.0%) | not tested |

- There is no evidence that the sentence can be determined from EEG in any setting.
- The within-person "signal" is explained by the structure of the data, not by language. The position oracle, which never sees EEG, gets 13.1%, and with blocks of 30 only the random level remains. A random split hands out a hint through neighbouring sentences (topics and slow drift of the recording), so real data uses `split.block_size` by default.
- Sensitivity was measured: a synthetic signal at 5% of the noise is found in 64–73% of cases. The signal is idealised (linear, identical across people, fitted to the target text representation), so the detection threshold for a real signal is probably worse.
- This is **not** proof that EEG carries no information about language: features are averaged over the sentence, text is encoded with plain TF-IDF, and word-level time alignment was not tested.

### 2. Identity leakage (ZuCo 2.0, 10 seeds)

Accuracy of a probe that names the person from a representation:

| Representation | New people (4, chance 25%) | All 16 (chance 6.3%) |
|---|---|---|
| Full spectral features (840) | 99.6% | 100% |
| Same, PCA to 32 | 99.1% | 100% |
| Untrained encoder (same initial state), linear / MLP | 49% / 52% | 23% / 25% |
| Trained encoder (32), linear / MLP | 45% / 44% | 15% / 14% |

Paired difference "trained minus untrained" (95% interval over seeds). New people: linear −0.041 [−0.079, −0.003], MLP −0.084 [−0.126, −0.042]. All 16: −0.082 [−0.097, −0.067] and −0.114 [−0.128, −0.100].

- The spectral features alone identify a person almost without error, and compressing to 32 dimensions does not remove this: dimensionality is not the cause.
- The encoder keeps noticeably less (45% against a 25% chance level among new people), and training **reduces** leakage. The cause was not investigated. A possible explanation: alignment with text moves the representation away from differences between people, while an untrained network simply passes part of the input through.
- For the linear probe on new people the upper bound of the interval is only −0.003, so the estimate is conservative.
- "Training reduces leakage" applies to an encoder that cannot decode text (see section 1). A working decoder might behave differently.

### 3. Defences and their price (synthetic, 24 people, 10 seeds)

Real data is not suitable here: the decoder on it is at chance, and the "cost in quality" would be measured from zero. Defences are therefore tested on synthetic data where the decoder works (top-1 0.37 against a random 0.03). The attacker is adaptive: the probe is trained on the protected representations of other sentences from the same people. Cost is measured on new people and new sentences. Figure: `results/defence_tradeoff.png`.

![Price of privacy: decoder accuracy versus attacker accuracy for each defence (synthetic data, 10 seeds, 95% intervals). Top row: MLP attacker, bottom row: linear probe. The best defence sits in the top left.](results/defence_tradeoff.png)

No defence: top-1 0.371; the attacker names the person (among 6 new people, chance 0.167): linear probe 0.324, MLP 0.492.

| Defence | Decoder top-1 | MLP attacker | Takeaway |
|---|---|---|---|
| PCA, 8 directions | 0.345 (−0.026, not distinguishable) | 0.373 (−0.120) | almost free, best of those tested |
| PCA, 4 directions | 0.190 (−0.181) | 0.281 | leakage falls, accuracy halves |
| Noise, norm 0.5 | 0.329 (−0.041) | 0.395 (−0.097) | reasonable compromise |
| Noise, norm 1.0 | 0.257 (−0.113) | 0.283 (−0.209) | accuracy 30% lower |
| Noise, norm 2.0 | 0.153 (−0.217) | 0.198 | attacker almost at chance (0.167), but the decoder is badly damaged |
| Subtract 8 identity directions | 0.278 (−0.093) | 0.476 (not distinguishable) | has a cost, no protection from the MLP |
| Train against an "identity guesser" (weight 8) | 0.337 (−0.033) | 0.528 (**+0.036**) | leakage did not fall, it rose for the MLP |

- Subtracting identity directions barely bothers a non-linear attacker (linear probe 0.324 → 0.289, MLP 0.492 → 0.476). Assessing privacy needs a strong attacker, not just a linear probe.
- PCA turned out to be the best defence: cheaper than noise 0.5 and it cuts leakage more. But this is one scenario on synthetic data where differences between people are defined as channel mixing, so on live EEG the ranking of defences may differ.
- With these parameters adversarial training did not help new people. This does not prove it cannot work: the parameters were not tuned.
- Instead of LDA direction subtraction, subtraction of the subspace of per-person means is used, because after removing all LDA directions 16% of the separation of the means remained.
- The strength of between-person differences in the synthetic data was chosen using the no-defence condition only. A first trial run on one seed was seen before the list of defences was fixed. 15 settings were tried without correction for multiple comparisons.
- There are 6 new people in the test per seed; the synthetic data is regenerated on every seed, so seeds are independent.

### 4. Per-person adapters (synthetic)

The base model is frozen and a light adapter is trained per person (channel scaling, low-rank, channel mixing).

- The initial variant made results worse. Paired comparison with zero calibration over 10 seeds: 2 sentences −0.046 [−0.075, −0.017], 4 sentences −0.076 [−0.132, −0.020], 8 sentences −0.033 [−0.074, +0.008].
- The harm came from overfitting: with a smaller step and fewer epochs (lr 1e-3), or with a penalty for departing from the identity transform, it disappears. Both components hurt on their own.
- No variant shows a benefit, even when between-person differences are amplified (`smoke_synthetic_highvar.yaml`, zero calibration is ≈0.24 there).
- A channel-mixing adapter gave +0.06 at 8 sentences on the first 10 seeds, but variant J, chosen in advance, gave +0.018 [−0.004, +0.039] on 20 new seeds: **not confirmed** (a winner's-curse effect among ~12 variants). The comparison is partly circular: in the synthetic data the between-person differences are exactly channel mixing.
- Result: adapters stopped hurting, a benefit is not proven. There are 3 new people in the synthetic test.

### 5. Pipeline notes (synthetic)

- The smoke run on synthetic data checks that the pipeline works: the gain over baselines is positive, and on noise and shuffled labels it stays near the random level. This is not a scientific result.
- The first version of the identity probe split records randomly, and accuracy came out **below** the random level (repeats of one sentence landed in both training and test). The split is now by sentence (`groups=`).
- A single run is too noisy for comparisons: a small test shifts top-1 by several points. For conclusions use `run_seeds.py`. The same seed gives the same result (checked by a test).
- Quantisation: standard PyTorch builds for macOS lack the needed engine, so the int8 size cannot be measured. In that case `quantized_size_bytes` returns `None`, and `estimate_int8_linear_bytes` gives an analytical estimate (on Linux it differs from the measurement by about 6%).

## Limitations

- Identity leakage was measured within a single session (one recording per person). For another day accuracy may be noticeably lower. Absolute power can carry a fingerprint of the session and equipment rather than of the brain.
- There are 4 new people in the test on real data and 6 on synthetic. Seeds change initialisation and the choice of "new people" but not the data, so intervals over seeds are a lower bound.
- Defences were tested on synthetic data only.
- The open-set setting "same person or different" for new people was not tested.
- Decoding results apply to sentence-averaged features and to TF-IDF as the text encoder.

## Related work

Status as of 1 November 2026. Abstracts and pages were read; only numbers checked against the source are quoted.

- **Tai, "Pretrained, Frozen, Still Leaking"** ([arXiv 2606.09189](https://arxiv.org/abs/2606.09189), June 2026). Audit of frozen BIOT, LaBraM and EEGPT embeddings. Identity linkage is strong when the attacker has a reference set of candidate subjects, but the authors explicitly note that for held-out subjects they claim attribute leakage, not identity recovery, and that the strong identity result does not replicate on Sleep-EDF. A noise defence against an adaptive attacker and DP-SGD leave the attribute leakage channel essentially unchanged. Not about text.
- **Chebrolu et al.**, "Disentangling Neural Identity from Encoder Identity in a Privacy Audit of BP-GPT's fMRI-to-Text Decoding" ([NeurIPS 2026 page](https://neurips.cc/virtual/2026/160157)). 8 participants: embeddings and raw functional connectivity identify a person at 100% (chance 12.5%), but BP-GPT trains a separate encoder per participant, which leaves a signature of its own. Matching against profiles from independently seeded encoders gives 54.3% (95% CI 46.6–61.6%). The idea of separating the data fingerprint from the model fingerprint is close to our untrained-encoder control.
- **Srivastava, Bellet, Tommasi, Vincent**, "Privacy-Preserving Adversarial Representation Learning in ASR: Reality or Illusion?" ([Interspeech 2019, arXiv 1911.04913](https://arxiv.org/abs/1911.04913)). Adversarial training sharply lowers closed-set speaker classification accuracy but does not raise open-set verification error on unseen speakers, so it gives no protection. Our result on adversarial training and new people is consistent with this precedent.
- **Jo et al., 2024:** EEG-to-text models on ZuCo give similar results on pure noise input; consistent with our null result on decoding.
- Protecting identity in EEG with perturbations, masking and adversarial training, mostly in a closed setting on the same people: Meng et al. 2023, Chen et al. 2024, ID-RemovalNet 2025, SAFE, NeuroPriv 2026.

**Not new:** the fact that EEG and its embeddings leak identity; that noise, compression and masking reduce it; that adversarial identity removal can be an illusion against new people; that EEG-to-text results must be tested against noise.

**Possibly new** (absence of prior work is not proven, the search was not exhaustive): an audit of the embeddings of a sentence-level EEG decoder on natural reading; a population of "new people"; the control "untrained encoder with the same start"; the leakage-versus-decoder-accuracy trade-off for four defences against an adaptive non-linear attacker; controls for EEG → sentence (position oracle, block split, planted signal); open reproducible code.

## Structure

```
configs/        yaml experiment configs
data/           preparation instructions and scripts (the data itself is not stored in git)
src/neurotext/
  data/         splits, ZuCo 2.0 loader, synthetic data
  models/       encoder, text alignment, adapters
  eval/         metrics and baselines
  privacy/      identity-leakage probes and defences
  efficiency/   parameter counting, channel subsets, quantisation (no experiments on these in the report)
experiments/    one file = one experiment
results/        figures
tests/          including split-leakage tests
```

The Python package is called `neurotext` for historical reasons.

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

Smoke run on synthetic data (a pipeline check, not a scientific result):

```bash
python experiments/run.py --config configs/smoke_synthetic.yaml                  # one seed, verbose output
python experiments/run_seeds.py --config configs/smoke_synthetic.yaml --seeds 10 # repeats and confidence intervals
```

## Reproducing the results

Real data (ZuCo 2.0 in `data/raw/zuco/`; PyTorch is required for the network and the encoder):

```bash
python experiments/ridge_baseline.py     # flags --block-size, --designs, --plant
python experiments/run.py --config configs/zuco2_nr.yaml
python experiments/identity_leakage.py --config configs/zuco2_nr.yaml --seeds 10
```

Synthetic data:

```bash
python experiments/defence_tradeoff.py --config configs/defence_synthetic.yaml --seeds 10   # ~25 minutes
python experiments/plot_defence_tradeoff.py                                                 # results/defence_tradeoff.png, needs matplotlib
python experiments/adapter_sweep.py --config configs/smoke_synthetic_highvar.yaml --seeds 10 --only A F I J
python experiments/adapter_sweep.py --config configs/smoke_synthetic_highvar.yaml --start-seed 100 --seeds 20 --only A J   # check on new seeds
```

## Scope

See [ETHICS.md](ETHICS.md). Only explicit tasks (text that was read or heard) and open data collected with participants' consent. The project does **not** decode inner speech or hidden intentions, and does not try to.

## License

MIT, see [LICENSE](LICENSE). Data is not included in the repository. Dataset licences are defined by their authors.

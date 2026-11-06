# Ethical boundaries of the project

Neural data is especially sensitive: unlike a password, it cannot be changed. The project follows these rules.

1. **Explicit tasks only.** The data comes from deliberate tasks: reading or listening to a presented text. The project does not decode inner speech, hidden intentions, memories or emotional states.
2. **Consented data only.** Only public datasets collected with the informed consent of participants and distributed under their own licenses are used. Own recordings are acceptable only from volunteers with written consent.
3. **No data is published.** The repository contains no raw recordings, only scripts and instructions for obtaining the data from its owners.
4. **Privacy as a metric.** Identity leakage from embeddings is measured (`neurotext.privacy`) and reported next to decoding quality.
5. **Local processing by default.** The intended architecture decodes on the user's device; only the result is meant to leave it.
6. **No claims beyond the data.** Results are reported with baselines and limitations. Metrics on synthetic data are never presented as scientific results.
7. **Identification is measured, not performed.** Identity leakage is studied as a privacy risk, to show what an attacker could do and how much defences help. The code evaluates closed groups of consenting, de-identified volunteers in public datasets (ZuCo 2.0 and the PhysioNet EEG Motor Movement/Imagery database, EEGMMIDB). It is not meant for, and the repository contains no tooling for, linking recordings to named real people.
8. **Pre-registered claims.** Confirmatory claims are fixed in `PREREGISTRATION.md` before the held-out people are evaluated; everything else is labelled exploratory. Held-out evaluation is run once and any rerun is logged.

If you see a violation of these rules in the project, please open an issue.

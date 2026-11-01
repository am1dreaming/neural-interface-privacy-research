# Ethical boundaries of the project

Neural data is especially sensitive: unlike a password, it cannot be changed. The project follows these rules.

1. **Explicit tasks only.** The data comes from deliberate tasks: reading or listening to a presented text. The project does not decode inner speech, hidden intentions, memories or emotional states.
2. **Consented data only.** Only public datasets collected with the informed consent of participants and distributed under their own licenses are used. Own recordings are acceptable only from volunteers with written consent.
3. **No data is published.** The repository contains no raw recordings, only scripts and instructions for obtaining the data from its owners.
4. **Privacy as a metric.** Identity leakage from embeddings is measured (`neurotext.privacy`) and reported next to decoding quality.
5. **Local processing by default.** The intended architecture decodes on the user's device; only the result is meant to leave it.
6. **No claims beyond the data.** Results are reported with baselines and limitations. Metrics on synthetic data are never presented as scientific results.

If you see a violation of these rules in the project, please open an issue.

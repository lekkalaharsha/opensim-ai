# Colab sweep runner

`OpenQSim_Colab.ipynb` runs a Phase 0A benchmark sweep on Google Colab.

- **CPU-pinned** (`OPENQSIM_DEVICE=CPU`) — Aer GPU kernels don't run on Colab T4.
- **Resumes across sessions** — output + checkpoints live on Google Drive at
  `MyDrive/openqsim/benchmark_outputs`, so a disconnect/12h-cap continues, not restarts.
- **Pinned** to qiskit==1.4.2 / qiskit-aer==0.15.1 (newer Aer segfaults on qft/qaoa).

## Use

1. Open the notebook in Colab, run cells 1–6 top to bottom.
2. Set `SWEEP` in cell 4 to the band you want:
   - `sweep_config_colab.yaml` — full high band 18–28q (720)
   - `sweep_config_colab_22-28.yaml` — remaining 22/24/26/28q (480)
3. Cell 5 verifies counts and lists anything missing.

Pull the data back from `MyDrive/openqsim/benchmark_outputs/raw` and consolidate
into the local `data/phase0a/raw/` (new overlays old). The runner clones `main`,
so configs must be pushed there first.

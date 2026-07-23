# Environment notes

The two files in this directory are snapshots of development environments used
during earlier experiments:

- `hpc_environment_snapshot.txt` was exported from the GPU/HPC environment.
- `audio_environment_snapshot.txt` records a separate audio-processing setup.

They are retained for reference and are not portable lock files. In particular,
they may contain platform-specific package builds or local wheel paths. Use the
root `requirements.txt` for the portable Python dependencies, then install a
PyTorch, CUDA, `causal-conv1d`, and `mamba-ssm` combination that matches the
target machine.

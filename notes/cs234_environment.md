# CS234 Environment

Last updated: 2026-07-08

## Active Environment

Use the dedicated conda environment:

```bash
conda activate cs234
```

Location:

```text
/Users/nyvo/miniconda3/envs/cs234
```

This environment is separate from `cs336` and from the existing course-root `.venv`.

## What Goes Into The Virtual Environment

All course Python packages belong in `cs234`, not in system Python:

- Assignment 1: `numpy`
- Assignment 2: `gym`, `pybullet`, `torch`, `matplotlib`, `scipy`
- Course tooling: `pypdf` for reading lecture PDFs
- Build tooling pinned for old Gym compatibility: `pip==23.0.1`, `setuptools==65.5.0`, `wheel==0.38.4`

Important installed versions:

```text
python      3.9.23
pip         23.0.1
setuptools  65.5.0
wheel       0.38.4
numpy       1.23.0
gym         0.21.0
torch       2.8.0
matplotlib  3.9.4
scipy       1.13.1
pypdf       6.14.2
pybullet    3.25 from conda-forge
```

## System-Level Policy

No CS234 Python packages were installed into system Python.

System-level tools are only for package/environment management or non-Python command-line utilities:

- Keep Miniconda installed system-wide/user-wide.
- Keep course Python libraries inside `cs234`.
- Do not install Assignment 1/2 packages globally.
- Optional future system tool: `poppler` if robust `pdftotext`/`pdfinfo` CLI support is needed for lecture extraction.

## PyBullet Note

The original Assignment 2 requirement asks for `pybullet==3.2.6`. On this macOS arm64 machine, pip could not find a compatible wheel and tried to build from source; the build failed in clang.

Resolution:

```bash
conda install -n cs234 -c conda-forge pybullet=3.25
```

This keeps PyBullet inside the `cs234` environment while avoiding a fragile local C++ build.

## Environment Variables

The following variables are configured on the `cs234` conda environment:

```text
MPLCONFIGDIR=/tmp/cs234-matplotlib
XDG_CACHE_HOME=/tmp/cs234-cache
PYTHONNOUSERSITE=1
```

They avoid Matplotlib/font cache warnings in restricted shells and prevent user-level Python packages from leaking into this course environment.

If the environment is already active when these variables are changed, reactivate it:

```bash
conda deactivate
conda activate cs234
```

## Verification

These checks passed:

```bash
conda run -n cs234 python -m pip check
conda run -n cs234 python -B -c "import main; print('assignment2 main import ok')"
conda run -n cs234 python -c "import gym, pybullet_envs; env = gym.make('CartPoleBulletEnv-v1'); obs = env.reset(); out = env.step(env.action_space.sample()); env.close(); print('cartpole bullet ok', getattr(obs, 'shape', None), len(out))"
```

Expected PyBullet/Gym smoke-test output includes:

```text
cartpole bullet ok (4,) 4
```

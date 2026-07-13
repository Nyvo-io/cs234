# Starter code for HW3

## Environment set up

To set up the environment for this assignment, we recommend using `uv` package manager:
If you do not have `uv` installed, see:
- https://docs.astral.sh/uv/getting-started/installation/
- https://docs.astral.sh/uv/

```
uv venv --python 3.10.6
```

Then activate the environment:

macOS/Linux:
```
source .venv/bin/activate
```

Windows (Git Bash):
```
source .venv/Scripts/activate
```

Install dependencies:
```
uv pip install -r requirements.txt
```

If you would like to use `conda`:
```
conda create -n cs234_hw3 python=3.10.6
conda activate cs234_hw3
pip install -r requirements.txt
```

Note: If `pip install` prints a warning about `jupyter-client` missing dependencies
(`jupyter-core`, `tornado`, `traitlets`), this is non-blocking for this assignment
as long as the install completes successfully.

After this you are good to go!

## Submission

To generate your submission zip file, run:
```
bash collect_submission.sh
```
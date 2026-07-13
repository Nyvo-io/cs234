# Installation

* Download and uncompress this archive onto the machine you are planning to use for Assignment 2. Make sure your current working directory is the directory containing this README file.

* For this assignment, we will be using environments from PyBullet, a free and open-source physics engine. As with all the assignments, we recommend using a Python `venv` or a `conda` environment to prevent dependency issues. For this assignment, you will need to use Python 3.9. For example, to create a virtual environment using `venv`, which comes built into your Python interpreter, you can run this command to create a folder called `env`:

```bash
# venv setup 
python3 -m venv env

# conda setup
conda create -n cs234_a2 python=3.9 pip=23.0
```

In every new terminal that you use to interact with your code, you will then also need to run this command to activate your environment, from the folder where you ran the last command and created the virtual environment `env`:

```bash
# venv setup
./env/bin/activate

# conda setup 
conda activate cs234_a2
```

* Install the requirements for this assignments. With the virtual environment active, run these commands:

```bash
pip install setuptools==65.5.0 "wheel<0.40.0"
pip install -r requirements.txt
```

* Package version tip: If you run into any errors with package installation, first run `pip install pip==23.0` (to downgrade the pip version to be compatible with `gym=0.21.0`).

* If any of the above steps fail and you are unsure of how to debug the issue, please make a post on EdStem and we will try to assist you.

* After completing the assignment, you may submit your solutions to Gradescope. You can generate your submissions using:

```bash
bash collect_submission.sh
```

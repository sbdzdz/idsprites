# ♾ Infinite dSprites

Easily generate simple continual learning benchmarks. Inspired by [dSprites](https://github.com/google-deepmind/dsprites-dataset).

# insert img/shapes.gif
![A grid of 2D shapes undergoing rotation, translation, and scaling.](img/shapes.gif)

## Install
It's easiest to install the package from PyPI:
```
python -m pip install idsprites
```

Verify the installation:
```
python -c "import idsprites"
```

## Contribute
First, clone the repo:
```
git clone git@github.com:sbdzdz/idsprites.git
cd idsprites
```

It's a good idea to install the in interactive mode inside a virtual environment:
```
python -m virtualenv venv
source venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -e .
```
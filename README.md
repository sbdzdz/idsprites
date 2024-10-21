# ♾ Infinite dSprites

Easily generate simple continual learning benchmarks. Inspired by [dSprites](https://github.com/google-deepmind/dsprites-dataset).

![A grid of 2D shapes undergoing rotation, translation, and scaling.](img/shapes.gif)

## Install

Install the package from PyPI:

```bash
python -m pip install idsprites
```

Verify the installation:

```bash
python -c "import idsprites"
```

## Usage

See the [examples](examples) directory for notebooks demonstrating how to use the package.

## Contribute

Clone the repo:

```bash
git clone git@github.com:sbdzdz/idsprites.git
cd idsprites
```

It's a good idea to install the package in interactive mode inside a virtual environment:

```bash
python -m virtualenv venv
source venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -e .
```

## Citation

If you use this work in your research, please consider citing:

```
@article{dziadzio2023disentangled,
  title={Disentangled Continual Learning: Separating Memory Edits from Model Updates},
  author={Dziadzio, Sebastian and Y{\i}ld{\i}z, {\c{C}}a{\u{g}}atay and van de Ven, Gido M and Trzci{\'n}ski, Tomasz and Tuytelaars, Tinne and Bethge, Matthias},
  journal={arXiv preprint arXiv:2312.16731},
  year={2023}
}
```

Thanks!

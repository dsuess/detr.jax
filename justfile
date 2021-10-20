setup:
    poetry install
    mkdir -p .git/hooks
    ln -f -s `pwd`/hooks/* .git/hooks

setup-tpu:
    @just setup
    poetry run pip install \
        jax[tpu]==`poetry export | grep jax== | cut -d';' -f1 | cut -d'=' -f3` \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

test:
    poetry run pytest

test-models:
    poetry run pytest -m models

lint:
    poetry run isort --check jass tests
    poetry run black --check --include .py --exclude ".pyc|.pyi|.so" jass tests
    poetry run black --check --pyi --include .pyi --exclude ".pyc|.py|.so" jass tests
    poetry run pylint jass
    poetry run pyright jass tests

fix:
   poetry run isort jass tests
   poetry run black --include .py --exclude ".pyc|.pyi|.so" jass tests
   poetry run black --pyi --include .pyi --exclude ".pyc|.py|.so" jass tests

setup-cocostuff:
    mkdir -p data/cocostuff
    wget -nc http://images.cocodataset.org/zips/train2017.zip -O data/cocostuff/train2017.zip || True
    wget -nc http://images.cocodataset.org/zips/val2017.zip -O data/cocostuff/val2017.zip || True
    wget -nc http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip -O data/cocostuff/stuffthingmaps_trainval2017.zip || True
    unzip data/cocostuff/train2017.zip -d data/cocostuff/
    unzip data/cocostuff/val2017.zip -d data/cocostuff/
    unzip data/cocostuff/stuffthingmaps_trainval2017.zip -d data/cocostuff/
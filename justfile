setup:
    poetry install
    mkdir -p .git/hooks
    ln -f -s `pwd`/hooks/* .git/hooks

setup-tpu:
    @just setup
    poetry run pip install \
        jax[tpu]==`poetry export | grep jax== | cut -d';' -f1 | cut -d'=' -f3` \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

setup-cuda:
    @just setup
    poetry run pip install \
        jax[cuda]==`poetry export | grep jax== | cut -d';' -f1 | cut -d'=' -f3` \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

test:
    poetry run pytest

test-models:
    poetry run pytest -m models

lint:
    poetry run isort --check jdetr tests
    poetry run black --check --include .py --exclude ".pyc|.pyi|.so" jdetr tests
    poetry run black --check --pyi --include .pyi --exclude ".pyc|.py|.so" jdetr tests
    poetry run pylint jdetr
    poetry run pyright jdetr tests

fix:
   poetry run isort jdetr tests
   poetry run black --include .py --exclude ".pyc|.pyi|.so" jdetr tests
   poetry run black --pyi --include .pyi --exclude ".pyc|.py|.so" jdetr tests

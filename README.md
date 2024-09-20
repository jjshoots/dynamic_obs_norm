# How Do?

## Dependencies

These dependencies are required for the makefile to run:

- [Make](https://www.gnu.org/software/make/manual/make.html)
    > Ubuntu: `sudo apt install make` or alternatively: `sudo apt install build-essential`.
- [Poetry](https://python-poetry.org/docs/#installation)
    > Ubuntu: `sudo apt install python3-poetry`
- [Curl](https://curl.se/)
    > Ubuntu: `sudo apt install curl`

Additionally, you probably want to set poetry to put venvs into the root directory of the project, instead of a systemwide cache.
```
poetry config virtualenvs.in-project true
```

## Creating new projects

1. Click `use this template` and clone the resulting repository. Or optionall do the following:
    a) Make a new folder where you want the project to reside.
    b) Create a file called `Makefile` and copy everything from `Makefile` into it.
4. Run `make init`.

## Sourcing the virtual environment

You have options here:

1. Just source the venv directly:
    ```
    source ./.venv/bin/activate.fish
    ```
2. Wrap all commands in `poetry run`, e.g.:
    ```
    poetry run python3 src/main.py
    poetry run nvim
    poetry run code
    ```

## Installing additional python packages.

The following line installs dependencies and automatically adds it to the toml file.
```
poetry add {your dependency here}
```

To install a dependency without adding it to the toml file, do the usual `pip3 install X` with a poetry runtime, e.g.:
```
poetry run pip3 install {your dependency here}
```
or
```
source ./.venv/bin/activate.fish
pip3 install {your dependency here}
```

## Make commands for convenience

- For initializing a brand new project:
    ```
    make init
    ```
- For generating lock and `requirements.txt` files:
    ```
    make lock
    ```
- For running a code formatter:
    ```
    make format
    ```
- For running precommit (lock, format, tests):
    ```
    make precommit
    ```
- For installing the project (assuming the project has been just cloned from a repo with this make file):
    ```
    make install
    ```

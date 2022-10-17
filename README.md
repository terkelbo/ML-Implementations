# ML Implementations

This repo has different implementations of algorithms that are useful when doing data science. The implementations are in Python and are meant to be used as a reference for people who are learning about the algorithms. The implementations are not meant to be used in production.

# Development

The project uses python 3.10 and can be setup using the following command:

`make dev-setup`

The project is optimised for working on VS code and a settings file is included in the git repo. The project uses `black` for formatting, `pylint` for linting and `isort` for import sorting. The project also uses `pytest` for testing. The project uses `pre-commit` to run the formatting and linting checks before every commit. The project uses `poetry` for dependency management.

The project uses tox for testing. You can run the tests by running the command `tox` in the root of the project.

# News Signals

#### New environment

Run `conda create -n <env-name> python=3.8` if you're using Anaconda, alternatively `python3.8 -m venv <env-path>`

Activate: <br>
Run `conda activate <env-name>` or `source <env-path>/bin/activate`

#### Install library
Run `make dev`

## Create New Project

To create an new empty project, pick a project directory and a name for the project's Python package, and run:

`PROJ_DIR=my_project PKG_NAME=my_lib make new-project`

This will create an empty new project from scratch, including all of the default components.

Here is a checklist to turn the new project into a fully functional tool:
- [ ] implement your project's core functionality in the Python package
- [ ] maintain dependencies in `requirements.txt`
- [ ] implement a demo
- [ ] implement service

### New Demo
Within a project, you can initialize a new demo as follows: <br>
`DEMO_NAME=mydemo make new-demo`

A demo directory with the given name and running streamlit skeleton will be created in [/demos](demos).

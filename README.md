## Managing Dependencies with  `poetry`

This boilerplate uses `poetry` as the dependecy and package manager, which can be used together wiht environment manager, such as `conda` or `virtualenv`. 

1. Create a virtual environemnt for the project using the preferred environment manager (e.g. `conda`).
2. Install `poetry` in the created environment using `pip install poetry`
3. Install dependencie listed in `poetry.lock` file. These dependecies are needed in this boilerplate for testing and checking. This is done by running 
``` 
poetry install 
```

### Install new dependecies
```
poetry add [PACKAGE]
```
This command will install the package to the virtual environment and add the package information to `poetry.lock` and `pyproject.toml` files. 


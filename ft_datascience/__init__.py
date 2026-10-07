# inside a directory, an init.py file is used to define the repository as a library
# put this at the root of our lib for data science, first we need to find a namy name

# put in brakets the name of our files
__all__ = ['data']

# same here, names of the files
from . import data


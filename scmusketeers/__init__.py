import logging
from importlib.metadata import version

__version__ = version("sc-musketeers")

# print("import arguments")
from . import arguments
# print("import tools")
from . import tools
# print("import transfer")
from . import transfer
# print("import hptoptim")
from . import hpoptim
#print("__main__")
from . import __main__

# import all classes from the scripts in this folder
import os

# get list of modules
modules = os.listdir(os.path.dirname(__file__))

#import all classes from modules
for module in modules:
    if module == '__init__.py' or module[-3:] != '.py':
        continue

    path2module = "src.models.transformers.components.base." + module[:-3]

    __import__(path2module, locals(), globals())
    exec(f"from {path2module} import *")

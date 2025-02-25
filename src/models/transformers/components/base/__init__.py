# import all classes from the scripts in this folder
import os
import inspect
import importlib

BASE_COMPONENT_REGISTRY = {}

# get list of modules
modules = [
    module_name[:-3] for module_name in os.listdir(os.path.dirname(__file__))
    if module_name != '__init__.py' and module_name[-3:] == '.py'
]

#import all classes from modules
for module_name in modules:

    module = importlib.import_module(f".{module_name}", package=__name__)

    #path2module = "src.models.transformers." + module_name[:-3]
    #__import__(path2module, locals(), globals())
    #exec(f"from {path2module} import *")

    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Ensure the class is defined in the module (not imported)
        if obj.__module__ == module.__name__:
            BASE_COMPONENT_REGISTRY[name] = obj  # Store class reference

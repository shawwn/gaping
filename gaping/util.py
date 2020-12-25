import re
import importlib
import types
import inspect
import os
import sys

from subprocess import call
from shlex import quote as shellquote

from collections import OrderedDict
from distutils.util import strtobool
from typing import Any, List, Tuple, Union


# Util classes
# ------------------------------------------------------------------------------------------


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""
    if '/' in obj_name:
      obj_name, local_name = obj_name.rsplit('/', 1)
      module, local_obj_name = get_module_from_obj_name(obj_name)
      return module, os.path.join(local_obj_name, local_name)

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name) # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
        except ImportError:
            pass

    if '.' not in obj_name and '/' not in obj_name:
      return get_module_from_obj_name('__main__.' + obj_name)

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.replace('/', '.').split("."):
        obj = getattr(obj, part)
    return obj


def constant_name(name: str) -> bool:
    return isinstance(name, str) and len(name) > 0 and name[0] == ':'


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    if constant_name(name):
      return name
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_indirect(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def get_module_file_by_obj_name(obj_name: str) -> str:
    """Get the file path of the module containing the given object name."""
    module, _ = get_module_from_obj_name(obj_name)
    return inspect.getfile(module)


def get_module_dir_by_obj_name(obj_name: str) -> str:
    """Get the directory path of the module containing the given object name."""
    return os.path.dirname(get_module_file_by_obj_name(obj_name))


def is_top_level_function(obj: Any) -> bool:
    """Determine whether the given object is a top-level function, i.e., defined at module scope using 'def'."""
    return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__


def get_top_level_function_name(obj: Any) -> str:
    """Return the fully-qualified name of a top-level function."""
    assert is_top_level_function(obj)
    return obj.__module__ + "." + obj.__name__

class CyclicIndirection(Exception):
  pass

def get_indirect(name):
  if constant_name(name):
    return name
  tortoise = hare = name
  while True:
    if not isinstance(hare, str):
      break
    hare = get_obj_by_name(hare)
    if not isinstance(hare, str):
      break
    hare = get_obj_by_name(hare)
    tortoise = get_obj_by_name(tortoise)
    if hare == tortoise:
      raise CyclicIndirection()
  return hare

def get_obj_name(obj: Any) -> str:
  """Return the fully-qualified name of the object."""
  if isinstance(obj, str):
    obj = get_indirect(obj)
  if inspect.ismodule(obj):
    return obj.__name__
  mod = inspect.getmodule(obj)
  if mod is None:
    raise ValueError("Couldn't get module for {!r}".format(obj))
  if not hasattr(obj, '__qualname__'):
    obj = type(obj)
  if not hasattr(obj, '__qualname__'):
    raise ValueError("Couldn't get qualified name for {!r} in module {!r}".format(obj, mod.__name__))
  return os.path.join(mod.__name__, obj.__qualname__)

def find_objs(name: str, ignore_case: str = 'smart') -> Tuple[str, Any]:
  name = re.compile(name)
  if ignore_case == 'smart':
    ignore_case = not re.search('[A-Z]', name.pattern)
  if ignore_case:
    name = re.compile(name.pattern, flags=name.flags | re.IGNORECASE)
  h = OrderedDict()
  for modname, module in sys.modules.items():
    for k, v in module.__dict__.items():
      if re.match(name, k) and re.match(name, getattr(v, '__name__', '')):
        fqn = get_obj_name( v )
        h[fqn] = v
  return h

def edit(obj):
  name = get_obj_name( obj )
  filename = get_module_file_by_obj_name( name )
  editor = os.environ.get('EDITOR', 'vim')
  #call([editor, filename])
  # https://stackoverflow.com/questions/1196074/how-to-start-a-background-process-in-python
  #os.spawnl(os.P_NOWAIT, editor, filename)
  os.system(editor + ' ' + shellquote(filename) + ' &')


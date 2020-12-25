import collections.abc
from typing import Any, List, Tuple, Union
import re
import inspect
import sys

NoneType = None.__class__

def numeric(x: str) -> bool:
  if not isinstance(x, str):
    return False
  for c in x:
    if not ('0' <= c <= '9'):
      return False
  return True

def length(x: Any, upto: int = None) -> int:
  if isinstance(x, NoneType):
    return 0
  if isinstance(x, collections.abc.Mapping):
    n = 0
    for k, v in x.items():
      if numeric(k):
        k = int(k)
      if isinstance(k, int):
        n = max(n, k + 1)
        if upto is not None and n > upto:
          return n
    return n
  try:
    return len(x)
  except TypeError:
    pass
  return sys.maxsize

def none(x): return length(x, 0) <= 0
def some(x): return length(x, 0) >= 1
def one(x): return length(x, 1) == 1
def two(x): return length(x, 2) == 2

def symbolp(x: Any) -> bool:
  return isinstance(x, str) or nilp(x)

def keywordp(x: Any) -> bool:
  return isinstance(x, str) and at(x, 0) == ':'

def stringp(x: Any) -> bool:
  return isinstance(x, (str, bytes))

def listp(x: Any) -> bool:
  return not stringp(x) and isinstance(x, (collections.abc.Sequence, collections.abc.Mapping, NoneType))

def consp(x: Any) -> bool:
  return listp(x) and some(x)

def atom(x: Any) -> bool:
  return not consp(x)

def nilp(x: Any) -> bool:
  return none(x)

def at(l, i, default=None):
  if some(l):
    try:
      return l[i]
    except IndexError:
      pass
  return default

def functionp(x):
  return inspect.isfunction(x)

def testify(x, compare=None):
  if functionp(x):
    return x
  if compare is not None:
    return lambda y: compare(x, y)
  return lambda y: x == y

def hd(l, test=None):
  x = at(l, 0)
  if test is not None:
    return testify(test)(x)
  return x

def tl(l):
  return l[1:]

def assq(v, l):
  while True:
    if nilp(l):
      return l
    x = hd(l)
    if hd(x, v):
      return x
    l = tl(l)

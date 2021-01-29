from gaping import el
from gaping import util
import inspect
import functools
from types import SimpleNamespace as NS
from copy import copy
import sys

Q = globals().get('Q', util.EasyDict())
V = globals().get('V', util.EasyDict())
F = globals().get('F', util.EasyDict())
M = sys.modules.get('__main__', util.EasyDict())

V.internal_interpreter_environment = V.get('internal_interpreter_environment', [])
V.initial_obarray = V.get('initial_obarray', [])

class LispSymbol(NS):
  pass

def symbol_name(x):
  return x.__name__

def symbol_plist(x):
  return x.__dict__

def make_symbol(name):
  sym = LispSymbol()
  sym.__name__ = name
  sym.function = None
  sym.value = None
  return sym

def intern_soft(name, obarray=None):
  if obarray is None:
    obarray = V.initial_obarray
  for v in obarray:
    if symbol_name(v) == name:
      return v

def intern(name, obarray=None):
  if obarray is None:
    obarray = V.initial_obarray
  v = intern_soft(name, obarray=obarray)
  if v is not None:
    return v
  v = make_symbol(name)
  obarray.append(v)
  return v

Lisp_Fwd_Obj = 'Lisp_Fwd_Obj'
SYMBOL_FORWARDED = 'SYMBOL_FORWARDED'

def SET_SYMBOL_FWD(sym, fwd):
  pass

def staticpro(address):
  pass

def defvar_lisp_nopro(o_fwd, namestring, address):
  sym = intern(namestring)
  o_fwd.type = Lisp_Fwd_Obj
  o_fwd.objvar = address
  sym.declared_special = True
  sym.redirect = SYMBOL_FORWARDED
  SET_SYMBOL_FWD(sym, o_fwd)

def defvar_lisp(o_fwd, namestring, address):
  devar_lisp_nopro(o_fwd, namestring, address)
  staticpro(address)

import keyword

def compile_id(name):
  out = []
  for c in name:
    if c == '-':
      c = '_'
    if c.isalnum() or c in ['_']:
      out += [c]
    else:
      out += [str(ord(c))]
  out = ''.join(out)
  if out:
    if out[0].isdigit():
      out = '__' + out
    if keyword.iskeyword(out):
      out = out + '_'
  return out

def DEFSYM(name, value):
  k = compile_id(name)
  if Q[k] is None:
    Q[k] = make_symbol(name)


Q.nil = Q.get('nil', None)
Q.t = Q.get('t', True)

UNEVALLED = 'UNEVALLED'
MANY = 'MANY'

"""
bool
FUNCTIONP (Lisp_Object object)
{
  if (SYMBOLP (object) && !NILP (Ffboundp (object)))
    {
      object = Findirect_function (object, Qt);

      if (CONSP (object) && EQ (XCAR (object), Qautoload))
	{
	  /* Autoloaded symbols are functions, except if they load
	     macros or keymaps.  */
	  for (int i = 0; i < 4 && CONSP (object); i++)
	    object = XCDR (object);

	  return ! (CONSP (object) && !NILP (XCAR (object)));
	}
    }

  if (SUBRP (object))
    return XSUBR (object)->max_args != UNEVALLED;
  else if (COMPILEDP (object) || MODULE_FUNCTIONP (object))
    return true;
  else if (CONSP (object))
    {
      Lisp_Object car = XCAR (object);
      return EQ (car, Qlambda) || EQ (car, Qclosure);
    }
  else
    return false;
}
"""

def SYMBOLP(x): return el.symbolp(x)
def INTP(x): return isinstance(x, int)
def CONSP(x): return el.consp(x)
def NILP(x): return el.nilp(x)
def NO(x): return x is False or el.nilp(x)
def YES(x): return x is True or not NO(x)
def SUBRP(x): return inspect.isbuiltin(x) or inspect.isfunction(x) or inspect.ismethod(x) or inspect.isclass(x)
def XSUBR_min_args(x): return getattr(x, 'min_args', 0)
def XSUBR_max_args(x): return getattr(x, 'max_args', MANY)
def XCAR(x): return el.hd(x)
def XCDR(x): return el.tl(x)

def FUNCTIONP(obj):
  if SYMBOLP(obj) and YES(fboundp(obj)):
    obj = indirect_function(obj)

def symbol_value(x):
  return util.get_indirect(x)

def indirect_function(f):
  return util.get_indirect(f)

def eval_sub(form, *, macroexpanding=False, **kws):
  if accessor_literal_p(form):
    return form
  if el.symbolp(form):
    # look up its binding in the lexical environment.
    lex_binding = el.assq(form, V.internal_interpreter_environment)
    if el.consp(lex_binding):
      return el.at(lex_binding, 1)
    else:
      return symbol_value(form)
  if not el.consp(form):
    return form
  original_fun = XCAR(form)
  original_args = XCDR(form)
  fun = indirect_function(original_fun)
  mac = CONSP(fun) and el.hd(fun, 'macro')
  if mac:
    fun = indirect_function(XCAR(XCDR(fun)))
  elif macroexpanding:
    return form
  args_left = [arg for arg in original_args]
  if CONSP(fun):
    fun = eval_sub(fun)
  if not SUBRP(fun):
    import pdb; pdb.set_trace()
    raise NotImplementedError()
  else:
    maxargs = XSUBR_max_args(fun)
    if maxargs == UNEVALLED or mac:
      val = fun(*args_left, **kws)
    elif maxargs == MANY:
      vals = []
      while CONSP(args_left):
        arg = XCAR(args_left)
        if el.keywordp(arg):
          k = arg[1:]
          args_left = XCDR(args_left)
          arg = XCAR(args_left)
          args_left = XCDR(args_left)
          kws[k] = eval_sub(arg)
        else:
          args_left = XCDR(args_left)
          vals.append(eval_sub(arg))
      val = fun(*vals, **kws)
    else:
      import pdb; pdb.set_trace()
      raise NotImplementedError()
      argvals = []
      for i in range(maxargs):
        argvals.append(eval_sub(XCAR(args_left)))
        args_left = XCDR(args_left)
      val = fun(*argvals, **kws)
    if mac and not macroexpanding:
      val = eval_sub(val)
    return val

def getenv(symbol, property):
  return Q.nil

def error(msg):
  raise ValueError(msg)

def accessor_literal_p(x):
  return el.stringp(x) and \
      el.at(x, 0) == "." and \
      el.at(x, 1) != "." and \
      el.some(x)

def id_literal_p(x):
  return el.stringp(x) and el.at(x, 0) == "|"

def get_place(place, setfn):
  place = macroexpand_1(place)
  if el.atom(place) or \
      (el.hd(place, "get") and el.nilp(getenv("get", "place-expander"))) or \
      accessor_literal_p(el.at(place, 1)):
    return setfn(place, lambda v: ["%set", place, v])
  else:
    head = el.hd(place)
    gf = getenv(head, "place-expander")
    if not el.nilp(gf):
      return apply(gf, setfn, el.tl(place))
    else:
      return error(str(place) + " is not a valid place expression")


def assign(name, value):
  # look up its binding in the lexical environment.
  lex_binding = el.assq(name, V.internal_interpreter_environment)
  if el.consp(lex_binding):
    lex_binding[1] = value
  else:
    mod, key = name.rsplit('.', 1) if '.' in name else ('__main__', name)
    main = util.get_indirect(mod)
    main.__dict__[key] = value
  return value

def DEFUN(name, max_args = MANY):
  def func(fn):
    fn.max_args = max_args
    F[compile_id(name)] = assign(name, fn)
    return fn
  return func

def DEFMACRO(name, max_args = MANY):
  def func(fn):
    fn.max_args = max_args
    F[compile_id(name)] = assign(name, ['macro', fn])
    return fn
  return func

@DEFUN('list')
def list_(*args, **kws):
  args = list(args)
  for k, v in kws.items():
    args.append(':' + k)
    args.append(v)
  return args

@DEFUN('eval')
def eval(x, **kws):
  # x = util.get_indirect(x)
  # if el.consp(x):
  #   return eval_sub(x, **kws)
  x = eval_sub(x, **kws)
  return x

@DEFUN('apply')
def apply(f, *args, **kws):
  if len(args) > 0:
    args = [*args[0:-1], *args[-1]]
  # return util.call_func_by_name(*args, **kws, func_name=f)
  return F.eval(F.list(f, *args, **kws))

@DEFUN('call')
def call(f, *args, **kws):
  return F.apply(f, args, **kws)

@DEFUN('macroexpand-1')
def macroexpand_1(form):
  return eval_sub(form, macroexpanding=True)

@DEFMACRO('get')
def get__macro(*args):
  return ['%get', *args]

def maybe_number(x):
  if el.stringp(x) and el.at(x, 0) == '-' and el.numeric(x[1:]):
    return int(x)
  if el.numeric(x):
    return int(x)
  return x


import contextlib

@contextlib.contextmanager
def to(out):
  def pr(x, *args):
    out.append(x)
    for arg in args:
      out.append(arg)
    return x
  yield pr
  return out

def search(s, char, pos=0):
  try:
    return s.index(char, pos)
  except ValueError:
    return

from collections import namedtuple
from types import SimpleNamespace as NS
import re

#class Stream(namedtuple('Stream', 'string pos len pending')):
class Stream(NS):
  def __init__(self, string, pos=0, length=None, **kws):
    if length is None:
      length = len(string)
    super().__init__(string=string, start=kws.pop('start', pos), pos=pos, length=length, pending=[], **kws)
    self.match_data = []

  def peek(self):
    if self.pending:
      return self.pending[-1]
    if self.pos < self.length:
      return el.at(self.string, self.pos)

  def read(self):
    if self.pending:
      return self.pending.pop()
    if self.pos < self.length:
      c = el.at(self.string, self.pos)
      self.pos += 1
      return c

  def unread(self, x):
    self.pending.append(x)
  def __call__(self, unread=None):
    if unread is not None:
      self.unread(unread)
    else:
      return self.read()

  @contextlib.contextmanager
  def save_excursion(self):
    pos = self.pos
    try:
      yield
    finally:
      self.pos = pos

  @property
  def contents(self):
    return self.string[self.pos:self.length]

  def re_search_forward(self, regexp, limit=None, noerror=None, count=None):
    match = re.search(regexp, self.contents)
    if match is None:
      self.match_data = []
    else:
      matchdata = []
      for start, end in match.regs:
        start += self.pos
        end += self.pos
        matchdata.extend([start, end])
      self.match_data = matchdata

def stream(x, pos=0, length=None, **kws):
  return Stream(string=x, pos=pos, length=length, **kws)

def search_until(s, char, pos=1, backspace=True):
  i = search(s, char, pos)
  while i is not None and backspace and el.at(s, i-1) == '\\':
    i = search(s, char, i+1)
  return i

def parse_accessor(key, out=None):
  if out is None:
    out = []
  if accessor_literal_p(key):
    i = search(key, '.', 1)
    while i is not None and el.at(key, i-1) == '\\':
      i = search(key, '.', i+1)
    out.append(maybe_number(key[1:i]))
    if i is not None:
      key = key[i:]
    else:
      key = ''
    return parse_accessor(key, out)
  if key:
    out.append(maybe_number(key))
  return out

def accessor(key):
  return parse_accessor(key)


@DEFUN('%get')
def get__special(place, key):
  if not el.stringp(key) and not INTP(key):
    raise ValueError("%get expected an index for key: {}".format(['%get', place, key]))
  parts = accessor(key)
  while parts:
    key, *parts = parts
    if INTP(key):
      place = el.at(place, key)
    else:
      place = util.get_obj_from_module(place, key)
  return place


@DEFUN('setq', UNEVALLED)
def setq(*args):
  while CONSP(args):
    name = XCAR(args)
    args = XCDR(args)
    value = F.eval(XCAR(args))
    args = XCDR(args)
    assign(name, value)
  return value

@DEFUN('quote', UNEVALLED)
def quote(x):
  return x

def eval_body(body):
  if len(body) >= 1:
    for expr in body[:-1]:
      F.eval(expr)
    return F.eval(body[-1])

@DEFUN('progn', UNEVALLED)
def progn(*body):
  return eval_body(body)

@DEFUN('prog1', UNEVALLED)
def prog1(x, *body):
  val = F.eval(x)
  eval_body(body)
  return val

@DEFUN('if', UNEVALLED)
def if_(cond, then, *else_):
  if YES(F.eval(cond)):
    return F.eval(then)
  return eval_body(else_)

@DEFUN('or', UNEVALLED)
def or_(*args):
  while CONSP(args):
    arg = F.eval(XCAR(args))
    if YES(arg):
      return arg
    args = XCDR(args)

@DEFUN('and', UNEVALLED)
def and_(*args):
  arg = None
  while CONSP(args):
    arg = F.eval(XCAR(args))
    args = XCDR(args)
    if NO(arg):
      return arg
  return arg

@DEFUN('let', UNEVALLED)
def let(bindings, *body):
  env = V.internal_interpreter_environment
  before = copy(env)
  try:
    while CONSP(bindings):
      slot = XCAR(bindings)
      name = XCAR(slot)
      value = F.eval(XCAR(XCDR(slot)))
      env.insert(0, [name, value])
      bindings = XCDR(bindings)
    return eval_body(body)
  finally:
    env[:] = before

@DEFMACRO('when')
def when(cond, *body):
  return ['if', cond, ['progn', *body]]

@DEFMACRO('unless')
def unless(cond, *body):
  return ['if', cond, [], *body]

@DEFUN('prn')
def prn(x, *args):
  print(x, *args)
  return x

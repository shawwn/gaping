from gaping import el
from gaping import util
import inspect
import functools
from types import SimpleNamespace as NS
from copy import copy

Q = globals().get('Q', util.EasyDict())
V = globals().get('V', util.EasyDict())
F = globals().get('F', util.EasyDict())

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

def compile_id(name):
  s = [c if c.isalpha() else '_' for c in name[0:1]]
  s += [c if c.isalnum() else '_' for c in name[1:]]
  s = ''.join(s)
  return s

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

def eval_sub(form, **kws):
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
  args_left = [arg for arg in original_args]
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
    if mac:
      val = eval_sub(val)
    return val

def eval(x, **kws):
  # x = util.get_indirect(x)
  # if el.consp(x):
  #   return eval_sub(x, **kws)
  x = eval_sub(x, **kws)
  return x

def apply(f, *args, **kws):
  if len(args) > 0:
    args = [*args[0:-1], *args[-1]]
  return util.call_func_by_name(*args, **kws, func_name=f)

def call(f, *args, **kws):
  return apply(f, args, **kws)

def assign(name, value):
  main = util.get_indirect('__main__')
  main.__dict__[name] = value
  return value

def DEFUN(name, max_args = MANY):
  def func(fn):
    fn.max_args = max_args
    assign(name, fn)
    return name
  return func

def DEFMACRO(name, max_args = MANY):
  def func(fn):
    fn.max_args = max_args
    assign(name, ['macro', fn])
    return name
  return func

@DEFUN('setq', UNEVALLED)
def setq(*args):
  while CONSP(args):
    name = XCAR(args)
    args = XCDR(args)
    value = eval(XCAR(args))
    args = XCDR(args)
    assign(name, value)
  return value

@DEFUN('quote', UNEVALLED)
def quote(x):
  return x

def eval_body(body):
  if len(body) >= 1:
    for expr in body[:-1]:
      eval(expr)
    return eval(body[-1])

@DEFUN('progn', UNEVALLED)
def progn(*body):
  return eval_body(body)

@DEFUN('prog1', UNEVALLED)
def prog1(x, *body):
  val = eval(x)
  eval_body(body)
  return val

@DEFUN('if', UNEVALLED)
def if_(cond, then, *else_):
  if YES(eval(cond)):
    return eval(then)
  return eval_body(else_)

@DEFUN('or', UNEVALLED)
def or_(*args):
  while CONSP(args):
    arg = eval(XCAR(args))
    if YES(arg):
      return arg
    args = XCDR(args)

@DEFUN('and', UNEVALLED)
def and_(*args):
  arg = None
  while CONSP(args):
    arg = eval(XCAR(args))
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
      value = eval(XCAR(XCDR(slot)))
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

#!/usr/bin/env python3
# Copyright (c) 2009 Denis Bilenko. See LICENSE for details.
# gevent-test-requires-resource: network
"""Spawn multiple workers and wait for them to complete"""
from gevent import monkey

# patches stdlib (including socket and ssl modules) to cooperate with other greenlets
monkey.patch_all()

import gevent
import tqdm
from gevent.pool import Pool

import requests
import httpx
import httpcore
#from gaping.httpx import transports


def fetch(client, url, pbar, bbar, ibar, ebar):
    def fail(e, rethrow=None):
      ebar.update(1)
      if rethrow is None:
        rethrow = not isinstance(e, (httpx.ReadError, httpcore.ReadError, RuntimeError, ValueError, StopIteration))
      msg = str(e)
      noisy = True
      if any([squelch in msg for squelch in [
          'Read on closed or unwrapped SSL socket',
          'File descriptor was closed in another greenlet']]):
        noisy = False
      if isinstance(e, (StopIteration, KeyboardInterrupt)):
        noisy = False
      if noisy:
        #pbar.write('%s: failed: %s(%s)' % (url, str(e.__class__), repr(str(e))))
        pbar.write('%s: failed: %s' % (url, repr(e)))
      if rethrow:
        raise e
    #pbar.write('Starting %s' % url)
    ibar.update(1)
    try:
      #req = requests.get(url)
      #with client.stream('GET', url) as req:
      req = client.get(url)
      n = 1
      if True:
        if req.status_code != 200:
          pbar.write('%s: failed: %d' % (url, req.status_code))
          ebar.update(1)
          total_bytes = 0
          n = 0
        elif True:
          total_bytes = len(req.content)
        else:
          total_bytes = 0
          for i, data in enumerate(req.stream):
            #data = req.content
            #pbar.write('%s: %s bytes: %r' % (url, len(data), data[:50]))
            total_bytes += len(data)
            #bbar.update(len(data))
            #if i % 1 == 0:
            #  gevent.sleep()
        bbar.update(total_bytes)
        pbar.update(n)
    except Exception as e:
      return fail(e)
    finally:
      ibar.update(-1)
      if pbar.n % 50 == 0:
        ebar.refresh()
        pbar.refresh()
        bbar.refresh()

def request_urls(url):
  r = requests.get(url, stream=True)
  for line in r.iter_lines(decode_unicode=True):
    yield line
    #gevent.sleep()


from argparse import ArgumentParser

def prepare_parser():
  usage = 'Parser for all scripts.'
  parser = ArgumentParser(description=usage)
  return parser

def add_fetch_parser(parser):
  parser.add_argument(
      'urls', type=str, default='https://battle.shawwn.com/sdc/imagenet/validation_urls.txt', nargs='?',
      help='A url of a file containing image urls')
  parser.add_argument(
      '-p', '--parallel', type=int, default=10,
      help='Max simultaneous in-flight requests')
  parser.add_argument(
      '-t', '--total', type=int, default=None,
      help='Total number of urls in the urls file')
  parser.add_argument(
      '--retries', type=int, default=3,
      help='Total number of retries per url')
  parser.add_argument(
      '--timeout', type=int, default=30,
      help='Timeout per url')
  return parser

def fetch_urls(urls):
  if isinstance(urls, str):
    urls = request_urls(args.urls)
  pool = Pool(args.parallel)
  with \
      tqdm.tqdm(position=0, unit=' succeeded', total=args.total, dynamic_ncols=True) as pbar, \
      tqdm.tqdm(position=1, unit_scale=True, unit='B', dynamic_ncols=True) as bbar, \
      tqdm.tqdm(position=2, unit=' failed', dynamic_ncols=True) as ebar, \
      tqdm.tqdm(position=3, unit=' in flight', dynamic_ncols=True) as ibar, \
      httpx.Client(
          #transport=transports.HTTPTransport(retries=args.retries),
          #limits=httpx.Limits(max_keepalive_connections=args.parallel*2, max_connections=args.parallel*2),
          limits=httpx.Limits(max_keepalive_connections=None, max_connections=None),
          #limits=httpx.Limits(max_keepalive_connections=100, max_connections=100),
          timeout=httpx.Timeout(args.timeout),
          ) as client:
    try:
      for i, url in enumerate(urls):
        pool.spawn(fetch, client, url, pbar, bbar, ibar, ebar)
      pool.join()
    except KeyboardInterrupt:
      pass

import mock
import threading
from types import SimpleNamespace as NS
from collections import OrderedDict
import functools
import contextlib

mocks = globals().get('mocks') or NS(advice=OrderedDict({}), deactivate=None, lock=threading.RLock())

def mocks_active():
  return mocks.deactivate is not None

def mock_function(unique_name, lib, name=None, doc=None):
  return mock_method(unique_name, lib, name=name, doc=doc, toplevel=True)

def mock_method(unique_name, cls, name=None, doc=None, toplevel=False):
  def func(fn):
    nonlocal name
    if name is None:
      name = fn.__name__
    wrapped = getattr(cls, name)
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
      return fn(wrapped, cls, *args, **kwargs)
    if hasattr(wrapped, '__name__'):
      _fn.__name__ = wrapped.__name__
    if hasattr(wrapped, '__module__'):
      _fn.__module__ = wrapped.__module__
    if hasattr(wrapped, '__qualname__'):
      _fn.__qualname__ = wrapped.__qualname__
    if toplevel:
      mocks.advice[unique_name] = lambda: mock.patch.object(cls, name, side_effect=_fn)
    else:
      mocks.advice[unique_name] = lambda: mock.patch.object(cls, name, _fn)
    return _fn
  return func

def deactivate_mocks():
  with mocks.lock:
    if mocks.deactivate:
      mocks.deactivate()
      mocks.deactivate = None
      return True

def activate_mocks():
  with mocks.lock:
    deactivate_mocks()
    with contextlib.ExitStack() as stack:
      for creator in mocks.advice.values():
        stack.enter_context(creator())
      stk = stack.pop_all()
      mocks.deactivate = stk.close
      return stk


@mock_method('patch_set_cookie_header', httpx._models.Cookies, 'set_cookie_header',
    doc="""Disable cookies for performance""")
def patch_set_cookie_header(*args, **kws):
  #import pdb; pdb.set_trace()
  #print('patched')
  #sys.stdout.flush()
  pass

import http.cookiejar

@mock_method('patch_set_cookie', http.cookiejar.CookieJar, 'set_cookie',
    doc="""Disable cookies for performance""")
def patch_set_cookie(*args, **kws):
  #import pdb; pdb.set_trace()
  #print('patched')
  #sys.stdout.flush()
  pass


if __name__ == '__main__':
  import sys
  from gaping.util import EasyDict
  # parse command line and run    
  parser = prepare_parser()
  parser = add_fetch_parser(parser)
  args = vars(parser.parse_args())
  args = EasyDict(args)
  activate_mocks()
  fetch_urls(args.urls)

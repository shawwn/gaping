import asyncio
import inspect
import nest_asyncio
nest_asyncio.apply()

import functools
import contextvars
import asyncio.events

from pprint import pprint as pp
import time

from gaping.util import EasyDict

def err(msg, *args, **kws):
  raise ValueError('%s%s%s' % (msg,
    (': '+' '.join([repr(x) for x in args])) if len(args) > 0 else '',
    ((' '+repr(kws)) if len(kws) > 0 else '')))

async def to_thread(func, *args, **kwargs):
    """Asynchronously run function *func* in a separate thread.
    Any *args and **kwargs supplied for this function are directly passed
    to *func*. Also, the current :class:`contextvars.Context` is propogated,
    allowing context variables from the main thread to be accessed in the
    separate thread.
    Return a coroutine that can be awaited to get the eventual result of *func*.
    """
    loop = asyncio.events.get_running_loop()
    ctx = contextvars.copy_context()
    def thunk():
      asyncio.set_event_loop(loop)
      return ctx.run(func, *args, **kwargs)
    return await loop.run_in_executor(None, thunk)

def is_thunk(x):
  return inspect.isfunction(x) and str(inspect.signature(x)) == '()'

def awaitable(x):
  return asyncio.isfuture(x) or asyncio.iscoroutine(x)

def get_event_loop():
  return asyncio.get_event_loop()

async def calling(f, *args, **kws):
  return f(*args, **kws)

async def returner(result=None):
  return result

def gather(t):
  t = fut(t)
  if isinstance(t, (tuple, list, set)):
    return asyncio.gather(*t)
  return t

def fut(t):
  if isinstance(t, (tuple, list, set)):
    #return asyncio.gather(*[fut(x) for x in t])
    return [fut(x) for x in t]
  if is_thunk(t):
    t = t()
  if awaitable(t):
    return asyncio.ensure_future(t)
  else:
    return asyncio.ensure_future(returner(t))

def wait(t, parallel=True):
  if isinstance(t, (tuple, list, set)):
    if parallel:
      t = [fut(x) for x in t]
      return [wait(x) for x in t]
    else:
      return [wait(fut(x)) for x in t]
  if awaitable(t):
    return get_event_loop().run_until_complete(t)
  else:
    return t

def wait1(tasks, results=None):
  done, running = wait(asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED))
  if results is None:
    results = []
  for task in done:
    results.append(task.result())
  return running, results

def do(*t):
  results = wait(t, parallel=False)
  if len(results) > 0:
    return results[-1]

async def process(fn, items, *, parallel=None):
  tasks = set()
  if inspect.isasyncgen(items):
    async for item in items:
      item = wait(item)
      tasks.add(fut(fn(item)))
      if parallel is not None and len(tasks) >= parallel:
        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
          yield task.result()
  else:
    items = fut(items)
    for item in items:
      item = wait(item)
      tasks.add(fut(fn(item)))
      if parallel is not None and len(tasks) >= parallel:
        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
          yield task.result()
  while tasks:
    done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
      yield task.result()

async def pmap(fn, items, callback=None, *, parallel=None):
  results = []
  async for result in process(fn, items, parallel=parallel):
    if callback is not None:
      result = callback(result)
    results.append(result)
  return results

def ayield():
  wait(returner())

def cancel_everything():
  [task.cancel() for task in asyncio.tasks.all_tasks(get_event_loop())]
  wait(returner())

import httpx

def _trim_line(line):
  if line.endswith('\n'):
    line = line[:-1]
  if line.endswith('\r'):
    line = line[:-1]
  return line

async def fetch_lines(client, url):
  async with client.stream('GET', url) as req:
    async for line in req.aiter_lines():
      line = _trim_line(line)
      yield line

async def fetch(client, url):
  try:
    response = await client.get(url)
    if response.status_code == 200:
      return None, EasyDict(url=url, response=response, error=None)
    else:
      try:
        await response.raise_for_status()
      except Exception as e:
        return e, EasyDict(url=url, response=response, error=e)
  except Exception as e:
    return e, EasyDict(url=url, response=None, error=e)

from functools import partial
import traceback

def assert_ok(e):
  if e is not None:
    return False
    try:
      raise e
    except:
      traceback.print_exc()
    return False
  else:
    return True

def make_client(timeout=60.0):
  return httpx.AsyncClient(
      limits=httpx.Limits(max_keepalive_connections=None, max_connections=None),
      timeout=timeout)

async def apop(tasks):
  done, running = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
  for task in done:
    tasks.remove(task)
    err, result = task.result()
    assert_ok(err)
    yield result

async def apush(tasks, t, parallel=None):
  t = fut(t)
  tasks.add(t)
  if parallel is not None:
    while len(tasks) >= parallel:
      async for result in apop(tasks):
        yield result

async def adrain(tasks):
  while tasks:
    async for result in apop(tasks):
      yield result

async def fetch_files(client, urls_file, parallel=None, tasks=None):
  if tasks is None:
    tasks = set()
  async for url in fetch_lines(client, urls_file):
    async for result in apush(tasks, fetch(client, url), parallel=parallel):
      yield result
  while tasks:
    async for result in apop(tasks):
      yield result


def is_remote_url(url):
  #info = urlparse(url)
  #return bool(info.scheme and info.netloc)
  return url.startswith('http://') or url.startswith('https://')

import requests

def fetch_url_content_length(url):
  if is_remote_url(url):
    head = requests.head(url, headers={'Accept-Encoding': None})
    return int(head.headers['Content-Length'])
  else:
    return os.path.getsize(url)

import tqdm

async def main(urls='https://battle.shawwn.com/sdc/imagenet/validation_urls.txt', parallel=220, total=50, total_size=None):
  total_size = fetch_url_content_length(urls)
  # lbar_format = r'{l_bar}{bar}{r_bar}'.format(
  #     #l_bar='{desc}: {percentage:3.0f}%|',
  #     #r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]',
  #     l_bar='{desc}',
  #     r_bar='| {percentage:3.3f}% [{elapsed}<{remaining}]',
  #     bar='{bar}'
  #     )
  lbar_format = r'{l_bar}{bar}{r_bar}'.format(
      #l_bar='{desc}: {percentage:3.0f}%|',
      l_bar='{desc}| ',
      #r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]',
      #l_bar='{desc}',
      r_bar='|                            {percentage:3.3f}% [{elapsed}<{remaining}]',
      bar='{bar}'
      )
  bar_format = r'{l_bar}{bar}{r_bar}'.format(
      #l_bar='{desc}: {percentage:3.0f}%|',
      #r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]',
      # l_bar='{desc}',
      # r_bar='| {percentage:3.3f}% [{elapsed}<{remaining}]',
      #l_bar='{desc}: {percentage:3.0f}%|',
      l_bar='{desc}| ',
      r_bar='| {n_fmt:>10s} {desc} of {total_fmt:<10s} {rate_fmt:>16s}{postfix}',
      bar='{bar}'
      )
  def make_bar(position, bar_format=bar_format, **kws):
    return tqdm.tqdm(position=position, dynamic_ncols=True, bar_format=bar_format, **kws)
  with \
      make_bar(4, desc='fetch', colour='yellow', total=parallel, unit=' fetch') as ibar, \
      make_bar(3, desc=' fail', colour='red',    total=total,    unit='  fail') as ebar, \
      make_bar(2, desc=' done', colour='green',  total=total,    unit='  done') as pbar, \
      make_bar(1, desc='bytes', colour='blue',   total=int(1e9), unit='iB', unit_scale=True, unit_divisor=1024) as bbar, \
      make_bar(0, desc='    %', colour='cyan',   total=total_size, unit=' iB', unit_scale=True, unit_divisor=1024, bar_format=lbar_format) as lbar \
      :
    async with make_client() as client:
      tasks = set()
      async for result in fetch_files(client, urls, parallel=parallel, tasks=tasks):
        lbar.update(len(result.url) + 1)
        ibar.update(len(tasks) - ibar.n)
        t = lbar.n / lbar.total
        pbar.total = int((pbar.n + ebar.n) / t) if t > 0 else 0
        ebar.total = pbar.total
        bbar.total = int(bbar.n / t) if t > 0 else 0
        if result.response and result.response.status_code == 200:
          n = len(result.response.content)
          bbar.update(n)
          pbar.update(1)
          ebar.update(0)
          # if pbar.n >= pbar.total:
          #   pbar.total *= 10
          # if bbar.n >= bbar.total:
          #   bbar.total *= 10
          #with tqdm.tqdm.external_write_mode():
          #pbar.write(repr(('%.2f'%(total_bytes/1024/1024), '%.2f'%(n/1024/1024), result.url)))
        else:
          lbar.write(str(result))
          ebar.update(1)
          pbar.update(0)
          if ebar.n >= ebar.total:
            ebar.total *= 10

if __name__ == '__main__':
  import sys
  asyncio.run(main(*sys.argv[1:2]))
# now = time.time(); wait(pmap(lambda x: gather([asyncio.sleep(x), asyncio.sleep(x), x]), [4,9,1], print)); time.time() - now

# now = time.time(); wait(pmap(lambda x: print('zz', x[-1]) or asyncio.sleep(x[-1]), process(lambda x: gather([asyncio.sleep(x), asyncio.sleep(x), x]), [4,9,1])))

# >>> now = time.time(); wait(pmap(lambda x: print('zz', x[-1]) or asyncio.sleep(x[-1]), process(lambda x: gather([asyncio.sleep(x), asyncio.sleep(x), x]), [4,9,1], parallel=1), parallel=1)) ; time.time() - now
# zz 4
# zz 9
# zz 1
# [None, None, None]
# 28.021930932998657

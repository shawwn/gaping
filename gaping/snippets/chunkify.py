
import tensorflow as tf

from functools import partial

import hmac

import numpy as np

def xor(a, b):
  assert isinstance(a, bytes)
  assert isinstance(b, bytes)
  assert len(a) == len(b)
  a = np.array(list(a), dtype=np.uint8)
  b = np.array(list(b), dtype=np.uint8)
  return bytes(a ^ b)


def crypto_hmac(key, message, hashfn, block_size, output_size):
  #print('crypto_hmac(len(key)={}, len(message)={})'.format(len(key), len(message)))
  if len(key) > block_size:
    key = hashfn(key)
  if len(key) < block_size:
    key += b'\x00' * (block_size - len(key))
  
  # o_key_pad ← key xor [0x5c * blockSize]   // Outer padded key
  # i_key_pad ← key xor [0x36 * blockSize]   // Inner padded key
  o_key_pad = xor(key, b'\x5c' * block_size)
  i_key_pad = xor(key, b'\x36' * block_size)

  # return hash(o_key_pad ∥ hash(i_key_pad ∥ message))
  return hashfn(o_key_pad + hashfn(i_key_pad + message))

def crypto_hash_data_slow(key, data, length=None, buf=None, offset=0):
  with tf.Graph().as_default() as g, tf.Session(graph=g).as_default() as sess:
    if length is None:
      length = len(data)
    return bytes(sess.run(tf.fingerprint([data[offset:offset+length]])[0]))

with tf.Graph().as_default() as g:
  tf_fingerprint_string_ph = tf.placeholder(tf.string, shape=(None,))
  tf_fingerprint_string = tf.fingerprint(tf_fingerprint_string_ph)
  if 'tf_fingerprint_sess' in globals():
    tf_fingerprint_sess.close()
  tf_fingerprint_sess = tf.Session(graph=g)

def crypto_hash_data(key, data, length=None, buf=None, offset=0):
  if length is None:
    length = len(data)
  data = data[offset:offset+length]
  if key is not None:
    raise NotImplementedError()
  return bytes(tf_fingerprint_sess.run(tf_fingerprint_string, {tf_fingerprint_string_ph: [data]})[0])


hash_data = partial(crypto_hash_data, None)
hash_hmac = partial(crypto_hmac, hashfn=hash_data, block_size=64, output_size=8)

import struct

def hash_data_to_uint32(data):
  return struct.unpack('>I', hash_data(data)[0:4])[0]

class Fingerprint:
  def __init__(self, key=b''):
    self.block_size = 64
    self.digest_size = 8
    self.name = 'farmhash64'
    self._key = key
  
  def update(self, input):
    self._key = hash_hmac(self._key, input)

  def digest(self):
    return self._key

  def hexdigest(self):
    return self.digest().hex()

  def copy(self):
    return Fingerprint(key=self._key)


def uint32(x):
  return np.array(x, dtype=np.uint32)

def uint64(x):
  return np.array(x, dtype=np.uint64)

from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunkifier:
  # Chunkification parameters
  mu: int = 0    # Desired mean chunk length
  p: int = 0    # Modulus
  pp: int = 0    # - p^(-1) mod 2^32
  ar: int = 0    # alpha2^32 mod p */
  cm: List[int] = field(default_factory=list)    # Coefficient map modulo p
  htlen: int = 0    # Size of hash table in 2-word entries
  blen: int = 0    # Length of buf[]
  w: int = 0    # Minimum substring length; size of b[]

  # Callback parameters
  chunkdone: type(None) = None  # Callback
  cookie: type(None) = None    # Cookie passed to callback

  # Current state
  k: int = 0    # Number of bytes in chunk so far
  r: int = 0    # floor(sqrt(4k - mu)) */
  rs: int = 0    # (r + 1)^2 - (4k - mu) */
  akr: int = 0    # a^k2^32 mod p */
  yka: int = 0    # Power series truncated before x^k term
        # evaluated at a mod p
  b: List[int] = field(default_factory=list)    # Circular buffer of values waiting to
        # be added to the hash table.
  ht: List[int] = field(default_factory=list)    # Hash table; pairs of the form (yka, k).
  buf: List[int] = field(default_factory=list)    # Buffer of bytes processed
  


def isqrt(x):
  """Return the greatest y such that y^2 <= x."""
  x = uint32(x)
  for y in range(1, 65536):
    if uint32(y * y) > x:
      break
  return uint32(y - 1)
  

def isprime(n):
  """Return nonzero iff n is prime."""
  n = uint32(n)
  x = uint32(2)
  while (uint32(x * x) <= n) and (x < 65536):
    if ((n % x) == 0):
      return 0
    x += uint32(1)
  return (n > 1)

def nextprime(n):
  """Return the smallest prime satisfying n <= p < 2^32, or 0 if none exist."""
  p = uint32(n)
  while p != 0:
    if isprime(p):
      break
    p += uint32(1)
  return p


def mmul(a, b, p, pp):
  """Compute $(a * b + (a * b * pp \bmod 2^{32}) * p) / 2^{32}$.

  Note that for $b \leq p$ this is at most $p * (1 + a / 2^{32})$."""
  ab = uint64(uint64(a) * uint64(b));
  abpp = uint32(uint32(ab) * uint32(pp))
  ab += uint64(uint64(abpp) * uint64(p));
  # ab = uint64(int(ab) + uint64(abpp) * uint64(p));
  out = (int(ab) >> 32);
  assert 0 <= out <= 0xffffffff
  return uint32(out)
  # ab //= uint64(2*32)
  # return int(ab)


def minorder(ar, ord, p, pp):
  """Return nonzero if (ar / 2^32) has multiplicative order at least ord mod p."""
  ar = uint32(ar)
  ord = uint32(ord)
  p = uint32(p)
  pp = uint32(pp)

  akr = uint32(0)
  akr0 = uint32(0)
  k = uint32(0)

  akr = akr0 = u32mod(- p, p)

  for k in range(ord):
    akr = u32mod(mmul(akr, ar, p, pp), p);
    if akr == akr0:
      return (0);

  return (1);

# 
# Maximum chunk size.  This is chosen so that after deflating (which might
# add up to 0.1% + 13 bytes to the size) and adding cryptographic wrapping
# (which will add 296 bytes) the final maximum file size is <= 2^18.
# 
MAXCHUNK = 261120


# Mean chunk size desired.
MEANCHUNK  = 65536

# 
# Minimum size of chunk which will be stored as a chunk rather than as a
# file trailer.  As this value increases up to MEANCHUNK/4, the time spent
# chunkifying the trailer stream will increase, the total amount of data
# stored will remain roughly constant, and the number of chunks stored (and
# thus the per-chunk overhead costs) will decrease.
# 
MINCHUNK = 4096

  

import warnings

def chunkify_start(c):
  """Prepare the CHUNKIFIER for input."""
  i = uint32(0)

  # No entries in the hash table.
  for i in range(2 * c.htlen):
    c.ht[i] = int(uint32(- c.htlen));

  # Nothing in the queue waiting to be added to the table, either.
  for i in range(c.w):
    c.b[i] = c.p;

  # No bytes input yet.
  c.akr = u32mod(-c.p, c.p);
  c.yka = 0;
  c.k = 0;
  c.r = 0;
  c.rs = u32add(1, c.mu)


def chunkify_init(meanlen, maxlen, chunkdone, cookie):
  if (meanlen > 1262226) or (maxlen <= meanlen):
    warnings.warn("Incorrect API usage")
    return None
  #
  # Allocate memory for CHUNKIFIER structure.
  #
  #c = malloc(sizeof(*c));
  c = Chunkifier()
  if c == None:
    return None

  # 
  # Initialize fixed chunkifier parameters.
  # 
  c.mu = meanlen;
  c.blen = maxlen;
  c.w = 32;
  c.chunkdone = chunkdone;
  c.cookie = cookie;

  # 
  # Compute the necessary hash table size.  At any given time, there
  # are sqrt(4 k - mu) entries and up to sqrt(4 k - mu) tombstones in
  # the hash table, and we want table inserts and lookups to be fast,
  # so we want these to use up no more than 50% of the table.  We also
  # want the table size to be a power of 2.
  # 
  # Consequently, the table size should be the least power of 2 in
  # excess of 4 * sqrt(4 maxlen - mu) = 8 * sqrt(maxlen - mu / 4).
  # 
  c.htlen = 8;
  i = c.blen - c.mu // 4
  while i > 0:
    c.htlen <<= 1
    i >>= 2

  # 
  # Allocate memory for buffers.
  # 
  c.cm = c.b = c.ht = None;
  c.buf = None;

  # c->cm = malloc(256 * sizeof(c->cm[0]));
  # c->b = malloc(c->w * sizeof(c->b[0]));
  # c->ht = malloc(c->htlen * 2 * sizeof(c->ht[0]));
  # c->buf = malloc(c->blen * sizeof(c->buf[0]));
  c.cm = np.zeros([256], dtype=np.uint32)
  c.b = np.zeros([c.w], dtype=np.uint32)
  c.ht = np.zeros([c.htlen * 2], dtype=np.uint32)
  c.buf = np.zeros([c.blen], dtype=np.uint32)

  # Generate parameter values by computing HMACs.

  # /* p is generated from HMAC('p\0'). */
  # pbuf[0] = 'p';
  # pbuf[1] = 0;
  # if (crypto_hash_data(CRYPTO_KEY_HMAC_CPARAMS, pbuf, 2, hbuf))
  #   goto err;
  # memcpy(&c->p, hbuf, sizeof(c->p));
  c.p = hash_data_to_uint32(b'p\0')

  # /* alpha is generated from HMAC('a\0'). */
  # pbuf[0] = 'a';
  # pbuf[1] = 0;
  # if (crypto_hash_data(CRYPTO_KEY_HMAC_CPARAMS, pbuf, 2, hbuf))
  #   goto err;
  # memcpy(&c->ar, hbuf, sizeof(c->ar));
  c.ar = hash_data_to_uint32(b'a\0')

  # /* cm[i] is generated from HMAC('x' . i). */
  # for (i = 0; i < 256; i++) {
  #   pbuf[0] = 'x';
  #   pbuf[1] = i & 0xff;
  #   if (crypto_hash_data(CRYPTO_KEY_HMAC_CPARAMS, pbuf, 2, hbuf))
  #     goto err;
  #   memcpy(&c->cm[i], hbuf, sizeof(c->cm[i]));
  # }
  for i in range(256):
    c.cm[i] = hash_data_to_uint32(b'x' + bytes(i & 0xff))


  # /*
  #  * Using the generated pseudorandom values, actually generate
  #  * the parameters we want.
  #  */

  # /*
  #  * We want p to be approximately mu^(3/2) * 1.009677744.  Compute p
  #  * to be at least floor(mu*floor(sqrt(mu))*1.01) and no more than
  #  * floor(sqrt(mu)) - 1 more than that.
  #  */
  # pmin = c->mu * isqrt(c->mu);
  # pmin += pmin / 100;
  pmin = c.mu * isqrt(c.mu)
  pmin += pmin // 100;
  c.p = nextprime(u32add(pmin, u32mod(c.p, isqrt(c.mu))))
  # c->p = nextprime(pmin + (c->p % isqrt(c->mu)));
  assert c.p <= 1431655739 < int(2**32 / 3)


  # /* Compute pp = - p^(-1) mod 2^33. */
  # c.pp = 0xffffffff & (((2 * c.p + 4) & 8) - c.p)  # /* pp = - 1/p mod 2^4 */
  # c.pp = 0xffffffff & (c.pp * (2 + c.p * c.pp))   # /* pp = - 1/p mod 2^8 */
  # c.pp = 0xffffffff & (c.pp * (2 + c.p * c.pp))   # /* pp = - 1/p mod 2^16 */
  # c.pp = 0xffffffff & (c.pp * (2 + c.p * c.pp))   # /* pp = - 1/p mod 2^32 */
  c.pp = uint32(uint64(0xffffffff) & (uint64(uint64(2 * c.p + 4) & uint64(8)) - uint64(c.p)))  # /* pp = - 1/p mod 2^4 */
  c.pp = uint32(uint64(0xffffffff) & uint64(c.pp * (2 + c.p * c.pp)))   # /* pp = - 1/p mod 2^8 */
  c.pp = uint32(uint64(0xffffffff) & uint64(c.pp * (2 + c.p * c.pp)))   # /* pp = - 1/p mod 2^16 */
  c.pp = uint32(uint64(0xffffffff) & uint64(c.pp * (2 + c.p * c.pp)))   # /* pp = - 1/p mod 2^32 */

  # /*
  #  * We want to have 1 < ar < p - 1 and the multiplicative order of
  #  * alpha mod p greater than mu.
  #  */
  c.ar = u32add(2, u32mod(c.ar, u32sub(c.p, 3)))
  while not minorder(c.ar, c.mu, c.p, c.pp):
    c.ar = u32add(c.ar, 1)
    if c.ar == c.p:
      c.ar = uint32(2)
  
  # /*
  #  * Prepare for incoming data.
  #  */
  chunkify_start(c);

  return c

class GotoEndOfChunk(Exception):
  pass


def u32add(a, b):
  #return 0xffffffff & (int(a) + int(b))
  return uint32(uint32(a) + uint32(b))

def u32sub(a, b):
  #return 0xffffffff & (int(a) - int(b))
  return uint32(uint32(a) - uint32(b))

def u32mul(a, b):
  #return 0xffffffff & (int(a) * int(b))
  return uint32(a) * uint32(b)

def u32mod(a, b):
  #return 0xffffffff & (int(a) * int(b))
  return uint32(uint32(a) % uint32(b))

import tqdm

def chunkify_write(c, buf):
  buflen = len(buf)
  #with tqdm.trange(buflen, initial=int(c.k), total=c.blen) as pbar:
  with tqdm.trange(buflen) as pbar:
    for i in pbar:
      try:
        # /* Add byte to buffer. */
        c.buf[c.k] = buf[i]

        # /* k := k + 1 */
        c.k = u32add(c.k, 1)
        while c.rs <= 4:
          c.rs = u32add(c.rs, u32add(u32mul(2, c.r), 1))
          c.r = u32add(c.r, 1)
        c.rs = u32sub(c.rs, 4)

        # /*
        #  * If k = blen, then we've filled the buffer and we
        #  * automatically have the end of the chunk.
        #  */
        if c.k == c.blen:
          raise GotoEndOfChunk()


        # /*
        #  * Don't waste time on arithmetic if we don't have enough
        #  * data yet for a permitted loop to ever occur.
        #  */
        if (c.r == 0):
          pbar.write('continuing at {0} c.k={1.k} c.r={1.r} c.rs={1.rs}'.format(i, c))
          continue;

        # /*
        #  * Update state to add new character.
        #  */

        # /* y_k(a) := y_k(a) + a^k * x_k mod p */
        # /* yka <= p * (2 + p / (2^32 - p)) <= p * 2.5 < 2^31 + p */
        # c->yka += mmul(c->akr, c->cm[buf[i]], c->p, c->pp);
        c.yka = u32add(c.yka, mmul(c.akr, c.cm[buf[i]], c.p, c.pp));

        # /* Each step reduces yka by p iff yka >= p. */
        # c->yka -= c->p & (((c->yka - c->p) >> 31) - 1);
        # c->yka -= c->p & (((c->yka - c->p) >> 31) - 1);
        # c.yka = u32sub(c.yka, c.p & (((c.yka - c.p) >> 31) - 1));
        # c.yka = u32sub(c.yka, c.p & (((c.yka - c.p) >> 31) - 1));
        c.yka = u32sub(c.yka, c.p & u32sub((u32sub(c.yka, c.p) >> 31), 1));
        c.yka = u32sub(c.yka, c.p & u32sub((u32sub(c.yka, c.p) >> 31), 1));

        # /* a^k := a^k * alpha mod p */
        # /* akr <= p * 2^32 / (2^32 - p) */
        # c->akr = mmul(c->akr, c->ar, c->p, c->pp);
        c.akr = mmul(c.akr, c.ar, c.p, c.pp);

        # /*
        #  * Check if yka is in the hash table.
        #  */
        htpos = c.yka & (c.htlen - 1);
        while True:
          # /* Have we found yka? */
          if (c.ht[2 * htpos + 1] == c.yka):
            # /* Recent enough to be a valid entry? */
            if (c.k - c.ht[2 * htpos] - 1 < c.r):
              raise GotoEndOfChunk();

          # /* Have we found an empty space? */
          if (c.k - c.ht[2 * htpos] - 1 >= 2 * c.r):
            break;

          # /* Move to the next position in the table. */
          htpos = (htpos + 1) & (c.htlen - 1);

        # /*
        #  * Insert queued value into table.
        #  */
        yka_tmp = c.b[c.k & (c.w - 1)];
        htpos = yka_tmp & (c.htlen - 1);
        while True:
          # /* Have we found an empty space or tombstone? */
          if (c.k - c.ht[2 * htpos] - 1 >= c.r):
            c.ht[2 * htpos] = c.k;
            c.ht[2 * htpos + 1] = yka_tmp;
            break;

          # /* Move to the next position in the table. */
          htpos = (htpos + 1) & (c.htlen - 1);

        # /*
        #  * Add current value into queue.
        #  */
        c.b[c.k & (c.w - 1)] = c.yka;


        # /*
        #  * Move on to next byte.
        #  */
        continue;

      except GotoEndOfChunk:
        pass
      # /*
      #  * We've reached the end of a chunk.
      #  */
      rc = chunkify_end(c);
      if rc:
        return rc
      #pbar.reset(total=c.blen)
  

def chunkify_end(c):
  # /* If we haven't started the chunk yet, don't end it either. */
  if (c.k == 0):
    return (0);

  # /* Process the chunk. */
  rc = (c.chunkdone)(c.cookie, c.buf[0:c.k], c.k);
  if (rc):
    return (rc);

  # /* Prepare for more input. */
  chunkify_start(c);

  return (0);


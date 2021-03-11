# see also: https://github.com/modanesh/MS-Celeb-Aligned-Extractor/blob/master/tsv_extractor.py

import os

import tensorflow as tf
from functools import partial

from .. import tf_api as api
from ..util import EasyDict

def get_bucket(bucket=None):
  if bucket is None:
    bucket = os.environ.get('TPU_DATA_BUCKET', 'gs://mldata-euw4')
  return 'gs://' + bucket.replace('gs://', '').strip('/') + '/'

class MSCeleb1M:
    def __init__(self, batch_size=4096*16, repeat=True, bucket=None, aligned=True):
      self.bucket = get_bucket(bucket)
      if aligned:
        self.path = 'data/MS-Celeb-1M/data/aligned_face_images/FaceImageCroppedWithAlignment.tsv'
      else:
        self.path = 'data/MS-Celeb-1M/data/croped_face_images/FaceImageCroppedWithOutAlignment.tsv'
      self.lines, self.dataset, self.features, self.batched = self.build_dataset(batch_size=batch_size, repeat=repeat)

    @property
    def full_path(self):
      return self.bucket + self.path

    def map(self, dataset, fn, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False):
      #return dataset.map(fn, num_parallel_calls=num_parallel_calls, deterministic=deterministic)
      # in tf 1.15: map() got an unexpected keyword argument 'deterministic'
      assert deterministic == False
      return dataset.map(fn, num_parallel_calls=num_parallel_calls)

    def batch(self, dataset, batch_size):
      dataset = dataset.batch(batch_size)
      if isinstance(batch_size, int):
        def set_size(*args):
          return api.map_structure(partial(api.set_batch_size, batch_size=batch_size), *args)
      return dataset.map(set_size)

    def build_dataset(self, batch_size, repeat=True):
      lines = tf.data.TextLineDataset(self.full_path)
      lines = lines.prefetch(tf.data.experimental.AUTOTUNE)
      if repeat:
        lines = lines.repeat()
      parsed = self.map(lines, self.parse)
      processed = self.map(parsed, self.process)
      batched = self.batch(processed, batch_size)
      return lines, parsed, processed, batched

    def parse(self, line):
      # >>> pp( line.split()[0:-1] )
      # [b'm.025xnf0',
      #  b'79',
      #  b'http://www.welkgirls.com/south111.jpg',
      #  b'http://www.welkgirls.com/photoalbum3.html',
      #  b'FaceId-0',
      #  b'xAt1PpwBIT5uZao+ENWPPg==']
      cells = api.tf_string_split([line], '\t')
      cells.set_shape([7])
      cells = tf.unstack(cells)
      # 0: Freebase MID (unique key for each entity), e.g. m.025xnf0
      # 1: ImageSearchRank, e.g. 79
      # 2: ImageURL: the downloadable image URL, e.g. http://beatlephotoblog.com/photos/2011/04/188.jpg
      # 3: PageURL: the page containing the image, e.g. http://beatlephotoblog.com/tag/magic-alex
      # 4: FaceID, e.g. FaceID-0
      # 5: bbox, e.g. [0.427256  , 0.38245612, 0.5616943 , 0.46783626]
      # 6: image_data: base64-encoded image data (web unsafe; requires special processing for tf.io.decode_base64)
      out = EasyDict()
      out.freebase_mid = cells[0]
      out.image_search_rank = tf.cast(tf.strings.to_number(cells[1]), tf.int64)
      out.image_url = cells[2]
      out.page_url = cells[3]
      out.face_id = self.parse_face_id(cells[4])
      out.bbox = tf.io.decode_raw(api.tf_decode_base64(cells[5]), tf.float32)
      out.bbox.set_shape([4])
      out.image_data = api.tf_decode_base64(cells[6])
      out.image_hash = api.tf_farmhash64(cells[6])
      return out

    def parse_face_id(self, face_id, dtype=tf.int32):
      # face_id looks like 'FaceID-16'
      # we want to convert the '16' to an integer
      face_id = api.check(
          api.tf_re_match(face_id, 'FaceId-[0-9]+'),
          "FaceID wasn't formatted correctly",
          face_id)
      ending = api.tf_string_split(face_id, '-')[-1]
      return api.tf_parse_i64(ending)

    def process(self, parsed):
      parsed = EasyDict(parsed)
      features = EasyDict()
      features.image_data = parsed.image_data
      # features.image = tf.io.decode_image(parsed.image_data, channels=3)
      # features.image.set_shape([None, None, 3])
      #features.label = api.tf_i32(parsed.face_id)
      features.label = parsed.freebase_mid + tf.strings.format('-{}', parsed.face_id)
      return features


      


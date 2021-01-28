import tensorflow.compat.v1 as tf
import re

def fetch_dataset(filename,
    buffer_size = 8 * 1024 * 1024): # 8 MiB per file
  dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
  return dataset

def fetch_filenames(file_patterns, current_host, num_hosts, seed=None):
    file_patterns = [x.strip() for x in file_patterns.split(',') if len(x.strip()) > 0]

    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = None
    for pattern in file_patterns:
      count, pattern = re.findall("([0-9]+[*])?(gs://.*)", pattern)[0]
      count = count.rstrip('*')
      count = 1 if len(count) <= 0 else int(count)
      x = tf.data.Dataset.list_files(pattern, shuffle=False, seed=seed)
      if count != 1:
        x = x.repeat(count)
      x = x.shard(num_hosts, current_host)
      dataset = x if dataset is None else dataset.concatenate(x)

    # Memoize the filename list to avoid lots of calls to list_files.
    dataset = dataset.cache()

    return dataset


def filenames_to_records(dataset, num_parallel_calls=1): # usually use num_parallel_calls=64 or something large
  # dataset = dataset.apply(
  #     tf.contrib.data.parallel_interleave(
  #         fetch_dataset, cycle_length=cycle_length, sloppy=True))
  dataset = dataset.interleave(fetch_dataset, cycle_length=num_parallel_calls)
  return dataset


def parse_image(value, image_key="image/encoded", label_key="image/class/label", label_bias=0):
  keys_to_features = {
      image_key: tf.FixedLenFeature((), tf.string, ''),
      label_key: tf.FixedLenFeature([], tf.int64, -1),
  }
  parsed = tf.parse_single_example(value, keys_to_features)
  image_bytes = tf.reshape(parsed[image_key], shape=[])
  image = tf.io.decode_image(image_bytes, 3)

  # For imagenet records, set label_bias = -1 so that labels are in [0, 1000).
  label = tf.cast(tf.reshape(parsed[label_key], shape=[]), dtype=tf.int32) + label_bias

  # compute a hash of the image
  fingerprint = tf.raw_ops.Fingerprint(data=[image_bytes], method="farmhash64")
  fingerprint = tf.bitcast(fingerprint, tf.int64)
  fingerprint = fingerprint[0]

  return {
    'image': image,
    'label': label,
    'hash': fingerprint,
  }
  
# >>> reload(datasets); ds = datasets.fetch_filenames( "gs://mldata-euw4/datasets/imagenet/validation-*", 0, 1 ); ds = datasets.filenames_to_records( ds ); ds = ds.map(datasets.parse_image, num_parallel_calls=1)

# reload(datasets); ds = datasets.fetch_filenames( "gs://mldata-euw4/datasets/imagenet/validation-*", 0, 1 ); ds = tf.data.Dataset.zip((ds, tf.data.Dataset.range(200))).repeat()

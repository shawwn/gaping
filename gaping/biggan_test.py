import tensorflow as tf
import tensorflow.compat.v1 as tf1

from absl import flags
from absl.testing import parameterized

from gaping import test_utils

class BigGanTest(parameterized.TestCase, test_utils.GapingTestCase):

  def setUp(self):
    super(BigGanTest, self).setUp()
    #self.model_dir = self._get_empty_model_dir()
    self.model_dir = 'gs://dota-euw4a/tmp/test/biggan1'
    self.run_config = tf1.estimator.tpu.RunConfig(
        model_dir=self.model_dir,
        tpu_config=tf1.estimator.tpu.TPUConfig(iterations_per_loop=1))

  @parameterized.parameters([42,99])
  def testSingleTrainingStepPenalties(self, value):
    print(value)
    

if __name__ == "__main__":
  tf.test.main()

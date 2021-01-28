from tensorflow.python.training import optimizer

class WrappedOptimizer(optimizer.Optimizer):
  def __init__(self,
               opt,
               name="WrappedOptimizer",
               use_locking=False):
    if not isinstance(opt, optimizer.Optimizer):
      raise TypeError(
          "WrappedOptimizer only works with tf.training.Optimizer and not "
          "Optimizer_v2. If you are using TPUStrategy, OptimizerV2 will sum "
          "gradients across replicas."
          "If you are using TPUEstimator, you may instead sum your gradients "
          "with: grads = [tf.compat.v1.tpu.cross_replica_sum(g) for g in grads]"
          ". If you want to average your gradients, rescale your loss with: "
          "loss /= global_batch_size")
    super().__init__(use_locking=use_locking, name=name)
    self._opt = opt

  def compute_gradients(self, loss, var_list=None, **kwargs):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps `compute_gradients()` from the real optimizer."""
    return self._opt.compute_gradients(loss, var_list=var_list, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This simply wraps `apply_gradients()` from the real optimizer."""
    # summed_grads_and_vars = []
    # for (grad, var) in grads_and_vars:
    #   if grad is None:
    #     summed_grads_and_vars.append((grad, var))
    #   else:
    #     with ops.colocate_with(grad):
    #       summed_grads_and_vars.append((tpu_ops.cross_replica_sum(
    #           grad, self._group_assignment), var))
    # return self._opt.apply_gradients(summed_grads_and_vars, global_step, name)
    return self._opt.apply_gradients(grads_and_vars, global_step=global_step, name=name)

  def get_slot(self, *args, **kwargs):
    """Return a slot named "name" created for "var" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    """Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    """
    return self._opt.get_slot_names(*args, **kwargs)

  def variables(self):
    """Forwarding the variables from the underlying optimizer."""
    return self._opt.variables()
  

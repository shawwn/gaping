from . import sequenced
from . import data_provider

class EventSequencer(
    data_provider.DataProvider,
    sequenced.Sequenced):
  pass

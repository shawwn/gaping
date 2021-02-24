from typing_extensions import Protocol

class Cursored(Protocol):

  def get_cursor(self):
    ...


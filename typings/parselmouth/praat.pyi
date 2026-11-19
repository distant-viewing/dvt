from . import Data

# What a command gives back depends on the command: a number, a string, a
# new object, or nothing.
def call(object: Data | list[Data], command: str, *args: float | str | bool) -> object: ...

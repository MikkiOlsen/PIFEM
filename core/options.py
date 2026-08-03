import json
import os
from types import SimpleNamespace

_OPTIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'config', 'options.json',
)
with open(_OPTIONS_PATH, 'r') as _f:
    OPTIONS = SimpleNamespace(**json.load(_f))

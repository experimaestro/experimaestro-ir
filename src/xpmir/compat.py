import sys
import typing

if sys.version_info.major == 3 and sys.version_info.minor < 10:
    from typing_extensions import ParamSpec, Protocol
else:
    from typing import ParamSpec, Protocol

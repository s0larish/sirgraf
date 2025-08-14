"""
sirgraf package

Auto-discovers and re-exports public symbols from all modules in this package.
- Any module-level `__all__` controls what gets exported.
- Otherwise, all names not starting with '_' are exported.
- Skips private modules (leading '_'), tests, and the 'experiment' dir.

Usage:
    from sirgraf import create_static_background
"""

from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path
from typing import Set, List

__all__: List[str] = []

_pkg_dir = Path(__file__).parent
_exported: Set[str] = set()

# Names (files/dirs) to skip entirely
_SKIP = {"__init__", "__pycache__", "experiment", "tests", "test"}

for m in iter_modules([str(_pkg_dir)]):
    name = m.name
    # Skip private/utility modules and known non-package dirs
    if name.startswith("_") or name in _SKIP:
        continue

    full_name = f"{__name__}.{name}"
    mod = import_module(full_name)

    # Re-export the submodule itself (optional but handy)
    globals()[name] = mod

    # Determine public API of the module
    public = getattr(mod, "__all__", None)
    if public is None:
        public = [n for n in vars(mod).keys() if not n.startswith("_")]

    # Bind into top-level namespace
    for n in public:
        try:
            globals()[n] = getattr(mod, n)
            _exported.add(n)
        except AttributeError:
            # If the module listed a name in __all__ that it doesn't have
            # we just ignore it gracefully.
            pass

# Finalize package-level __all__
__all__ = sorted(_exported)
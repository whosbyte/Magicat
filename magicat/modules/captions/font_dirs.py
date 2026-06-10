# magicat/modules/captions/font_dirs.py
"""Font search locations: bundled OFL fonts + the OS font dir + user dirs.

Bundled fonts are free proxies for the short-form universe (the real
TikTok 'Classic' face is the commercial Proxima Nova - Montserrat is the
documented closest free substitute). Users add their own via the
MAGICAT_FONT_DIRS env var (os.pathsep-separated), which wins on collision.
"""
from __future__ import annotations

import sys
from pathlib import Path

ASSETS_FONTS = Path(__file__).resolve().parents[3] / "assets" / "fonts"


def bundle_dirs() -> list[str]:
    if not ASSETS_FONTS.is_dir():
        return []
    # the root is always included (it exists in-repo via its README even
    # when every font download failed), then any family subdirectories
    return [str(ASSETS_FONTS)] + [
        str(d) for d in sorted(ASSETS_FONTS.iterdir()) if d.is_dir()]


def system_font_dirs() -> list[str]:
    if sys.platform == "win32":
        return ["C:/Windows/Fonts"]
    return ["/usr/share/fonts", "/System/Library/Fonts"]


def default_font_dirs() -> list[str]:
    # bundle first, system second: a system font with the same stem wins,
    # and MAGICAT_FONT_DIRS (applied by FontMatcher.from_dirs) wins over both
    return bundle_dirs() + system_font_dirs()

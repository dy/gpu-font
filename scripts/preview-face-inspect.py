"""Read verified face metadata straight from a font binary.

The collector renders the exact bytes inspected here, so weight, slant, style
name, axes and glyph coverage are read from the file that produced the pixels
instead of being copied from a catalogue page.
"""
import hashlib
import json
import sys
from fontTools.ttLib import TTFont

NAME_IDS = {
    0: "copyright", 1: "family", 2: "subfamily", 3: "uniqueId", 4: "fullName",
    5: "version", 6: "postscriptName", 8: "manufacturer", 9: "designer",
    13: "license", 14: "licenseUrl", 16: "typographicFamily", 17: "typographicSubfamily",
}
LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def names(font):
    out = {}
    for record in font["name"].names:
        key = NAME_IDS.get(record.nameID)
        if key is None or key in out:
            continue
        try:
            out[key] = str(record)
        except UnicodeDecodeError:
            continue
    return out


def coverage(font, text):
    cmap = font.getBestCmap()
    missing = sorted({ch for ch in text if ord(ch) not in cmap})
    return {"covered": not missing, "missing": missing}


def case_folded(font):
    """True when lowercase letters map to the same glyphs as their capitals."""
    cmap = font.getBestCmap()
    pairs = [(cmap.get(ord(c)), cmap.get(ord(c.upper()))) for c in LATIN[:26]]
    pairs = [(low, up) for low, up in pairs if low and up]
    return bool(pairs) and all(low == up for low, up in pairs)


def monospaced(font):
    cmap, widths = font.getBestCmap(), font["hmtx"].metrics
    advances = {widths[cmap[ord(c)]][0] for c in LATIN if ord(c) in cmap and cmap[ord(c)] in widths}
    return len(advances) == 1


def inspect(path, texts):
    raw = open(path, "rb").read()
    font = TTFont(path, fontNumber=0, lazy=False)
    os2, post, head = font.get("OS/2"), font.get("post"), font["head"]
    fvar = font.get("fvar")
    report = {
        "path": path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "flavor": font.flavor or "sfnt",
        "names": names(font),
        "unitsPerEm": head.unitsPerEm,
        "os2WeightClass": getattr(os2, "usWeightClass", None),
        "os2WidthClass": getattr(os2, "usWidthClass", None),
        "italicBit": bool(getattr(os2, "fsSelection", 0) & 0x01) if os2 else None,
        "obliqueBit": bool(getattr(os2, "fsSelection", 0) & 0x200) if os2 else None,
        "macStyleItalic": bool(head.macStyle & 0x02),
        "italicAngle": getattr(post, "italicAngle", None),
        "isFixedPitch": bool(getattr(post, "isFixedPitch", 0)),
        "monospacedLatin": monospaced(font),
        "variable": fvar is not None,
        "axes": {a.axisTag: {"min": a.minValue, "default": a.defaultValue, "max": a.maxValue}
                 for a in (fvar.axes if fvar else [])},
        "namedInstances": [
            {"name": str(font["name"].getDebugName(i.subfamilyNameID)), "coordinates": i.coordinates}
            for i in (fvar.instances if fvar else [])
        ],
        "uppercaseFolded": case_folded(font),
        "texts": {text: coverage(font, text) for text in texts},
    }
    font.close()
    return report


if __name__ == "__main__":
    path, texts = sys.argv[1], sys.argv[2:]
    json.dump(inspect(path, texts), sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")

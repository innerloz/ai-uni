"""Static integrity checks; no browser or external dependencies needed."""
from html.parser import HTMLParser
from pathlib import Path
import ast
import re

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.references = []
        self.local_files = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        for key in ("src", "href"):
            value = a.get(key, "")
            if value.startswith("#") and len(value) > 1:
                self.references.append(value[1:])
            elif value and not value.startswith(("https:", "http:", "data:", "#")):
                self.local_files.append(value)


page = PageParser()
page.feed((DIST / "index.html").read_text())
assert len(page.ids) == len(set(page.ids)), "Duplicate IDs in index.html"
for ref in page.references:
    assert ref in page.ids, f"Missing anchor: {ref}"
for asset in page.local_files:
    assert (DIST / asset).is_file(), f"Missing local asset: {asset}"

for js in [*DIST.glob("*.js"), *DIST.glob("*.mjs")]:
    text = js.read_text()
    for specifier in re.findall(r"(?:from\s+|import\s*)['\"](\.[^'\"]+)['\"]", text):
        assert (js.parent / specifier).is_file(), f"Missing JS import: {specifier}"

for script in (DIST / "examples").glob("*.py"):
    ast.parse(script.read_text(), filename=str(script))

for image in DIST.glob("slide-*.png"):
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), f"Invalid PNG: {image}"

print(f"Static checks passed: {len(page.ids)} unique IDs, all anchors, assets, module imports, Python syntax, and slide PNGs.")

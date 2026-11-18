#!/usr/bin/env python3
"""Check every link in this repository's Markdown files.

Internal links: the target file must exist, and a #fragment must match a heading of the target
(GitHub's heading-to-anchor rule) or an explicit <a id="..."> / <a name="...">.
External links (http, https): listed and counted; with --external each is requested once and its
status reported. A request that cannot be made is reported as INCONCLUSIVE, never as a pass.

Usage:  python3 tools/check_links.py [--external] [ROOT]
Exit:   0 = every internal link resolves (and, with --external, every external link answered 2xx/3xx)
        1 = at least one broken link
        2 = at least one external link INCONCLUSIVE and none broken
"""
import os
import re
import sys
import urllib.request

LINK = re.compile(r"(?<!\!)\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")
HTML_REF = re.compile(r"""(?:href|src)=["']([^"']+)["']""")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
ANCHOR = re.compile(r"""<a\s+(?:id|name)=["']([^"']+)["']""")
FENCE = re.compile(r"^\s*(```|~~~)")
SPACED = re.compile(r"\]\(((?!https?://)[^)\s\"]+(?:\s+[^)\s\"]+)+)\)")


def slug(text):
    """GitHub's rule: strip markup, lowercase, drop everything but word characters, spaces and hyphens."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("`", "").replace("*", "")
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def read_lines(path):
    """Lines of a Markdown file with fenced code blocks blanked (links inside code are not links)."""
    out, fenced = [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if FENCE.match(line):
                fenced = not fenced
                out.append("")
                continue
            out.append("" if fenced else line.rstrip("\n"))
    return out


def anchors_of(path):
    seen, found = {}, set()
    for line in read_lines(path):
        m = HEADING.match(line)
        if m:
            s = slug(m.group(2))
            n = seen.get(s, 0)
            seen[s] = n + 1
            found.add(s if n == 0 else f"{s}-{n}")
        for a in ANCHOR.findall(line):
            found.add(a)
    return found


def strip_code_spans(line):
    return re.sub(r"`[^`]*`", "", line)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    external = "--external" in sys.argv
    root = os.path.abspath(args[0] if args else os.path.join(os.path.dirname(__file__), ".."))
    md_files = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in (".git", "_site")]
        md_files += [os.path.join(d, f) for f in files if f.endswith(".md")]
    md_files.sort()

    broken, internal, ext = [], 0, {}
    anchor_cache = {}
    for path in md_files:
        rel = os.path.relpath(path, root)
        for num, raw in enumerate(read_lines(path), 1):
            line = strip_code_spans(raw)
            targets = LINK.findall(line) + IMAGE.findall(line) + AUTOLINK.findall(line) + HTML_REF.findall(line)
            for bad in SPACED.findall(line):
                internal += 1
                broken.append(f"{rel}:{num}: link target contains spaces: {bad}")
            for t in targets:
                if t.startswith(("http://", "https://")):
                    ext.setdefault(t, f"{rel}:{num}")
                    continue
                if t.startswith(("mailto:", "tel:")):
                    continue
                internal += 1
                target, _, frag = t.partition("#")
                dest = path if target == "" else os.path.normpath(os.path.join(os.path.dirname(path), target))
                if not dest.startswith(root):
                    broken.append(f"{rel}:{num}: leaves the repository: {t}")
                    continue
                if os.path.isdir(dest):
                    if not any(os.path.exists(os.path.join(dest, n)) for n in ("README.md", "index.md")):
                        broken.append(f"{rel}:{num}: folder without README.md or index.md: {t}")
                    continue
                if not os.path.exists(dest):
                    broken.append(f"{rel}:{num}: missing file: {t}")
                    continue
                if frag and dest.endswith(".md"):
                    if dest not in anchor_cache:
                        anchor_cache[dest] = anchors_of(dest)
                    if frag not in anchor_cache[dest]:
                        broken.append(f"{rel}:{num}: no such heading #{frag} in {os.path.relpath(dest, root)}")

    inconclusive = []
    if external:
        for url, where in sorted(ext.items()):
            req = urllib.request.Request(url.split("#")[0], method="GET",
                                         headers={"User-Agent": "Mozilla/5.0 (link checker)"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    code = r.status
            except urllib.error.HTTPError as e:
                code = e.code
            except Exception as e:  # network failure: cannot decide
                inconclusive.append(f"{where}: INCONCLUSIVE ({type(e).__name__}): {url}")
                continue
            if code in (401, 403, 429):
                inconclusive.append(f"{where}: INCONCLUSIVE (HTTP {code}, the site refuses automated requests): {url}")
            elif code >= 400:
                broken.append(f"{where}: HTTP {code}: {url}")

    for b in broken:
        print("BROKEN  " + b)
    for i in inconclusive:
        print(i)
    checked_ext = f"{len(ext)} external links " + ("requested" if external else "listed, not requested")
    print(f"check_links: {len(md_files)} files, {internal} internal links, {checked_ext}; "
          f"{len(broken)} broken, {len(inconclusive)} inconclusive")
    sys.exit(1 if broken else (2 if inconclusive else 0))


if __name__ == "__main__":
    main()

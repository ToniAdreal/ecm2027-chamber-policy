#!/usr/bin/env python3
"""Render submission/ECM2027_Policy_Submission_Team_ECM-122.md to PDF via weasyprint.

Minimal deterministic Markdown subset: ATX headers, **bold**, `code`,
fenced code blocks, bullet lists, links, paragraphs. No third-party deps.
Usage: python3 tools/render_submission_pdf.py
"""
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "submission" / "ECM2027_Policy_Submission_Team_ECM-122.md"
OUT_HTML = ROOT / "submission" / "ECM2027_Policy_Submission_Team_ECM-122.html"
OUT_PDF = ROOT / "submission" / "ECM2027_Policy_Submission_Team_ECM-122.pdf"


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def convert(md: str) -> str:
    out: list[str] = []
    lines = md.split("\n")
    i = 0
    in_list = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            lang = line[3:].strip()
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # closing fence
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>")
            i += 1
            continue
        if line.startswith("* "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + inline(line[2:]) + "</li>")
            i += 1
            continue
        if line.strip() == "":
            if in_list:
                out.append("</ul>")
                in_list = False
            i += 1
            continue
        if line.strip() == "---":
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append("<hr>")
            i += 1
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        # paragraph: gather consecutive non-blank lines
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "* ", "```", "---")):
            para.append(lines[i])
            i += 1
        out.append("<p>" + inline(" ".join(p.strip() for p in para)) + "</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def main() -> int:
    body = convert(MD.read_text())
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 2cm; }}
body {{ font-family: sans-serif; font-size: 11pt; line-height: 1.5; color: #111; }}
h1 {{ font-size: 16pt; }} h2 {{ font-size: 13pt; margin-top: 1.2em; }}
code {{ font-size: 9.5pt; background: #f0f0f0; padding: 1px 3px; }}
pre {{ background: #f0f0f0; padding: 10px; font-size: 9.5pt; white-space: pre-wrap; overflow-wrap: break-word; }}
a {{ color: #0645ad; }}
</style></head><body>{body}</body></html>"""
    OUT_HTML.write_text(doc)
    from weasyprint import HTML

    HTML(string=doc).write_pdf(str(OUT_PDF))
    print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

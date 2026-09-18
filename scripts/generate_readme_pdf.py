"""
Generate a styled, publication-grade PDF from README.md for SolarVision.
Compiles README.md into README.pdf via Microsoft Edge / Chrome Headless engine.
"""

import base64
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
README_MD = ROOT_DIR / "README.md"
OUTPUT_HTML = ROOT_DIR / "docs" / "README.html"
OUTPUT_PDF = ROOT_DIR / "README.pdf"

CHROME_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
BROWSER_EXE = next((p for p in CHROME_PATHS if os.path.exists(p)), None)


def img_to_b64(path: Path) -> str:
    """Convert an image file to base64 data URI."""
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def convert_markdown_to_html(md_text: str) -> str:
    """A clean, robust Markdown to HTML converter with math and table support."""
    # Pre-encode figures to base64
    fig1_path = ROOT_DIR / "data" / "detection_outputs" / "detected_sdo_hmi_ar3664_20240510.jpg"
    fig2_path = ROOT_DIR / "data" / "preprocessing_visuals" / "preprocessing_sdo_hmi_ar3664_20240510.jpg"
    fig1_b64 = img_to_b64(fig1_path)
    fig2_b64 = img_to_b64(fig2_path)

    lines = md_text.splitlines()
    html_lines = []
    in_code_block = False
    in_table = False
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Fenced code blocks
        if stripped.startswith("```"):
            if in_code_block:
                html_lines.append("</pre></div>")
                in_code_block = False
            else:
                lang = stripped[3:].strip()
                html_lines.append(f'<div class="code-container"><pre class="code-block language-{lang}">')
                in_code_block = True
            continue

        if in_code_block:
            # Escape HTML characters in code
            escaped = (
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            html_lines.append(escaped)
            continue

        # Tables
        if "|" in line and not stripped.startswith(">"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if cells:
                # Check if separator row
                if all(re.match(r"^:?-+:?$", c) for c in cells):
                    continue
                if not in_table:
                    html_lines.append('<div class="table-container"><table><thead><tr>')
                    for c in cells:
                        html_lines.append(f"<th>{c}</th>")
                    html_lines.append("</tr></thead><tbody>")
                    in_table = True
                else:
                    html_lines.append("<tr>")
                    for c in cells:
                        html_lines.append(f"<td>{c}</td>")
                    html_lines.append("</tr>")
                continue
        else:
            if in_table:
                html_lines.append("</tbody></table></div>")
                in_table = False

        # Blank line
        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("<br/>")
            continue

        # Horizontal rule
        if stripped in ["---", "***", "___"]:
            html_lines.append("<hr/>")
            continue

        # Alerts / Callouts
        if stripped.startswith("> [!IMPORTANT]") or stripped.startswith("> [!CAUTION]") or stripped.startswith("> [!NOTE]") or stripped.startswith("> [!TIP]"):
            alert_type = "warning" if "CAUTION" in stripped or "IMPORTANT" in stripped else "info"
            title = stripped.split("[!")[1].split("]")[0]
            html_lines.append(f'<div class="callout callout-{alert_type}"><div class="callout-title">{title}</div>')
            continue
        elif stripped.startswith(">"):
            content = stripped[1:].strip()
            if html_lines and "callout" in html_lines[-1]:
                html_lines.append(f"<p>{content}</p></div>")
            else:
                html_lines.append(f'<div class="callout"><p>{content}</p></div>')
            continue

        # Images
        if stripped.startswith("![") and "](" in stripped:
            alt_text = stripped.split("![")[1].split("]")[0]
            img_src = stripped.split("](")[1].split(")")[0]
            actual_b64 = fig1_b64 if "detected_" in img_src else fig2_b64
            html_lines.append(f'<div class="figure-box"><img src="{actual_b64}" class="figure-img" alt="{alt_text}"/></div>')
            continue

        # Headings
        if stripped.startswith("# "):
            html_lines.append(f'<h1 class="h1-title">{stripped[2:]}</h1>')
            continue
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
            continue
        elif stripped.startswith("### "):
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
            continue
        elif stripped.startswith("#### "):
            html_lines.append(f"<h4>{stripped[5:]}</h4>")
            continue

        # Unordered list items
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{stripped[2:]}</li>")
            continue
        else:
            if in_list and not stripped.startswith(" "):
                html_lines.append("</ul>")
                in_list = False

        # Regular paragraph
        html_lines.append(f"<p>{line}</p>")

    if in_table:
        html_lines.append("</tbody></table></div>")
    if in_list:
        html_lines.append("</ul>")
    if in_code_block:
        html_lines.append("</pre></div>")

    raw_html = "\n".join(html_lines)

    # Convert inline bold and italics
    raw_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", raw_html)
    raw_html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", raw_html)
    raw_html = re.sub(r"`([^`]+)`", r"<code>\1</code>", raw_html)

    # Convert inline math formulas
    raw_html = re.sub(r"\$\$(.+?)\$\$", r'<div class="equation">\1</div>', raw_html)
    raw_html = re.sub(r"\$([^$]+)\$", r'<span class="inline-math">\1</span>', raw_html)

    # Convert links
    raw_html = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', raw_html)

    return raw_html


def build_readme_html() -> str:
    with open(README_MD, "r", encoding="utf-8") as f:
        md_text = f.read()

    body_html = convert_markdown_to_html(md_text)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SolarVision - README Documentation</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 18mm 15mm 20mm 15mm;
    @top-left {{
      content: "SolarVision: Autonomous Solar Active Region CV System";
      font-family: 'Helvetica Neue', Arial, sans-serif;
      font-size: 8pt;
      color: #666;
    }}
    @top-right {{
      content: "VIT Bhopal University • README Documentation";
      font-family: 'Helvetica Neue', Arial, sans-serif;
      font-size: 8pt;
      color: #666;
    }}
    @bottom-center {{
      content: "Page " counter(page);
      font-family: 'Helvetica Neue', Arial, sans-serif;
      font-size: 9pt;
      color: #444;
    }}
  }}

  body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
    color: #1a202c;
    background-color: #ffffff;
    line-height: 1.55;
    font-size: 9.5pt;
    margin: 0;
    padding: 0;
  }}

  .header-card {{
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #ffffff;
    padding: 22px 24px;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  }}

  .header-card h1 {{
    color: #f6ad55;
    margin: 0 0 8px 0;
    font-size: 18pt;
    border-bottom: none;
    padding-bottom: 0;
  }}

  .header-meta {{
    font-size: 9.5pt;
    color: #e2e8f0;
    margin-top: 6px;
    line-height: 1.4;
  }}

  h1.h1-title {{
    display: none; /* replaced by header-card */
  }}

  h2 {{
    font-size: 13pt;
    color: #1a365d;
    border-bottom: 2px solid #3182ce;
    padding-bottom: 4px;
    margin-top: 22px;
    margin-bottom: 10px;
    page-break-after: avoid;
  }}

  h3 {{
    font-size: 10.5pt;
    color: #2b6cb0;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}

  h4 {{
    font-size: 9.5pt;
    color: #2d3748;
    margin-top: 10px;
    margin-bottom: 4px;
  }}

  p {{
    margin: 6px 0;
  }}

  ul, ol {{
    margin: 6px 0 10px 20px;
    padding: 0;
  }}

  li {{
    margin-bottom: 4px;
  }}

  hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 16px 0;
  }}

  .table-container {{
    margin: 12px 0;
    page-break-inside: avoid;
    overflow-x: auto;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
    margin: 0;
  }}

  th, td {{
    border: 1px solid #cbd5e0;
    padding: 6px 8px;
    text-align: left;
  }}

  th {{
    background-color: #edf2f7;
    color: #2d3748;
    font-weight: 700;
  }}

  tr:nth-child(even) {{
    background-color: #f7fafc;
  }}

  .code-container {{
    margin: 10px 0;
    page-break-inside: avoid;
  }}

  pre.code-block {{
    background-color: #0f172a;
    color: #e2e8f0;
    padding: 10px 12px;
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8pt;
    line-height: 1.35;
    overflow-x: auto;
    margin: 0;
  }}

  code {{
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8.5pt;
    background-color: #edf2f7;
    color: #c53030;
    padding: 1px 4px;
    border-radius: 3px;
  }}

  .callout {{
    background-color: #f8fafc;
    border-left: 4px solid #3182ce;
    padding: 10px 14px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
    font-size: 9pt;
    page-break-inside: avoid;
  }}

  .callout-warning {{
    border-left-color: #dd6b20;
    background-color: #fffaf0;
  }}

  .callout-title {{
    font-weight: 700;
    color: #2b6cb0;
    margin-bottom: 4px;
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 0.5px;
  }}

  .callout-warning .callout-title {{
    color: #c05621;
  }}

  .figure-box {{
    text-align: center;
    margin: 16px 0;
    page-break-inside: avoid;
  }}

  .figure-img {{
    max-width: 90%;
    border: 1px solid #cbd5e0;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}

  .equation {{
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    padding: 6px 12px;
    margin: 8px 0;
    border-radius: 4px;
    text-align: center;
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-size: 10pt;
    page-break-inside: avoid;
  }}

  .inline-math {{
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-style: italic;
  }}

  a {{
    color: #2b6cb0;
    text-decoration: none;
  }}
</style>
</head>
<body>

<div class="header-card">
  <h1>☀️ SolarVision: Automated Solar Active Region Detection &amp; Analysis</h1>
  <div class="header-meta">
    <strong>Author:</strong> Tribhuwan Singh &nbsp;|&nbsp; 
    <strong>Registration ID:</strong> 24BAI10358 &nbsp;|&nbsp; 
    <strong>Institution:</strong> VIT Bhopal University<br/>
    <strong>Course:</strong> B.Tech Computer Vision Course Project &nbsp;|&nbsp;
    <strong>Repository:</strong> https://github.com/tribhu05/SolarVision
  </div>
</div>

{body_html}

</body>
</html>
"""
    return html


def generate_pdf():
    print("[1/3] Converting README.md to styled HTML...")
    html_content = build_readme_html()
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"      README HTML written to: {OUTPUT_HTML} ({len(html_content)} chars)")

    if not BROWSER_EXE:
        print("[-] Error: Browser engine (Edge/Chrome) not found.")
        sys.exit(1)

    print(f"[2/3] Compiling README.pdf via browser engine ({BROWSER_EXE})...")
    cmd = [
        BROWSER_EXE,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={OUTPUT_PDF}",
        str(OUTPUT_HTML),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if res.returncode != 0:
        print(f"[-] Compilation error: {res.stderr}")
        sys.exit(res.returncode)

    if not OUTPUT_PDF.exists():
        print("[-] Error: README.pdf was not created.")
        sys.exit(1)

    size_kb = OUTPUT_PDF.stat().st_size / 1024
    print(f"[3/3] README.pdf compiled successfully!")
    print(f"      Output Path: {OUTPUT_PDF}")
    print(f"      File Size:   {size_kb:.1f} KB")


if __name__ == "__main__":
    generate_pdf()

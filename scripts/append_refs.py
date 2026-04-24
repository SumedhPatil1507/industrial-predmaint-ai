"""Appends References and About page handlers to frontend/app.py"""
addition = (
    "\n\n"
    "# =============================================================================\n"
    "# REFERENCES & CITATIONS\n"
    "# =============================================================================\n"
    "elif page == \"\U0001f4da References & Citations\":\n"
    "    from frontend.page_references import render as render_refs\n"
    "    render_refs()\n"
    "\n\n"
    "# =============================================================================\n"
    "# ABOUT\n"
    "# =============================================================================\n"
    "elif page == \"\u2139\ufe0f About\":\n"
    "    from frontend.page_about import render as render_about\n"
    "    render_about()\n"
)

with open("frontend/app.py", "a", encoding="utf-8") as f:
    f.write(addition)

print("Done")

import ast
with open("frontend/app.py", encoding="utf-8") as f:
    src = f.read()
lines = src.splitlines()
print(f"Total lines: {len(lines)}")
try:
    ast.parse(src)
    print("Syntax: OK")
except SyntaxError as e:
    print(f"Syntax ERR line {e.lineno}: {e.msg}")

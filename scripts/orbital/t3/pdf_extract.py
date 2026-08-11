import sys
from pypdf import PdfReader
r = PdfReader(sys.argv[1])
out = []
for i, p in enumerate(r.pages):
    try:
        out.append(f"\n=== PAGE {i+1} ===\n" + (p.extract_text() or ""))
    except Exception as e:
        out.append(f"\n=== PAGE {i+1} ERR {e} ===")
print("".join(out))

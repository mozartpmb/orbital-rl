import sys, re
from pypdf import PdfReader
def text(p, pages=None):
    r = PdfReader(p)
    out=[]
    rng = range(len(r.pages)) if pages is None else pages
    for i in rng:
        try: out.append(f"\n===PAGE {i+1}===\n"+r.pages[i].extract_text())
        except Exception as e: out.append(f"\n===PAGE {i+1} ERR {e}===")
    return "\n".join(out)
if __name__=="__main__":
    p=sys.argv[1]
    pg=None
    if len(sys.argv)>2:
        a,b=sys.argv[2].split("-"); pg=range(int(a)-1,int(b))
    print(text(p,pg))

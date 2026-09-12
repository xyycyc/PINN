from pathlib import Path
import sys,re,hashlib,json
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
sys.stdout.reconfigure(encoding='utf-8')
r=Path(__file__).resolve().parent
results=[]
for p in sorted((r/'final').glob('*.pdf')):
    doc=pdfium.PdfDocument(p);textdoc=PdfReader(p)
    out=r/'qa'/p.stem;out.mkdir(parents=True,exist_ok=True)
    for i,page in enumerate(doc):
        im=page.render(scale=1.6).to_pil().convert('RGB')
        im.save(out/f'page-{i+1}.png')
    txt='\n'.join(page.extract_text() or '' for page in textdoc.pages)
    (out/'extracted.txt').write_text(txt,encoding='utf-8',errors='replace')
    results.append({'file':p.name,'pages':len(doc),'chars':len(txt),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    print(p.name,'pages',len(doc),'chars',len(txt))
    for j,page in enumerate(textdoc.pages):
        t=page.extract_text() or ''
        print('  ',j+1,'chars',len(t),'tail',t[-70:].replace('\n',' '))
    ims=list(out.glob('page-*.png'))
    for start in range(0,len(doc),3):
        chosen=[Image.open(out/f'page-{i+1}.png') for i in range(start,min(start+3,len(doc)))]
        w,h=chosen[0].size
        contact=Image.new('RGB',(w*len(chosen),h+30),'#ddd')
        draw=ImageDraw.Draw(contact)
        for i,im in enumerate(chosen):
            contact.paste(im,(i*w,30));draw.text((i*w+10,8),f'{p.stem[:2]} / page {start+i+1}',fill='black')
        contact.save(out/f'contact-{start//3+1}.png')
orig=PdfReader(r/'qa'/'original_word'/'01_AI模块程序使用说明.pdf')
copied=PdfReader(r/'final'/'01_AI模块程序使用说明.pdf')
norm=lambda text:re.sub(r'\s+','',text)
a=norm(''.join(p.extract_text() or '' for p in orig.pages))
b=norm(''.join(p.extract_text() or '' for p in copied.pages))
print('ORIGINAL_WORD_PDF_TEXT_EQUAL',a==b,'lengths',len(a),len(b))
source=Path(r'D:\Desktop\待交付')
for ext in ['docx','pdf']:
    print('REUSE_IDENTICAL',ext,(source/('技术文档_ai模块.'+ext)).read_bytes()==(r/'final'/('01_AI模块程序使用说明.'+ext)).read_bytes())
(r/'qa'/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

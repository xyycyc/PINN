from pathlib import Path
import sys
from docx import Document
from docx.shared import Pt,RGBColor
from docx.oxml.ns import qn
sys.stdout.reconfigure(encoding='utf-8')
root=Path(__file__).resolve().parent
src=Path(r'D:\Desktop\待交付\AI模块使用维护及培训资料')
out=root/'renamed';out.mkdir(exist_ok=True)
mapping={
 '01_AI模块程序使用说明':'程序使用说明',
 '02_AI模块维护手册':'维护手册',
 '03_AI模块程序调试维修方法':'程序调试维修方法',
 '04_AI模块培训记录':'培训记录',
}
replacements=list(mapping.items())+[(a[3:],b) for a,b in mapping.items()]
for old,new in mapping.items():
 d=Document(src/(old+'.docx'))
 for t in d.element.iter(qn('w:t')):
  text=t.text or ''
  for a,b in replacements:text=text.replace(a,b)
  t.text=text
 if new=='程序使用说明':
  p=d.paragraphs[0].insert_paragraph_before(new,style='Title')
  p.paragraph_format.space_before=Pt(0)
  p.paragraph_format.space_after=Pt(10)
  p.paragraph_format.keep_with_next=True
  for run in p.runs:
   run.font.name='宋体';run.font.size=Pt(20);run.font.bold=True;run.font.color.rgb=RGBColor(0,0,0)
   run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
  st=d.styles['Title']
  for border in list(st.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
 d.core_properties.title=new
 d.save(out/(new+'.docx'))
 print(new)

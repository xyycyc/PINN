from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
import hashlib,json
R=Path(__file__).resolve().parent
source=Path(r'D:\Desktop\待交付\培训记录参考.docx')
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
ns={'w':W}
def tag(n): return '{'+W+'}'+n
with ZipFile(source) as z:
    parts={n:z.read(n) for n in z.namelist()}
root=E.fromstring(parts['word/document.xml']);body=root.find('w:body',ns)
paras=body.findall('w:p',ns);tables=body.findall('w:tbl',ns)
def text(p,s):
    ts=p.findall('.//w:t',ns)
    if ts:
        ts[0].text=s
        for t in ts[1:]:t.text=''
    else:
        r=E.SubElement(p,tag('r'));E.SubElement(r,tag('t')).text=s
def cell(t,r,c,s):
    ce=tables[t].findall('w:tr',ns)[r].findall('w:tc',ns)[c]
    ps=ce.findall('w:p',ns)
    text(ps[0],s)
    for p in ps[1:]:ce.remove(p)
text(paras[8],'培训记录')
text(paras[10],'基于人工智能的温度场反演方法模块培训记录')
goals=[
'参训人员能够启动人工智能模块，设置数据与结果路径，识别Python与Fortran的功能分工。',
'参训人员能够组织三类材料的波形与温度场数据，完成样本配对、固定节点采样及数据校验。',
'参训人员能够按材料和工况完成模型训练与预测，掌握MSE及MSE＋PINN训练模式和材料路由。',
'参训人员能够执行在线增量训练，核对参数冻结配置，保存基础模型、增量模型和更新记录。',
'参训人员能够核查五测点平均温度误差与完整场推理效率，完成数据、模型、配置和结果归档。']
for i,s in enumerate(goals,14):text(paras[i],s)
cell(0,1,1,'基于人工智能的温度场反演方法模块交付培训')
cell(0,6,2,'交付目录')
modules=[
('系统组成','三类材料；CNN—LSTM—BP模型；Python与Fortran分工'),
('环境与启动','Windows 10/11；Python 3.10；Intel oneAPI Fortran；图形化入口与路径设置'),
('数据构建','波形与温度場配对；材料标识；manifest清单与材料独立划分'.replace('場','场')),
('固定物理节点','10000个固定节点；编号、坐标、温度、材料与界面身份；统一采样索引'),
('模型训练','MSE与MSE＋PINN模式；物理和平滑损失权重；验证评估与模型保存'),
('温度场预测','材料路由；完整二维场；归一化x=0.50中心线的一维结果'),
('数据校验','数据规模、格式、温度范围、网格与采样一致性；核查报告'),
('增量训练','基础模型与新增实验数据；参数冻结和指定参数更新；新模型及日志'),
('精度与效率','五测点先求平均再算相对误差；四类工况；完整场平均推理时间'),
('维护与归档','分类存储和定期备份；清单、模型、配置、日志及版本对应'),
('调试维修','接口、数据配对、模型配置、测点映射检查；处理记录与结果复核')]
for i,(name,s) in enumerate(modules,1):
    cell(1,i,1,name);cell(1,i,2,s)
practices=[
('启动界面并设置路径','界面打开，输入、结果与工作目录设置正确'),
('构建材料数据库','波形与温度场完成配对，生成清单和固定采样索引'),
('配置并运行模型训练','生成对应材料模型、训练配置和损失日志'),
('执行预测并查看结果','生成二维温度场或一维中心线，模型与材料对应'),
('校验数据与模型信息','完成格式、网格、采样及模型信息核查，保存报告'),
('执行在线增量训练','生成新模型，保留冻结配置、更新记录与基础模型'),
('核查五测点误差与效率','测点映射一致，先求平均再算误差，记录完整场时间'),
('记录调试并归档资料','处理过程可复核，数据、配置、模型、日志和结果对应')]
for i,(name,s) in enumerate(practices,1):cell(2,i,1,name);cell(2,i,2,s)
assess=[
('基本操作','可独立设置路径，完成建库、训练、预测及增量训练'),
('输入检查','能核对波形与温度场配对、材料、固定节点和数据清单'),
('结果识别','能查看场结果、中心线、五测点误差和完整场效率记录'),
('故障处理','能核对接口、模型及测点映射，记录处理过程与复核结果'),
('维护操作','能分类备份资料，核对模型版本并归档配置、日志和结果')]
for i,(name,s) in enumerate(assess,1):cell(4,i,0,name);cell(4,i,1,s)
cell(6,6,1,'程序使用说明；维护手册；程序调试维修方法')
# Keep each repeated table header with at least its first data row.
for tbl in tables[:8]:
    for p in tbl.findall('w:tr',ns)[0].findall('.//w:p',ns):
        pr=p.find('w:pPr',ns)
        if pr is None:pr=E.Element(tag('pPr'));p.insert(0,pr)
        keep=pr.find('w:keepNext',ns)
        if keep is None:keep=E.SubElement(pr,tag('keepNext'))
        keep.set(tag('val'),'1')
# Keep only the first reference module's structure, adapted to the AI module.
start=body.index(paras[36])
for el in list(body)[start:]:
    if el.tag!=tag('sectPr'):body.remove(el)
# Remove the reference's page-break-only separator after its first module.
for p in [paras[34],paras[35]]:
    if not ''.join(p.itertext()).strip() or p.findall('.//w:br',ns):
        if p.getparent() is body:body.remove(p)
parts['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
out=R/'final';out.mkdir(exist_ok=True)
with ZipFile(out/'培训记录.docx','w',ZIP_DEFLATED) as z:
    for name,data in parts.items():z.writestr(name,data)
with ZipFile(source) as a, ZipFile(out/'培训记录.docx') as b:
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
assert changed==['word/document.xml'],changed
alltext=''.join(root.itertext())
for forbidden in ['Worker','bootstrap','self test','SimulationPlatform','gui.py','TOF','Python 3.9.13','VTU','PVD','ParaView']:
    assert forbidden not in alltext,forbidden
(R/'source_map.json').write_text(json.dumps({'format_source':str(source),'technical_source':r'D:\Desktop\技术报告合稿-0904_1609.docx','technical_paragraphs':'1599;1631-1635;1659-1667;1697-1737;1742-1759;1764-1783;1818-1819;1954;2024-2025;2081;2614-2639','changed_package_parts':changed,'blank_fields':'培训日期、地点、人员、软件版本、完成情况、实际结果、考核结论及签字'},ensure_ascii=False,indent=2),encoding='utf-8')
print(out/'培训记录.docx')

"""Generate vector illustrations for the normalization article (stdlib only)."""
from pathlib import Path
from html import escape
O=Path(__file__).resolve().parents[1]/'img/in-post'
blue='#377ac0';orange='#d37a32';green='#218578';purple='#8860ae';colors=[blue,orange,green,purple]
def begin(title,desc,h):
 return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{h}" viewBox="0 0 1000 {h}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#64748b"/></marker></defs><style>text{{font-family:"Noto Sans CJK SC","Noto Sans CJK JP","Microsoft YaHei",sans-serif;fill:#24354a}} text.small{{font-size:17px}}</style><rect width="1000" height="{h}" fill="#f7f9fc"/>',f'<text x="500" y="48" text-anchor="middle" font-size="28" font-weight="bold">{escape(title)}</text>']
def text(s,x,y,t,size=20,anchor='middle'): s.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}">{escape(str(t))}</text>')
def rect(s,x,y,w,h,fill,stroke='none',radius=6):s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')
def line(s,x,y,xx,yy,c='#aebdcd',arrow=False):s.append(f'<path d="M{x} {y}L{xx} {yy}" stroke="{c}" stroke-width="2" fill="none"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
def write(s,name): (O/f'ai-infra-norm-{name}.svg').write_text('\n'.join(s+['</svg>'])+'\n')
def vector(s,x,y,vals,c=orange,w=65):
 for i,v in enumerate(vals):
  rect(s,x+i*w,y,w-4,46,'#fff',c);text(s,x+i*w+(w-4)/2,y+30,v,18)
# Actual matrix with cells selected by statistic direction.
s=begin('同一张特征矩阵，两种统计方向','二维输入中 BatchNorm 沿样本轴统计一列，LayerNorm 沿特征轴统计一行。',470)
for offset,title,is_bn in [(40,'BatchNorm：同一特征，跨样本',True),(525,'LayerNorm：同一样本，跨特征',False)]:
 text(s,offset+215,104,title,21)
 vals=[[1,2,3,4],[10,20,30,40],[100,200,300,400]]
 for j in range(4):text(s,offset+98+j*76,152,f'特征 {j}',17)
 for i,row in enumerate(vals):
  text(s,offset+28,202+i*60,chr(65+i),18)
  for j,v in enumerate(row):
   selected=(j==0 if is_bn else i==0)
   rect(s,offset+62+j*76,172+i*60,71,51,'#d8e9fa' if selected else '#e8edf3')
   text(s,offset+98+j*76,205+i*60,v,20)
 text(s,offset+215,390,'统计 [1, 10, 100]' if is_bn else '统计 [1, 2, 3, 4]',22)
 text(s,offset+215,425,'每列一组均值、方差' if is_bn else '每行一组均值、方差',18)
write(s,'axes')
# Number line with consistent feature identity.
s=begin('同一个向量：移动中心，或缩小幅度','四种颜色和编号对应四个特征；数轴展示原始值及两种归一化结果，gamma 为一，beta 为零，忽略 epsilon。',550)
text(s,500,89,'示例 x = [1, 2, 3, 4]；γ = 1，β = 0，忽略 ε',18)
for y,label,vs,note in [(180,'原始 x',[1,2,3,4],'均值 2.5'),(310,'LayerNorm',[-1.342,-.447,.447,1.342],'减去均值，再缩放'),(440,'RMSNorm',[.365,.730,1.095,1.461],'只缩放，保留比例')]:
 text(s,30,y-28,label,22,'start');text(s,30,y+7,note,16,'start')
 def pos(v):return 260+(v+2)/6*690
 line(s,260,y,965,y)
 for tick in range(-2,5):
  xx=pos(tick);line(s,xx,y-6,xx,y+6);text(s,xx,y+30,tick,16)
 for j,(v,c) in enumerate(zip(vs,colors)):
  xx=pos(v);line(s,xx,y-35,xx,y,c)
  s.append(f'<circle cx="{xx}" cy="{y-35}" r="13" fill="{c}"/>')
  s.append(f'<text x="{xx}" y="{y-30}" text-anchor="middle" font-size="14" style="fill:white">{j}</text>')
text(s,500,524,'圆点编号对应特征 0、1、2、3；三行使用同一数轴尺度',18)
write(s,'values')
# Residual: show literal vectors retained vs overwritten; F output numerical example.
s=begin('追踪残差里的数值，而不只看 Norm 的位置','假设某次残差相加得到 h 等于 1 2 3 4；Pre-Norm 保留原 h 和归一化副本 z，Post-Norm 把归一化结果作为后续表示。',760)
text(s,500,89,'假设上一次相加得到 [1, 2, 3, 4]；使用 LayerNorm，γ=1、β=0',17)
for x in [30,515]:rect(s,x,120,455,580,'#fff','#dbe3ec',14)
text(s,255,160,'Pre-Norm：保留原始 h',24);text(s,740,160,'Post-Norm：相加后归一化',24)
vector(s,110,192,[1,2,3,4]);vector(s,590,192,[1,2,3,4])
text(s,255,275,'原始 h 继续留在残差分支',18)
line(s,130,244,80,320,orange)
line(s,80,320,80,603,orange)
line(s,80,603,104,603,orange,True)
text(s,295,323,'为加工分支生成 z = Norm(h)',17)
vector(s,150,345,['−1.34','−0.45','0.45','1.34'],blue,72)
text(s,290,425,'FFN(z) 的示例输出',18);vector(s,150,446,[.1,.2,.3,.4],green,72)
text(s,255,552,'最后加回的仍是 [1, 2, 3, 4]',19)
vector(s,110,580,[1.1,2.2,3.3,4.4],orange)
text(s,255,669,'y = h + FFN(z)',23)
line(s,740,244,740,315,arrow=True);text(s,740,294,'Norm',20)
vector(s,580,345,['−1.34','−0.45','0.45','1.34'],blue,78)
text(s,740,431,'归一化后的向量成为新 h',20)
text(s,740,473,'后续计算分支和残差分支',20)
text(s,740,508,'都从这个蓝色向量出发',20)
rect(s,568,558,345,70,'#edf4fc')
text(s,740,600,'原始 [1, 2, 3, 4] 不再直通',20)
text(s,740,669,'h = Norm(上一次残差之和)',21)
text(s,500,738,'橙色：未归一化的残差表示　蓝色：归一化后的表示',18)
write(s,'residual')
# GPU: physical row strip, thread registers and partial sums of explicit input 1..16.
s=begin('一行特征，分给四个线程处理','十六个特征值等于一到十六；线程按下标模四分工，局部平方和为276 336 404 480，合并为1496。',730)
text(s,500,88,'缩小示例：D = 16，4 个线程；实际可用 256 个线程处理 D = 4096',17)
text(s,65,142,'HBM',23)
for j in range(16):
 x=65+j*55;rect(s,x,164,51,49,colors[j%4]);s.append(f'<text x="{x+25}" y="196" text-anchor="middle" font-size="20" style="fill:white">{j+1}</text>');text(s,x+25,238,j,15)
text(s,500,266,'上排是特征值；下排是下标。颜色表示负责该元素的线程。',17)
rect(s,40,300,920,355,'#e9eef5','#c5d1df',16);text(s,500,334,'GPU 线程块：读取后尽量保存在寄存器中',22)
for t,c in enumerate(colors):
 x=65+t*225
 rect(s,x,365,195,185,'#fff',c,10);text(s,x+97,398,f'线程 {t}',22)
 vals=list(range(t+1,17,4));text(s,x+97,442,'  '.join(map(str,vals)),22)
 text(s,x+97,482,'局部平方和',18);text(s,x+97,522,sum(v*v for v in vals),25)
text(s,500,602,'归约：276 + 336 + 404 + 480 = 1496',24)
text(s,500,635,'共享 inv_rms = 1 / √(1496 / 16 + ε)',21)
text(s,500,695,'各线程复用自己的 x，计算 y = γ × x × inv_rms，再写回显存',19)
write(s,'cuda')
print('Generated four normalization illustrations')

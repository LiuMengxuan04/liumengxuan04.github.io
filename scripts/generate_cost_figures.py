"""Rebuild parameter/FLOPs SVG illustrations using the standard library."""
from pathlib import Path
from html import escape
OUT=Path(__file__).resolve().parents[1]/'img/in-post'
B='#377ac0';G='#218578';A='#d37a32';P='#8860ae'
def start(title,desc,h):
 return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{h}" viewBox="0 0 1000 {h}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><style>text{{font-family:"Noto Sans CJK SC","Noto Sans CJK JP","Microsoft YaHei",sans-serif;fill:#24354a}}</style><rect width="1000" height="{h}" fill="#f7f9fc"/>',f'<text x="500" y="48" text-anchor="middle" font-size="28" font-weight="bold">{escape(title)}</text>']
def text(s,x,y,t,size=20,anchor='middle'):s.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}">{escape(str(t))}</text>')
def rect(s,x,y,w,h,c):s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{c}"/>')
def grid(s,x,y,rows,cols,k,color,values=None,select=None):
 for r in range(rows):
  for c in range(cols):
   fill=color if select is None or select(r,c) else '#e4eaf1'
   rect(s,x+c*k,y+r*k,k-4,k-4,fill)
   if values: text(s,x+c*k+(k-4)/2,y+r*k+k*.6,values[r][c],18)
def save(s,name):(OUT/f'ai-infra-cost-{name}.svg').write_text('\n'.join(s+['</svg>'])+'\n')
s=start('参数数格子，FLOPs 数每个格子的计算','二乘三输入和三乘四权重得到二乘四输出；示例展示一个输出为一乘一加二乘二加三乘三等于十四。',490)
text(s,500,91,'同一份 W 服务所有输入行；增加输入行数不会增加参数',19)
grid(s,60,160,2,3,55,'#d9eafa',[[1,2,3],[4,5,6]],lambda r,c:r==0)
grid(s,375,132,3,4,55,'#f8dfbf',[[1,0,2,1],[2,1,0,3],[3,-1,1,2]],lambda r,c:c==0)
grid(s,740,160,2,4,48,'#ccece6',[[14,-1,5,13],[32,-1,14,31]],lambda r,c:r==0 and c==0)
text(s,290,220,'×',38);text(s,665,220,'=',38)
text(s,137,313,'X：2 × 3',23);text(s,482,313,'W：3 × 4',23);text(s,835,313,'Y：2 × 4',23)
text(s,500,365,'高亮输出：1 × 1 + 2 × 2 + 3 × 3 = 14',23)
text(s,500,413,'W 有 12 个参数；Y 有 8 个元素，每个执行 3 次乘法 + 2 次加法',20)
text(s,500,455,'精确计算：8 × 5 = 40 FLOPs；常用估算：2 × 2 × 3 × 4 = 48 FLOPs',19)
save(s,'matmul')
s=start('6.74B 参数主要存在哪里','三十二个等大的 Block，每个包含约三分之一 Attention 投影与三分之二 FFN，整体约百分之九十六点一参数位于 Block。',690)
text(s,500,88,'LLaMA-2-7B：d=4096，d_ff=11008，32 个 Block',19)
for i in range(32):
 x=65+(i%8)*110;y=132+(i//8)*78
 rect(s,x,y,34,43,B);rect(s,x+34,y,68,43,G);text(s,x+51,y+64,f'Block {i+1}',15)
text(s,500,471,'每个 Block：Attention 67.11M　+　FFN 135.27M　+　Norm 0.008M',19)
rect(s,140,494,15,15,B);text(s,169,508,'Attention',18,'start');rect(s,350,494,15,15,G);text(s,379,508,'FFN',18,'start');text(s,695,508,'Norm 太小，未单独绘制',17)
vals=[131072000,6476267520,4096,131072000];cs=[A,G,P,A];x=65
for v,c in zip(vals,cs):
 w=870*v/sum(vals);rect(s,x,552,w,43,c);x+=w
text(s,500,626,'32 个 Block：96.11%　|　Embedding + LM Head：3.89%',20)
text(s,500,662,'上方每个块的面积表示参数组成；下方按全模型参数量比例绘制',17)
save(s,'parameters')
s=start('序列翻倍：行数 ×2，配对数量 ×4','矩阵格子比较四 token 和八 token 的线性层输入及全 Attention 分数矩阵，底部显示单 Query 的 Decode 匹配行。',770)
text(s,235,104,'线性层输入 X',24);text(s,700,104,'Attention 分数 QK^T',24)
grid(s,70,150,4,4,30,'#d9eafa');grid(s,265,150,8,4,30,'#d9eafa')
text(s,125,428,'S=4：4 行',20);text(s,320,428,'S=8：8 行',20)
grid(s,505,150,4,4,30,'#f8dfbf');grid(s,690,150,8,8,30,'#f8dfbf')
text(s,560,428,'4² = 16 格',20);text(s,807,428,'8² = 64 格',20)
text(s,235,475,'特征宽度 d 不变，计算量约 ×2',19);text(s,720,475,'两条序列轴都变长，计算量约 ×4',19)
text(s,500,535,'Decode：一个新 Query 对照 C 个缓存位置',24)
grid(s,225,573,1,12,46,'#ccece6');text(s,155,603,'1 个 Q',19)
text(s,500,664,'1 × C 的匹配行：单步计算随缓存长度线性增长',20)
text(s,500,716,'方阵按完整配对示意；因果 Attention 可跳过未来位置',18)
save(s,'context')
s=start('单 token Decode：算得少，也可能搬得久','相同零到七毫秒轴比较峰值算术时间零点零四三毫秒与权重搬运时间六点七毫秒，使用理想算力带宽估算并非实测。',430)
text(s,500,90,'粗略假设：13.4 GFLOPs，13.4 GB 权重，312 TFLOP/s，2 TB/s',18)
left=255;scale=90
for t in range(8):
 x=left+t*scale;s.append(f'<path d="M{x} 130V310" stroke="#dce4ed"/>');text(s,x,342,str(t),17)
text(s,225,179,'峰值算术时间',20,'end');rect(s,left,151,.043*scale,40,B);text(s,left+15,179,'0.043 ms',20,'start')
text(s,225,270,'权重搬运时间',20,'end');rect(s,left,242,6.7*scale,40,A);text(s,870,270,'6.7 ms',20,'start')
text(s,575,374,'时间 / ms（同一线性刻度）',18)
text(s,500,410,'仅比较两种理想约束，不是实测延迟；KV Cache 与其他开销未计入',17)
save(s,'time')
print('Generated 4 cost illustrations')

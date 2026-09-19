"""Regenerate FFN article SVGs: python3 -m pip install matplotlib; run this file."""
from pathlib import Path
import math
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'img' / 'in-post'
plt.rcParams.update({'font.family': ['Noto Sans CJK JP', 'DejaVu Sans'], 'font.size': 13,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'path', 'axes.unicode_minus': False})
COLORS = ['#2864b7', '#c34e24', '#25836b']
xs = [-5 + i / 100 for i in range(1001)]
phi = lambda x: (1 + math.erf(x / math.sqrt(2))) / 2
sig = lambda x: 1 / (1 + math.exp(-x))
funcs = [lambda x: max(0, x), lambda x: x * phi(x), lambda x: x * sig(x)]
names = ['ReLU', 'GELU（精确）', 'SiLU（β=1）']

def style(ax, title):
    ax.set_title(title, pad=16, fontsize=16)
    ax.axhline(0, color='#8b98a8', lw=.8)
    ax.axvline(0, color='#8b98a8', lw=.8)
    ax.grid(alpha=.16)
    ax.set_xlabel('输入 x')

def save(fig, name):
    fig.savefig(OUT / f'ai-infra-ffn-{name}.svg', bbox_inches='tight', facecolor='white')
    path = OUT / f'ai-infra-ffn-{name}.svg'
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
for ax in axes:
    for fun, label, color in zip(funcs, names, COLORS):
        ax.plot(xs, [fun(x) for x in xs], label=label, color=color, lw=2.5)
    ax.set_ylabel('输出 f(x)')
    ax.legend(loc='upper left', frameon=False)
style(axes[0], '激活函数：整体曲线')
axes[0].set_xlim(-5, 5); axes[0].set_ylim(-.6, 5.2)
style(axes[1], '局部放大：负值并非全部归零')
axes[1].set_xlim(-3, 1); axes[1].set_ylim(-.36, 1.05)
for x, fn, color, label, tx, ty in [(-.75179, funcs[1], COLORS[1], 'GELU 最低约 −0.170', -2.85, .36), (-1.27846, funcs[2], COLORS[2], 'SiLU 最低约 −0.278', -2.85, .14)]:
    axes[1].plot(x, fn(x), 'o', color=color)
    axes[1].annotate(label, (x, fn(x)), xytext=(tx, ty), fontsize=11, color=color, arrowprops={'arrowstyle': '->', 'color': color})
save(fig, 'activation-curves')

fig, ax = plt.subplots(figsize=(11, 4.7), layout='constrained')
for beta, color in zip([0, .5, 1, 5], ['#8972b5', '#bc7825', '#25836b', '#2864b7']):
    ax.plot(xs, [x * sig(beta*x) for x in xs], lw=2.4, label=f'β = {beta}' + ('（SiLU）' if beta == 1 else ''), color=color)
ax.plot(xs, [max(0,x) for x in xs], '--', color='#444', label='ReLU（β → +∞）', lw=1.6)
style(ax, 'Swish：β 控制从线性到近似 ReLU 的变化')
ax.set_xlim(-5, 5); ax.set_ylim(-2.7, 5.2); ax.set_ylabel('x · sigmoid(βx)')
ax.legend(loc='upper left', frameon=False)
save(fig, 'swish-beta')

fig, ax = plt.subplots(figsize=(11, 4.7), layout='constrained')
ax.plot([-5, 0], [0, 0], color=COLORS[0], lw=2.5, label='ReLU（x=0 不可微）')
ax.plot([0, 5], [1, 1], color=COLORS[0], lw=2.5)
ax.scatter([0, 0], [0, 1], facecolors='white', edgecolors=COLORS[0], zorder=5)
ax.plot(xs, [phi(x)+x*math.exp(-x*x/2)/math.sqrt(2*math.pi) for x in xs], lw=2.5, color=COLORS[1], label='GELU 导数')
ax.plot(xs, [sig(x)+x*sig(x)*(1-sig(x)) for x in xs], lw=2.5, color=COLORS[2], label='SiLU 导数')
style(ax, '梯度对比：平滑不等于梯度始终为正')
ax.set_xlim(-5, 5); ax.set_ylim(-.2, 1.25); ax.set_ylabel("导数 f′(x)")
ax.legend(loc='upper left', frameon=False)
ax.annotate('GELU / SiLU 在零点的导数均为 0.5', (0,.5), xytext=(.7,.30), fontsize=12, arrowprops={'arrowstyle':'->','color':'#596579'})
save(fig, 'activation-gradients')

# Diagrams are vector-native, with accessible titles and descriptions.
def svg(title, desc, height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">', f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(desc)}</desc>', '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0L9 4.5L0 9Z" fill="#63758b"/></marker></defs>', '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#23344d} .line{stroke:#63758b;stroke-width:2.5;fill:none;marker-end:url(#arrow)}</style>', f'<rect width="1200" height="{height}" fill="#f7f9fc"/>', f'<text x="600" y="48" text-anchor="middle" font-size="29" font-weight="bold">{html.escape(title)}</text>']
def text(s, x, y, label, size=20):
    s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}">{html.escape(label)}</text>')
def box(s, x, y, w, h, label, fill='#e6effb', sub=None):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#bccbdf"/>')
    text(s,x+w/2,y+h/2+(7 if sub is None else -3),label)
    if sub: text(s,x+w/2,y+h/2+25,sub,16)
def arrow(s,path): s.append(f'<path d="{path}" class="line"/>')
def write(s,name): (OUT/f'ai-infra-ffn-{name}.svg').write_text('\n'.join(s+['</svg>']))

s=svg('Attention 汇聚上下文，FFN 逐位置加工', '简化子模块数据流，省略 Norm 和残差；因果 Attention 只访问当前及历史位置，同一层 FFN 权重在所有 token 间共享。',560)
text(s,600,83,'下图省略 Norm 和残差，仅突出 token 之间是否发生交互',17)
for x,label in [(150,'位置 1'),(600,'位置 2'),(1050,'位置 3')]:
    box(s,x-95,110,190,55,label)
    box(s,x-110,250,220,65,'上下文表示 h',sub=None)
    box(s,x-110,370,220,75,'同一个 FFN',fill='#e5f3ed',sub='共享 θ，逐 token 计算')
    arrow(s,f'M{x} 315V360')
    arrow(s,f'M{x} 445V480')
    box(s,x-95,488,190,48,'输出：D 维')
for i,x in enumerate([150,600,1050]):
    for prev in [150,600,1050][:i+1]: arrow(s,f'M{prev} 165L{x} 240')
s.append('<rect x="490" y="197" width="220" height="32" rx="8" fill="#f7f9fc"/>')
text(s,600,220,'因果 Attention',18)
text(s,375,348,'FFN 不直接读取其他位置的向量',17)
write(s,'role')

s=svg('标准 FFN 与 SwiGLU：两种中间特征加工方式', '标准 FFN 有两个投影；SwiGLU 有门控和内容两个上投影，逐元素乘积后下投影。',650)
text(s,600,82,'每个方框中的维度都针对一个 token；批量计算时在前面加上 [B, S]',17)
text(s,160,128,'标准 FFN：2 个矩阵',22)
for x,w,label,sub in [(45,120,'输入','D'),(215,190,'Up 投影','D → M'),(455,175,'GELU / ReLU','M → M'),(680,190,'Down 投影','M → D'),(920,200,'输出','D')]: box(s,x,155,w,80,label,sub=sub)
for a,b in [(165,205),(405,445),(630,670),(870,910)]: arrow(s,f'M{a} 195H{b}')
text(s,175,303,'SwiGLU：3 个矩阵',22)
box(s,45,398,120,80,'输入',sub='D')
box(s,240,330,200,80,'Gate 投影',sub='D → M')
box(s,485,330,155,80,'SiLU',fill='#f1eafa',sub='M → M')
box(s,240,495,200,80,'Up 投影',sub='D → M')
box(s,690,398,140,80,'逐元素 ×',fill='#f1eafa',sub='M')
box(s,880,398,160,80,'Down 投影',sub='M → D')
box(s,1080,398,95,80,'输出',sub='D')
for path in ['M165 438H200V370H230','M200 438V535H230','M440 370H475','M640 370H760V388','M440 535H760V488','M830 438H870','M1040 438H1070']: arrow(s,path)
text(s,620,612,'SiLU(Gate) 可以为负或大于 1；两条分支的形状必须一致才能逐元素相乘',18)
write(s,'swiglu')

s=svg('8/3 倍中间维度：把参数预算配平', '忽略偏置和取整，标准 FFN 与 SwiGLU 总参数均为 8D²；SwiGLU 继续采用 4D 宽度则为 12D²。',490)
text(s,600,85,'横条宽度与参数量成正比；每个色块代表一个权重矩阵',17)
for y,label,pieces,total in [(155,'标准 FFN：M = 4D',[('Up：4D²',4),('Down：4D²',4)],'8D²'),(270,'SwiGLU：M = 8D/3',[('Gate：8D²/3',8/3),('Up：8D²/3',8/3),('Down：8D²/3',8/3)],'8D²'),(385,'SwiGLU：M = 4D',[('Gate：4D²',4),('Up：4D²',4),('Down：4D²',4)],'12D²')]:
    text(s,600,y-20,label,21)
    x=130
    for (label,n),color in zip(pieces,['#e6effb','#e5f3ed','#f1eafa']):
        box(s,x,y,n*70,58,label,fill=color); x+=n*70
    text(s,x+60,y+36,total,21)
text(s,600,480,'同参数预算：2 × D × 4D = 3 × D × M  ⇒  M = 8D/3',21)
write(s,'budget')
print('Generated 6 SVG figures in', OUT)

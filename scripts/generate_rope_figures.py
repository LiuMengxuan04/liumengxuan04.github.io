"""Draw the sinusoidal positional encoding vs RoPE comparison (stdlib only).

Run:  python3 scripts/generate_rope_figures.py
Out:  img/in-post/ai-infra-rope-vs-sinusoidal.svg
"""
from pathlib import Path
from html import escape
import re

OUT = Path(__file__).resolve().parents[1] / 'img' / 'in-post'
W, H = 1600, 1020
ML, PW, GAP = 40, 732, 56
PLX, PRX = ML, ML + PW + GAP
PY, PH = 150, 790

SANS = '-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif'
SERIF = "Cambria,'Times New Roman',Georgia,serif"  # 单引号版，供 style 属性内联使用
WARM, WARM_D, WARM_BG = '#a33a29', '#7a2618', '#fbeae5'
COOL, COOL_D, COOL_BG = '#2c6ba6', '#1f4d7c', '#e7f0fa'
INK, MUT, RULE, EDGE = '#1a2733', '#5c6b7c', '#d3dae3', '#8a99a8'

s = []
s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">')
s.append('<title id="title">正弦位置编码与旋转位置编码 RoPE 的机制对照</title>')
s.append('<desc id="desc">左栏正弦位置编码把位置向量加到投影之前的 embedding 上，右栏 RoPE 把二维旋转乘到投影之后的 Q、K 上；因此前者的位置项与绝对位置有关，后者只与相对位置有关。</desc>')
s.append(f'<defs><marker id="a" markerWidth="9" markerHeight="9" refX="7.5" refY="4.5" orient="auto"><path d="M0 0L9 4.5L0 9Z" fill="{EDGE}"/></marker><style>'
         f'text{{font-family:{SANS};fill:{INK}}}'
         f'.t{{font-size:42px;font-weight:800;fill:#14202e}}'
         f'.s{{font-size:21px;fill:{MUT}}}'
         f'.h{{font-size:29px;font-weight:750;fill:#fff}}'
         f'.cap{{font-size:17px;font-weight:700;fill:{MUT};letter-spacing:.14em}}'
         f'.lbl{{font-size:18px;fill:{MUT}}}'
         f'.chip{{font-size:20px;font-weight:700;fill:#fff}}'
         f'.conc{{font-size:26px;font-weight:750;fill:#fff}}'
         f'.note{{font-size:18px;fill:{MUT}}}'
         '</style></defs>')
s.append(f'<rect width="{W}" height="{H}" fill="#f6f8fb"/>')

SUB = re.compile(r'(~[^~]+~|\^[^\^]+\^)')


def t(x, y, txt, cls, anchor='middle'):
    s.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}">{escape(str(txt))}</text>')


def r(x, y, w, h, fill='none', stroke='none', sw=2, rx=8):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def line(x, y, xx, yy, c=EDGE, sw=2.5):
    s.append(f'<path d="M{x} {y}L{xx} {yy}" stroke="{c}" stroke-width="{sw}" fill="none" marker-end="url(#a)"/>')


def m(x, y, spec, anchor='middle', size=26, color=None):
    """Math text; ~x~ is a subscript, ^x^ is a superscript.

    baseline-shift 自带回本位，不需要（也不能用）dy 手动复位：空 tspan 的 dy
    会被浏览器忽略，导致后面的正文一级级下沉。
    """
    out = []
    for tok in SUB.split(spec):
        if not tok:
            continue
        if tok.startswith('~'):
            d = round(size * .30)
            out.append(f'<tspan style="font-size:{round(size * .68)}px;baseline-shift:-{d}px">{escape(tok[1:-1])}</tspan>')
        elif tok.startswith('^'):
            d = round(size * .36)
            out.append(f'<tspan style="font-size:{round(size * .68)}px;baseline-shift:{d}px">{escape(tok[1:-1])}</tspan>')
        else:
            out.append(escape(tok))
    style = f'font-family:{SERIF};font-size:{size}px;fill:{color or "#16222f"}'
    s.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="{style}">{"".join(out)}</text>')


def caption(px, y, txt):
    t(px + PW / 2, y, txt, 'cap')


def panel(px, accent, header, chip):
    r(px, PY, PW, PH, '#ffffff', RULE, 2, 16)
    r(px, PY, PW, 76, accent, 'none', 0, 16)
    r(px, PY + 60, PW, 16, accent, 'none', 0, 0)
    t(px + 34, PY + 49, header, 'h', 'start')
    cw = 148
    s.append(f'<rect x="{px + PW - 34 - cw}" y="{PY + 16}" width="{cw}" height="44" rx="22" fill="#ffffff" opacity=".18"/>')
    t(px + PW - 34 - cw / 2, PY + 45, chip, 'chip')


def flow(px, nodes, gap, y=470, h=60):
    total = sum(w for _, w, _ in nodes) + gap * (len(nodes) - 1)
    x = px + (PW - total) / 2
    boxes = []
    for spec, w, op in nodes:
        if op:
            r(x, y, w, h, op[1], op[0], 2, 10)
            m(x + w / 2, y + h / 2 + 8, spec, size=23, color=op[0])
        else:
            r(x, y, w, h, '#ffffff', RULE, 2, 10)
            m(x + w / 2, y + h / 2 + 8, spec, size=23, color='#2a3a4c')
        boxes.append((x, w))
        x += w + gap
    for (x1, w1), (x2, _) in zip(boxes, boxes[1:]):
        line(x1 + w1 + 4, y + h / 2, x2 - 5, y + h / 2)
    return boxes


# ── 标题 ────────────────────────────────────────────────────────────────
t(W / 2, 66, '位置编码：加在输入上，还是乘在 Q、K 上', 't')
t(W / 2, 104, '同一个问题，正弦位置编码与 RoPE 的两种解法', 's')

# ── 左栏：正弦位置编码 ──────────────────────────────────────────────────
panel(PLX, WARM, '正弦位置编码', '向量加法')

caption(PLX, 276, '核心公式')
m(PLX + PW / 2, 316, 'PE(m, 2i) = sin(ω~i~ · m)')
m(PLX + PW / 2, 360, 'PE(m, 2i+1) = cos(ω~i~ · m)')

caption(PLX, 430, '注入阶段')
flow(PLX, [('x~m~', 108, None), ('+ PE(m)', 168, (WARM, WARM_BG)), ('W~Q~', 120, None), ('q~m~', 108, None)], 32)
t(PLX + PW / 2, 576, '位置向量加在投影之前', 'lbl')

caption(PLX, 634, 'q、k 的写法')
m(PLX + PW / 2, 682, 'q~m~ = W~Q~(x~m~ + PE(m))')
m(PLX + PW / 2, 722, 'k~n~ = W~K~(x~n~ + PE(n))')

caption(PLX, 768, '内积中的位置项')
m(PLX + PW / 2, 816, 'PE(m)^T^ PE(n) = Σ~i~ cos(ω~i~ (n − m))')

r(PLX + 34, 854, PW - 68, 62, WARM_D, 'none', 0, 10)
t(PLX + PW / 2, 895, '和绝对位置 m、n 有关', 'conc')

# ── 右栏：RoPE ─────────────────────────────────────────────────────────
panel(PRX, COOL, '旋转位置编码 RoPE', '矩阵乘法')

caption(PRX, 276, '核心公式')
lblw, cell, mh = 190, 150, 52
grp = lblw + 24 + cell * 2
x0 = PRX + (PW - grp) / 2
m(x0 + lblw / 2, 329, 'RoPE(m, i) =')
bx = x0 + lblw + 24
r(bx - 8, 258, cell * 2 + 26, mh * 2 + 20, COOL_BG, 'none', 0, 8)
for cx, cy, spec in [(0, 0, 'cos(ω~i~m)'), (1, 0, '−sin(ω~i~m)'), (0, 1, 'sin(ω~i~m)'), (1, 1, 'cos(ω~i~m)')]:
    m(bx + 9 + cx * cell + cell / 2, 297 + cy * mh, spec, size=23)
line(bx + 9 + cell, 264, bx + 9 + cell, 368, '#b9cde0', 1.5)
s.append(f'<path d="M{bx + 2} 262H{bx - 8}V370H{bx + 2}" fill="none" stroke="{COOL}" stroke-width="3"/>')
s.append(f'<path d="M{bx + cell * 2 + 16} 262H{bx + cell * 2 + 26}V370H{bx + cell * 2 + 16}" fill="none" stroke="{COOL}" stroke-width="3"/>')

caption(PRX, 430, '注入阶段')
flow(PRX, [('x~m~', 92, None), ('W~Q~', 104, None), ('q~m~', 92, None), ('RoPE(m)', 150, (COOL, COOL_BG)), ("q'~m~", 104, None)], 26)
t(PRX + PW / 2, 576, '位置旋转加在投影之后', 'lbl')

caption(PRX, 634, 'q、k 的写法')
m(PRX + PW / 2, 682, "q'~m~ = RoPE(m) (W~Q~ x~m~)")
m(PRX + PW / 2, 722, "k'~n~ = RoPE(n) (W~K~ x~n~)")

caption(PRX, 768, '内积中的位置项')
m(PRX + PW / 2, 816, 'RoPE(m)^T^ RoPE(n) = RoPE(n − m)')

r(PRX + 34, 854, PW - 68, 62, COOL_D, 'none', 0, 10)
t(PRX + PW / 2, 895, '只和相对位置 n − m 有关', 'conc')

# ── 脚注 ───────────────────────────────────────────────────────────────
t(W / 2, 980, '注：左栏“内积中的位置项”是化简写法，真实点积中还夹着投影矩阵，前三项仍依赖绝对位置；右栏则是严格成立的恒等式', 'note')

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'ai-infra-rope-vs-sinusoidal.svg').write_text('\n'.join(s + ['</svg>']) + '\n', encoding='utf-8')
print('Generated ai-infra-rope-vs-sinusoidal.svg')

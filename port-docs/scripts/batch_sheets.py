"""Contact sheets for the 9b batch: front (+ optional back) renders, labelled with id, type, outcome.
Usage: batch_sheets.py <table.csv> <sim_out_dir> <out_dir> [per_sheet=8]   (JPEG output)"""
import sys, csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
rows = list(csv.DictReader(open(sys.argv[1]))); sim = Path(sys.argv[2]); out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
per = int(sys.argv[4]) if len(sys.argv) > 4 else 8
cols = 4; W = 400  # tile width for front, back
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
except Exception: font = ImageFont.load_default()
def tile(r):
    n = r['name']; ims = []
    for side in ('front', 'back'):
        p = sim / n / f'{n}_render_{side}.png'
        im = Image.open(p).convert('RGB') if p.exists() else Image.new('RGB', (800, 800), (60, 0, 0))
        ims.append(im.resize((W // 2 * 1, W // 2 * 1)) if False else im.resize((W, W)))
    t = Image.new('RGB', (W * 2, W + 42), 'white'); t.paste(ims[0], (0, 42)); t.paste(ims[1], (W, 42))
    d = ImageDraw.Draw(t)
    typ = '/'.join(str(r[k]) for k in ('upper', 'wb', 'bottom') if r[k] not in ('', 'None'))
    d.text((4, 2), f"{n[5:]} {typ}", fill='black', font=font)
    fl = r.get('flags_final', r.get('flags')); first = r.get('flags_first', fl)
    lab = fl.replace('cloth_', '').replace('_intersection', '').replace('static_equilibrium', 'static') + ('' if first == fl else ' (1st: ' + first.replace('static_equilibrium', 'static') + ')')
    d.text((4, 22), f"{lab} s={r.get('self_final', r['self_int'])} b={r.get('body_final', r['body_int'])} fr={r['frames']}", fill='red' if fl != 'pass' else 'darkgreen', font=font)
    return t
rows_per = per // 2   # two garments per row (front+back each)
for s in range(0, len(rows), per):
    chunk = rows[s:s + per]; tiles = [tile(r) for r in chunk]
    tw, th = tiles[0].size; ncol = 2; nrow = (len(tiles) + 1) // 2
    sheet = Image.new('RGB', (tw * ncol, th * nrow), 'white')
    for i, t in enumerate(tiles): sheet.paste(t, ((i % ncol) * tw, (i // ncol) * th))
    f = out / f'{sys.argv[5] if len(sys.argv) > 5 else "sheet"}_{s // per + 1:02d}.jpg'
    sheet.save(f, quality=70); print(f, sheet.size)

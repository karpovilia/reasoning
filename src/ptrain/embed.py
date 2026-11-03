"""Fixed all-MiniLM-L6-v2 text vectors for a pack (architecture A).
Recipe as FINAL_TEXT_BINDING_V7: attention-mask mean pooling, L2 normalisation, max 256 wordpieces,
FP32 eager with TF32 off, output float16, dimension 384. Encoder assets are verified against
ENCODER_ASSET_BASE_BINDING_V7 sha256 before use.
Input: PACK/text_rows.json (list of text sha256 in row order) and a sqlite with table texts(sha,text).
Output: PACK/embeddings.npy, then PACK_MANIFEST.json files[] updated with its sha256."""
import argparse, hashlib, json, os, sqlite3, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptrain.data import sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pack', required=True); ap.add_argument('--texts', required=True)
    ap.add_argument('--encoder', required=True); ap.add_argument('--binding', required=True); ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    import numpy as np, torch
    from transformers import AutoModel, AutoTokenizer
    b = json.loads(Path(a.binding).read_text())
    for name, row in b['files'].items():
        if sha256_file(Path(a.encoder) / name) != row['sha256']:
            raise ValueError('ENCODER_ASSET_SHA:' + name)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(a.encoder); model = AutoModel.from_pretrained(a.encoder, attn_implementation='eager').to(a.device).float().eval()
    pack = Path(a.pack); rows = json.loads((pack / 'text_rows.json').read_text())
    con = sqlite3.connect(f'file:{a.texts}?mode=ro', uri=True)
    out = np.lib.format.open_memmap(pack / 'embeddings.npy.tmp', mode='w+', dtype=np.float16, shape=(len(rows), 384))
    t0 = time.time()
    for s in range(0, len(rows), a.batch):
        shas = rows[s:s + a.batch]
        got = dict(con.execute('select sha,text from texts where sha in (%s)' % ','.join('?' * len(shas)), shas).fetchall())
        missing = [h for h in shas if h not in got]
        if missing:
            raise ValueError('TEXT_MISSING:%d' % len(missing))
        for h in shas:
            if hashlib.sha256(got[h].encode('utf-8')).hexdigest() != h:
                raise ValueError('TEXT_SHA_MISMATCH:' + h)
        enc = tok([got[h] for h in shas], padding=True, truncation=True, max_length=256, return_tensors='pt').to(a.device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
            m = enc['attention_mask'].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
            v = torch.nn.functional.normalize(v, p=2, dim=1)
        out[s:s + len(shas)] = v.cpu().numpy().astype(np.float16)
        if (s // a.batch) % 200 == 0:
            print('rows', s + len(shas), 'of', len(rows), 'sec', round(time.time() - t0, 1), flush=True)
    out.flush(); del out
    os.replace(pack / 'embeddings.npy.tmp', pack / 'embeddings.npy')
    man = json.loads((pack / 'PACK_MANIFEST.json').read_text())
    man['files']['embeddings.npy'] = sha256_file(pack / 'embeddings.npy')
    man['embedding_recipe'] = dict(model='sentence-transformers/all-MiniLM-L6-v2', revision=b.get('revision'), pooling='attention mask mean, L2',
                                   max_wordpieces=256, fp32_tf32_off=True, dtype='float16', rows=len(rows), seconds=round(time.time() - t0, 1))
    tmp = pack / 'PACK_MANIFEST.json.tmp'; tmp.write_text(json.dumps(man, ensure_ascii=False, indent=1)); os.replace(tmp, pack / 'PACK_MANIFEST.json')
    print('EMBED_DONE', len(rows))


if __name__ == '__main__':
    main()

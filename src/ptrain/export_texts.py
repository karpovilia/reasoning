"""Write a sqlite with exactly the texts referenced by a pack (text_rows.json) from source sqlite(s).
Refuses a missing text or a text whose sha256 differs from its key."""
import hashlib, json, sqlite3, sys
from pathlib import Path


def main(pack, out, *sources):
    rows = json.loads((Path(pack) / 'text_rows.json').read_text()); need = set(rows)
    o = sqlite3.connect(out); o.execute('create table texts(sha TEXT PRIMARY KEY, text TEXT NOT NULL) WITHOUT ROWID')
    got = 0
    for s in sources:
        c = sqlite3.connect(f'file:{s}?mode=ro', uri=True)
        todo = sorted(need)
        for i in range(0, len(todo), 900):
            part = todo[i:i + 900]
            for sha, t in c.execute('select sha,text from texts where sha in (%s)' % ','.join('?' * len(part)), part):
                if hashlib.sha256(t.encode('utf-8')).hexdigest() != sha:
                    raise ValueError('TEXT_SHA_MISMATCH:' + sha)
                o.execute('insert into texts values (?,?)', (sha, t)); need.discard(sha); got += 1
    if need:
        raise ValueError('TEXTS_MISSING:%d' % len(need))
    o.commit(); print(json.dumps(dict(texts=got)))


if __name__ == '__main__':
    main(*sys.argv[1:])

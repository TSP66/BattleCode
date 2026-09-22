"""Waits for a set of battles to finish and prints results per opponent.

    python summarize.py <label> <first_id> <last_id>
"""
import json, sys, time, urllib.request, pathlib
KEY = next(iter(json.loads((pathlib.Path.home() / '.unswbc/keys.json').read_text()).values()))
A = 'https://game.battlecode.au/api/v1'
US = 62
label, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

def get(path):
    for i in range(8):
        try:
            req = urllib.request.Request(f'{A}/{path}', headers={'Authorization': f'Bearer {KEY}', 'User-Agent': 'bc-test'})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception:
            time.sleep(10)
    raise RuntimeError(path)

rows = {}
while True:
    for g in range(lo, hi + 1):
        if g in rows:
            continue
        d = get(f'battles/{g}')
        m = d['match']
        if m['status'] in ('completed', 'failed', 'errored'):
            us_a = m['teamAId'] == US
            opp = d['teamBName'] if us_a else d['teamAName']
            osub = m['submissionBId'] if us_a else m['submissionAId']
            res = 'err' if m['status'] != 'completed' else ('draw' if not m['winner'] else ('win' if (m['winner'] == 'a') == us_a else 'loss'))
            rows[g] = (opp, osub, d['mapName'], res)
        time.sleep(0.6)
    if len(rows) == hi - lo + 1:
        break
    time.sleep(60)
out = pathlib.Path(__file__).with_name(f'{label}.json')
out.write_text(json.dumps({g: r for g, r in sorted(rows.items())}, indent=1))
by = {}
for g, (opp, osub, mp, res) in sorted(rows.items()):
    by.setdefault((opp, osub), []).append((mp, res))
print(f'== {label}')
for (opp, osub), games in by.items():
    w = sum(r == 'win' for _, r in games); dr = sum(r == 'draw' for _, r in games)
    print(f'{opp} (their submission {osub}): {w}W {dr}D {len(games)-w-dr}L | ' + ', '.join(f'{m[:9]} {r}' for m, r in games))

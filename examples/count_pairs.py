import os
from mllf.mlp.setup_pairs import assemble_pairs

runs = assemble_pairs(os.path.join('examples', 'training_files'))
total = 0
per = {}
for run, pairs in runs.items():
    count = 0
    sites = {}
    for key, p in pairs.items():
        site = p.get('site')
        sub = p.get('sub')
        if site is None or sub is None:
            continue
        sites.setdefault(site, {})[sub] = p
    for site, subs in sites.items():
        subs_ids = sorted(subs.keys())
        for a in subs_ids:
            for b in subs_ids:
                if a == b:
                    continue
                p_a = subs[a]
                pw = p_a.get('biases', {}).get('pairwise_biases', {})
                key = f'pair_{a}_{b}'
                ok = True
                for g in ('lams','cs','ss','xs'):
                    if g not in pw or key not in pw[g]:
                        ok = False
                        break
                if ok:
                    count += 1
    per[run] = count
    total += count

print('Per-run ordered pair counts:')
for r in sorted(per):
    print(r, per[r])
print('Total ordered pairs:', total)

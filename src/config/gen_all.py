"""Generate ALL S2 configs (old + new targets) + slurm scripts."""
import json, os

BASE = "/mnt/storage/qisheng/github/wav2vec_test"

# All unique (target, source) pairs
ALL_PAIRS = {
    # Old targets (40 pairs)
    'mt':  ['ar','cs','en','eo','hu','it','ro','tr'],
    'af':  ['cs','en','nl','tr'],
    'da':  ['en','fr','lv','nl','tr'],
    'ky':  ['ba','tr','tt','ug'],
    'tk':  ['ba','tr','tt','ug'],
    'kk':  ['ba','tr','tt','ug'],
    'sk':  ['cs','lv','ro','ru'],
    'id':  ['ar','ca','cs','eo','it','sw','ta'],
    # New targets (61 pairs)
    'sq':  ['ca','cs','eo','it','lt','lv','nl','ro','tr'],
    'ltg': ['lt','lv','ru','tr'],
    'ur':  ['ar','cs','en','lv','nl','ro','ta','tr'],
    'cy':  ['ar','ba','hu','it','nl','sw','tr','ug'],
    'gn':  ['cs','eo','it','ro','sw'],
    'tn':  ['eo','hu','lt','sw'],
    'am':  ['ar','ba','ca','eo','hu','it','sw','ta'],
    'he':  ['ar','cs','en','eo','fr','nl','ru','sw','tr','ug'],
    'az':  ['ba','eo','lt','tr','ug'],
}
HALF = {'mt':830,'af':65,'da':1379,'ky':807,'tk':285,'kk':268,'sk':2526,'id':1845,
        'sq':958,'ltg':1765,'ur':2541,'cy':2704,'gn':480,'tn':184,'am':126,'he':196,'az':47}

items = []
for tgt in ALL_PAIRS:
    for src in ALL_PAIRS[tgt]:
        for ms in ['53', 'base']:
            cdir = '26_6_5' if ms == '53' else '26_6_5_base'
            
            if ms == '53':
                ckpt_num = '8000' if src == 'ug' else '28150'
                s1 = f"/mnt/storage/qisheng/github/wav2vec_test/weights/{src}_9000-xlsr-53_general_1e5/checkpoint-{ckpt_num}"
            else:
                ckpt_num = '10000' if src == 'ug' else '26000'
                s1 = f"/mnt/storage/qisheng/github/wav2vec_test/weights/{src}_9000-xlsr-base_general_1e5/checkpoint-{ckpt_num}"
            
            cfg_name = f"{tgt}_100-{src}{ms}.py"
            slurm_name = f"{tgt}_100-{src}{ms}.sh"
            wdir = f"{tgt}_100-xlsr-{src}{ms}"
            
            items.append({
                'cfg_name': cfg_name, 'cfg_path': f"{BASE}/src/config/lora/{cdir}/{cfg_name}",
                'slurm_path': f"{BASE}/slurm_config/lora/{cdir}/{slurm_name}",
                'tgt': tgt, 'src': src, 'ms': ms, 's1': s1,
                'wdir': wdir, 'cdir': cdir, 'eval_size': HALF[tgt],
            })

print(f"Total items: {len(items)}")
with open(f'{BASE}/src/config/gen_manifest.json', 'w') as f:
    json.dump(items, f, indent=2)
print("Manifest saved")

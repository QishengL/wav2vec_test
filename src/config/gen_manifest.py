"""Generate S2 configs + slurm — using write_file tool for reliability."""
import os, json

BASE = "/mnt/storage/qisheng/github/wav2vec_test"
PAIRS = {
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
HALF = {'sq':958,'ltg':1765,'ur':2541,'cy':2704,'gn':480,'tn':184,'am':126,'he':196,'az':47}

def write_config(tgt, src, mtype):
    ms = mtype
    cdir = '26_6_5' if ms == '53' else '26_6_5_base'
    
    if ms == '53':
        ckpt_num = '8000' if src == 'ug' else '28150'
        s1 = f"/mnt/storage/qisheng/github/wav2vec_test/weights/{src}_9000-xlsr-53_general_1e5/checkpoint-{ckpt_num}"
    else:
        ckpt_num = '10000' if src == 'ug' else '26000'
        s1 = f"/mnt/storage/qisheng/github/wav2vec_test/weights/{src}_9000-xlsr-base_general_1e5/checkpoint-{ckpt_num}"
    
    cfg_name = f"{tgt}_100-{src}{ms}.py"
    cfg_path = os.path.join(BASE, 'src', 'config', 'lora', cdir, cfg_name)
    slurm_name = f"{tgt}_100-{src}{ms}.sh"
    slurm_path = os.path.join(BASE, 'slurm_config', 'lora', cdir, slurm_name)
    wdir = f"{tgt}_100-xlsr-{src}{ms}"
    
    # Record what we need to create
    return {
        'cfg_name': cfg_name,
        'cfg_path': cfg_path,
        'slurm_path': slurm_path,
        'slurm_name': slurm_name,
        'tgt': tgt,
        'src': src,
        'ms': ms,
        's1': s1,
        'wdir': wdir,
        'cdir': cdir,
        'eval_size': HALF[tgt],
    }

# Collect all items
items = []
for tgt in PAIRS:
    for src in PAIRS[tgt]:
        items.append(write_config(tgt, src, '53'))
        items.append(write_config(tgt, src, 'base'))

# Write to JSON for processing
manifest_path = os.path.join(BASE, 'src', 'config', 'gen_manifest.json')
with open(manifest_path, 'w') as f:
    json.dump(items, f, indent=2)

print(f"Manifest saved: {len(items)} items to {manifest_path}")
print("Now run: python3 src/config/gen_write_files.py")

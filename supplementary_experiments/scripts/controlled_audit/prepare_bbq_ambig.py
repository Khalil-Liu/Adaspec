"""Extract ambiguous records from a locally supplied official BBQ data directory."""
import argparse,json,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    files=sorted(a.source.glob('*.jsonl')) if a.source.is_dir() else [a.source]
    if not files:raise FileNotFoundError('No source jsonl files')
    seen=set();records=[]
    for file in files:
        for line in file.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            r=json.loads(line)
            if r['context_condition']!='ambig':continue
            key=(r['category'],r['example_id'])
            if key in seen:raise ValueError('Duplicate official example ID: '+str(key))
            seen.add(key);records.append(r)
    if not records:raise ValueError('No ambiguous examples; disambiguated records cannot substitute')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    manifest={'n':len(records),'selection':'all ambiguous examples from explicitly supplied local files; no label-based selection','sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('Saved',len(records),'ambiguous examples to',a.output)
if __name__=='__main__':main()

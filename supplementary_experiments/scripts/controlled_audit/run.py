"""Single entry point. Keeps logs, checks prerequisites, resumes completed stages."""
import argparse,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(HERE/'config.json'));p.add_argument('--tasks',nargs='+',choices=['preflight','existing_stats','cases','collect','train','truthfulqa','bbq_disambig','bbq_ambig','statistics','summarize'],default=['preflight','existing_stats','cases','collect','train','truthfulqa','bbq_disambig','bbq_ambig','statistics','summarize']);a=p.parse_args()
    cfg=Path(a.config).resolve();c=json.loads(cfg.read_text(encoding='utf-8'));out=ROOT/c['output_dir'];out.mkdir(parents=True,exist_ok=True)
    targets={'existing_stats':'existing_records/paired_statistics.json','preflight':'preflight.json','cases':'cases.json','collect':'router_pool.json','train':'router_test_report.json','truthfulqa':'truthfulqa_summary.json','bbq_disambig':'bbq_disambig_summary.json','bbq_ambig':'bbq_ambig_summary.json','statistics':'paired_statistics.json','summarize':'combined_summary.json'}
    failures=[]
    for task in a.tasks:
        if (out/targets[task]).exists():print('SKIP completed:',task,flush=True);continue
        cmd=[sys.executable,'-u',str(HERE/('existing_records.py' if task=='existing_stats' else 'paired_stats.py' if task=='statistics' else 'validate.py'))]
        if task not in ['statistics','existing_stats']:cmd += ['bbq','--subset',task[4:]] if task.startswith('bbq_') else [task]
        cmd += ['--config',str(cfg)]
        if task=='collect' and not c.get('confirm_projector_split'):
            failures.append(task);print('BLOCKED collect: confirm projection training IDs in config. Cases remain independent.',flush=True);continue
        if task in ['train'] and not (out/'router_pool.json').exists():failures.append(task);print('BLOCKED train: collect required',flush=True);continue
        if task in ['truthfulqa','bbq_disambig','bbq_ambig'] and not (out/'trained_router.json').exists():failures.append(task);print('BLOCKED',task,': train required',flush=True);continue
        log=out/('log_'+task+'.txt')
        with log.open('a',encoding='utf-8') as f:
            f.write('\nCOMMAND '+repr(cmd)+'\n');f.flush()
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:print(line,end='',flush=True);f.write(line);f.flush()
            rc=process.wait()
        if rc:failures.append(task);print('FAILED',task,'see',log,flush=True)
    (out/'run_status.json').write_text(json.dumps({'requested':a.tasks,'failed_or_blocked':failures},indent=2),encoding='utf-8')
    if failures:raise SystemExit(1)
if __name__=='__main__':main()

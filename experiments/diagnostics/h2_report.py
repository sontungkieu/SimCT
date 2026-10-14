"""Honest partial/complete H2 receipts and prompt-clustered estimates."""
import json
import math
from pathlib import Path

from h2_pilot import CONFIG, OUT, write, marker
from h2_measurements import js, bootstrap_prompt_mean


def normalized(logps):
    peak=max(logps); weights=[math.exp(x-peak) for x in logps]; total=sum(weights)
    return [x/total for x in weights]


def entropy(p): return -sum(x*math.log(x) for x in p if x)


def entropy_matched(logps,target):
    lo=.01; hi=100.
    for _ in range(60):
        tau=(lo+hi)/2; p=normalized([x/tau for x in logps])
        if entropy(p)<target: lo=tau
        else: hi=tau
    tau=(lo+hi)/2
    return normalized([x/tau for x in logps]),tau


def report():
    assets=json.loads((OUT/'assets.json').read_text())
    stages={}
    for path in OUT.glob('*-receipt.json'):
        r=json.loads(path.read_text()); stages[path.stem]={'exit_code':r['exit_code'],'elapsed_seconds':r['elapsed_seconds']}
    missing=[]
    for arm in ('S','G9','Q','L'):
        for kind in ('parity','quality'):
            path=OUT/(kind+'-'+arm+('.json' if kind=='parity' else '-scores.json'))
            if not path.exists(): missing.append(kind+'-'+arm)
            elif kind=='parity' and not json.loads(path.read_text())['pass']: missing.append('parity-failed-'+arm)
    e2_path=OUT/'e2-records.jsonl'
    e2=[json.loads(line) for line in e2_path.read_text().splitlines()] if e2_path.exists() else []
    e2_summary={}
    for arm in ('Q','L'):
        for cat in ('all','content','digit','latex','punctuation','whitespace'):
            selected=[r for r in e2 if not r['calibration'] and not r['first_five'] and 'X_E2_'+arm in r
                      and (cat=='all' or r['category']==cat)]
            if selected: e2_summary[arm+'/'+cat]=bootstrap_prompt_mean(selected,'X_E2_'+arm,repeats=CONFIG['bootstrap_replicates'])
    e3_rows=[]; audits={}; e3_summary={}
    union_path=OUT/'e3-candidates-union.json'
    if union_path.exists():
        union=json.loads(union_path.read_text()); arms=union['arms']
        scored={a:json.loads((OUT/('e3-scores-'+a+'.json')).read_text()) for a in arms if (OUT/('e3-scores-'+a+'.json')).exists()}
        calibrations={a:json.loads((OUT/('calibration-'+a+'.json')).read_text()) for a in scored}
        if len(scored)!=len(arms): missing.append('e3-score-incomplete')
        else:
            for row in union['rows']:
                item=dict(row); native={a:scored[a].get(row['key'],{}).get('native',{}) for a in arms}
                keep=[i for i in range(len(row['candidates'])) if all(native[a].get('candidate_loglikelihoods',[])
                      and native[a]['candidate_loglikelihoods'][i] is not None for a in arms)]
                if len(keep)<2:
                    audits['insufficient_joint_canonical_candidates']=audits.get('insufficient_joint_canonical_candidates',0)+1; continue
                distributions={}; calibrated={}; masses={}; cal_masses={}
                for a in arms:
                    grid=calibrations[a]['temperatures']; k=grid.index(calibrations[a]['best_temperature']); raw_k=grid.index(1.)
                    raw=[native[a]['candidate_loglikelihoods'][i][raw_k] for i in keep]
                    cal=[native[a]['candidate_loglikelihoods'][i][k] for i in keep]
                    distributions[a]=normalized(raw); calibrated[a]=normalized(cal)
                    masses[a]=sum(math.exp(x) for x in raw); cal_masses[a]=sum(math.exp(x) for x in cal)
                    if masses[a]>1+1e-5 or cal_masses[a]>1+1e-5:
                        raise RuntimeError('Candidate path probabilities exceed one; overlapping path bug')
                b={a:js(distributions['S'],distributions[a]) for a in arms}
                cb={a:js(calibrated['S'],calibrated[a]) for a in arms}
                shape={}; taus={}
                target=entropy(calibrated['S'])
                for a in arms:
                    matched,tau=entropy_matched([math.log(max(x,1e-300)) for x in calibrated[a]],target)
                    shape[a]=js(calibrated['S'],matched); taus[a]=tau
                item.update(candidate_indices_kept=keep,E3_JS=b,E3_JS_calibrated=cb,E3_JS_shape=shape,
                            candidate_path_mass=masses,calibrated_candidate_path_mass=cal_masses,event_entropy_match_tau=taus)
                for a in ('Q','L'):
                    if a in arms:
                        item['X_E3_'+a]=b[a]-b['G9']; item['X_E3_calibrated_'+a]=cb[a]-cb['G9']
                        item['X_E3_shape_'+a]=shape[a]-shape['G9']
                # Paired template sensitivity uses the same jointly valid candidates.
                plain={a:scored[a].get(row['key'],{}).get('plain',{}) for a in arms}
                if all(plain[a].get('candidate_loglikelihoods') and
                       all(plain[a]['candidate_loglikelihoods'][i] is not None for i in keep) for a in arms):
                    plain_p={a:normalized([plain[a]['candidate_loglikelihoods'][i][2] for i in keep]) for a in arms}
                    plain_b={a:js(plain_p['S'],plain_p[a]) for a in arms}
                    for a in ('Q','L'):
                        if a in arms: item['X_E3_plain_'+a]=plain_b[a]-plain_b['G9']
                e3_rows.append(item)
            for a in ('Q','L'):
                for cat in ('all','content','digit','latex','punctuation'):
                    selected=[r for r in e3_rows if not r['calibration'] and 'X_E3_'+a in r and (cat=='all' or r['category']==cat)]
                    if not selected: continue
                    for variant in ('','calibrated_','shape_','plain_'):
                        key='X_E3_'+variant+a; subset=[r for r in selected if key in r]
                        if subset: e3_summary[a+'/'+cat+'/'+(variant or 'raw')]=bootstrap_prompt_mean(subset,key,repeats=CONFIG['bootstrap_replicates'])
            write(OUT/'e3-records.json',{'rows':e3_rows,'audit':audits})
    else: missing.append('e3-unavailable')
    # This intentionally cannot grant the draft's Gate1 without L/capacity/placebo controls.
    missing.extend(['within-family-capacity-reference','Gemma27B-placebo'])
    summary={'run_id':CONFIG['run_id'],'scope':'H2 only; no OPD training/H1/H3',
             'config':CONFIG,'access_errors':assets['access_errors'],'stages':stages,
             'E2':e2_summary,'E3':e3_summary,'missing_controls_or_stages':missing,
             'scientific_status':'EXPLORATORY_PILOT_GATE1_NOT_ESTABLISHED',
             'limitations':['one frozen rollout seed; bootstrap measures prompt sampling uncertainty',
                 'finite E3 view conditions on jointly canonical candidates and records captured path mass',
                 'event-space entropy matching is a finite-candidate control, not a full text-distribution identity',
                 'quality scorer is a heuristic SimCT pilot control; no asserted teacher quality matching']}
    write(OUT/'h2-summary.json',summary); marker('SUMMARY_JSON',summary)

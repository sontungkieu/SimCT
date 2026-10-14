"""Finite next-word candidate probe with native chain-rule temperature controls.

The candidate view is conditional and approximate: it is not an exhaustive byte
event distribution. Prefix overlap and changed canonical prefix paths are audited.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

from h2_pilot import (CONFIG, OUT, model_entry, require_parity, responses, render,
                      load_hf, canonical_head, write, marker, scoring_record)
from h2_measurements import next_word_event, assert_prefix_free

TEMPERATURES = [.5, .75, 1., 1.25, 1.5, 2.]


def positions_and_indices(arm):
    positions = json.loads((OUT/'e3-position-manifest.json').read_text())
    pool = {r['id']:r for r in responses()}
    index = json.loads((OUT/'hidden'/arm/'index.json').read_text())
    return positions, pool, index


def native_prefix(tok, arm, position, row, record, policy='native'):
    from h2_measurements import strict_pairs
    s = scoring_record('S',row['id'])['spans']
    j = strict_pairs(s, record['spans'])[position['position']]
    if policy == 'native': return render(tok, row['prompt']) + record['content_ids'][:j], j
    raw = row['text'].encode()[:position['byte_start']].decode('utf-8')
    return tok.encode(row['prompt'] + '\n' + raw, add_special_tokens=True), j


def candidates(arm):
    import sglang
    import torch
    from transformers import AutoTokenizer
    from hf_frozen_sampler import terminal_token_ids
    from mp_opd_parity_worker import _engine_kwargs
    require_parity(arm)
    tok = AutoTokenizer.from_pretrained(model_entry(arm)['path'], local_files_only=True)
    positions, pool, index = positions_and_indices(arm)
    destination = OUT/('e3-candidates-'+arm+'.json')
    completed = json.loads(destination.read_text()) if destination.exists() else {}
    stop = set(terminal_token_ids(tok)); special = set(tok.all_special_ids)
    kwargs = _engine_kwargs(model_entry(arm)['path'], torch.cuda.get_device_name(), 'default', None)
    kwargs.update(dtype='bfloat16', mem_fraction_static=.8, context_length=8192,
                  disable_cuda_graph=True, max_running_requests=64, random_seed=42)
    engine = sglang.Engine(**kwargs); started = time.monotonic()
    try:
        for off in range(0, len(positions), 16):
            jobs = []; done_keys = []
            for p in positions[off:off+16]:
                key = p['response_id']+'/'+str(p['position'])
                if key in completed: continue
                row = pool[p['response_id']]; record = scoring_record(arm,p['response_id'])
                try: prefix,j = native_prefix(tok,arm,p,row,record)
                except (UnicodeDecodeError,KeyError):
                    completed[key]={'candidates':[],'error':'incomplete UTF8/prefix'}; continue
                top = [i for i in record['top64_ids'][j] if i not in special and i<len(tok)][:6]
                done_keys.append(key)
                for token in top: jobs.append((key,prefix,token))
            if jobs:
                outputs = engine.generate(input_ids=[prefix+[token] for key,prefix,token in jobs],
                    sampling_params={'temperature':0.,'max_new_tokens':12,'stop_token_ids':list(stop)})
                for (key,prefix,token),out in zip(jobs,outputs):
                    tail = [token]+out['output_ids']
                    end = next((i for i,v in enumerate(tail) if v in stop),len(tail))
                    prefix_text = tok.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    text = tok.decode(prefix+tail[:end],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    if not text.startswith(prefix_text): raise RuntimeError('Candidate decoding changes prefix')
                    event = next_word_event(text[len(prefix_text):])
                    item = completed.setdefault(key,{'candidates':[],'incomplete_extensions':0})
                    if event: item['candidates'].append(event)
                    else: item['incomplete_extensions']+=1
            for key in done_keys: completed.setdefault(key,{'candidates':[]})
            write(destination,completed)
            if off%64==0: marker('E3_CANDIDATES_PROGRESS',{'arm':arm,'positions':len(completed),'seconds':time.monotonic()-started})
    finally: engine.shutdown()
    for value in completed.values(): value['candidates']=sorted(set(value['candidates']))
    write(destination,completed)


def merge():
    positions = json.loads((OUT/'e3-position-manifest.json').read_text())
    paths = {a:OUT/('e3-candidates-'+a+'.json') for a in ('S','G9','Q','L')}
    arms = [a for a,path in paths.items() if path.exists()]
    if 'S' not in arms or 'G9' not in arms or not any(a in arms for a in ('Q','L')):
        raise RuntimeError('Incomplete E3 candidate generation')
    sources = {a:json.loads(paths[a].read_text()) for a in arms}
    pool = {r['id']:r for r in responses()}; result=[]
    for p in positions:
        key=p['response_id']+'/'+str(p['position']); union=set()
        for a in arms: union.update(sources[a][key]['candidates'])
        raw=pool[p['response_id']]['text'].encode()[p['byte_start']:]
        try: realized=next_word_event(raw.decode('utf-8'))
        except UnicodeDecodeError: realized=None
        if realized: union.add(realized)
        candidates=sorted(union)
        try: assert_prefix_free(candidates); error=None
        except ValueError as exc: error=str(exc)
        result.append(dict(p,key=key,candidates=candidates,realized=realized,error=error))
    write(OUT/'e3-candidates-union.json',{'arms':arms,'rows':result,
        'convention':'top6 per available model extended greedily up to12 tokens; include first boundary character; plus realized event',
        'scope':'finite conditional canonical-path approximation; not exhaustive text-event probability'})
    marker('E3_UNION_SUMMARY',{'arms':arms,'positions':len(result),'overlap_errors':sum(r['error'] is not None for r in result)})


def temperature_calibration(model,tok):
    import torch
    prompts=json.loads((OUT/'prompts.json').read_text())
    calibration=set(prompts['calibration_prompt_ids'])
    total=torch.zeros(len(TEMPERATURES),device='cuda'); tokens=0; used=[]
    from answer_verifier import extract_boxed_answer
    with torch.inference_mode():
        for row in prompts['pool']:
            if row['id'] not in calibration: continue
            prefix=render(tok,row['prompt'])
            gold=row['gold']; ids=tok.encode(gold,add_special_tokens=False)
            # Explicit calibration cap only; never truncates the student pool.
            target=ids[:1024]
            if not target: continue
            all_ids=prefix+target
            h=model.model(input_ids=torch.tensor([all_ids],device='cuda'),use_cache=False).last_hidden_state[0,len(prefix)-1:-1]
            labels=torch.tensor(target,device='cuda')
            for off in range(0,len(target),16):
                logits=canonical_head(model,h[off:off+16]); lab=labels[off:off+16]
                for k,tau in enumerate(TEMPERATURES):
                    scaled=logits/tau
                    total[k]+=(scaled.logsumexp(-1)-scaled.gather(1,lab[:,None]).squeeze(-1)).sum()
            tokens+=len(target); used.append({'prompt_id':row['id'],'tokens':len(target),'cap_hit':len(ids)>1024})
    if not tokens: raise RuntimeError('No calibration reference tokens')
    scores=(total/tokens).cpu().tolist(); best=min(range(len(scores)),key=scores.__getitem__)
    return {'temperatures':TEMPERATURES,'mean_reference_nll':scores,'best_temperature':TEMPERATURES[best],
            'tokens':tokens,'rows':used,'text':'held-out train reference solutions; exclude these prompt IDs from H2 estimates',
            'reference_cap_tokens':1024}


def score_candidates(arm):
    import torch
    require_parity(arm)
    union=json.loads((OUT/'e3-candidates-union.json').read_text())
    if arm not in union['arms']: raise RuntimeError('Arm absent from fixed candidate union')
    positions,pool,index=positions_and_indices(arm)
    model,tok=load_hf(arm)
    cal_path=OUT/('calibration-'+arm+'.json')
    if not cal_path.exists(): write(cal_path,temperature_calibration(model,tok))
    destination=OUT/('e3-scores-'+arm+'.json')
    done=json.loads(destination.read_text()) if destination.exists() else {}
    start=time.monotonic()
    # Paired template sensitivity on a fixed subset chosen before scoring.
    sensitivity=set(r['key'] for r in sorted(union['rows'],key=lambda r:hashlib.sha256(r['key'].encode()).hexdigest())[:100])
    with torch.inference_mode():
        for n,p in enumerate(union['rows']):
            if p['key'] in done: continue
            if p['error'] or len(p['candidates'])<2:
                done[p['key']]={'error':p['error'] or 'fewer than two complete candidates'}; continue
            row=pool[p['response_id']]; record=scoring_record(arm,p['response_id'])
            results={}
            for policy in ('native','plain') if p['key'] in sensitivity else ('native',):
                try: prefix,j=native_prefix(tok,arm,p,row,record,policy)
                except (UnicodeDecodeError,KeyError): results[policy]={'error':'incomplete UTF8/prefix'}; continue
                # One prefix prefill; each candidate gets an independent cache clone.
                base=model.model(input_ids=torch.tensor([prefix],device='cuda'),use_cache=True)
                prefix_h=base.last_hidden_state[0,-1:]
                prefix_text=tok.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                logps=[]; flags=[]
                for candidate in p['candidates']:
                    full=tok.encode(prefix_text+candidate,add_special_tokens=False)
                    canonical=full[:len(prefix)]==prefix
                    suffix=full[len(prefix):] if canonical else []
                    if not suffix:
                        logps.append(None); flags.append('canonical prefix changed'); continue
                    if len(suffix)>1:
                        cloned=copy.deepcopy(base.past_key_values)
                        extension=model.model(input_ids=torch.tensor([suffix[:-1]],device='cuda'),
                                              past_key_values=cloned,use_cache=False)
                        h=torch.cat([prefix_h,extension.last_hidden_state[0]],dim=0)
                        del extension,cloned
                    else: h=prefix_h
                    logits=canonical_head(model,h)
                    lab=torch.tensor(suffix,device='cuda')
                    scores=[]
                    for tau in TEMPERATURES:
                        scaled=logits/tau
                        scores.append(float((scaled.gather(1,lab[:,None]).squeeze(-1)-scaled.logsumexp(-1)).sum()))
                    logps.append(scores); flags.append(None)
                results[policy]={'candidate_loglikelihoods':logps,'canonicalization_errors':flags}
                del base
            done[p['key']]=results
            write(destination,done)
            if n%20==0: marker('E3_SCORE_PROGRESS',{'arm':arm,'positions':n+1,'seconds':time.monotonic()-start})
    marker('E3_SCORE_SUMMARY',{'arm':arm,'positions':len(done),'elapsed_seconds':time.monotonic()-start})

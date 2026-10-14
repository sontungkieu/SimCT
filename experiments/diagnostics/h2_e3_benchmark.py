"""100-position performance proxy. Never creates/qualifies H2 scientific data."""
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time

from h2_pilot import CONFIG, OUT, render, load_hf, canonical_head, write, marker, model_entry
from h2_measurements import vocabulary_bytes, byte_spans, strict_pairs, category

SOURCE_RUN = 'h2-pilot-s42-n2-20261015-r1'
ARMS = ('S', 'G9', 'Q')
N = 100


def word(text, *, terminated=False):
    """Guide-style candidate parser, used only for performance workload sizing."""
    match = re.match(r'\s*(?:\\[A-Za-z]+|[0-9]|[^\W\d_]+|[^\w\s])', text, re.UNICODE)
    if not match: return None
    value = match.group()
    if match.end() == len(text) and not terminated and (value[-1].isalpha() or value[-1] == '\\'):
        return None
    return value


def prepare():
    from transformers import AutoTokenizer
    source = OUT.parent / SOURCE_RUN
    assets = json.loads((source / 'assets.json').read_text())
    for a in ARMS:
        e = assets['models'][a]
        if e['revision'] != CONFIG['models'][a]['revision'] or not e.get('verified_weights'):
            raise RuntimeError('Pinned model cache not prepared')
    write(OUT / 'assets.json', assets)
    prompts = json.loads((source / 'prompts.json').read_text())['pool']
    toks = {a: AutoTokenizer.from_pretrained(assets['models'][a]['path'], local_files_only=True) for a in ARMS}
    vocabs = {a: vocabulary_bytes(toks[a], 'gemma' if a in ('S','G9') else 'bpe') for a in ARMS}
    per_prompt = {}; strata = defaultdict(list); counts = Counter(); failures = []
    for row in prompts:
        text = row['gold']
        spans = {}; sequences = {}; sizes = {}
        try:
            for a in ARMS:
                prefix = render(toks[a], row['prompt'])
                prefix_text = toks[a].decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                full = toks[a].encode(prefix_text + text, add_special_tokens=False)
                if full[:len(prefix)] != prefix: raise ValueError('reference template boundary changed')
                if toks[a].decode(full,skip_special_tokens=False,clean_up_tokenization_spaces=False) != prefix_text+text:
                    raise ValueError('reference contextual roundtrip failed')
                content = full[len(prefix):]
                spans[a] = byte_spans(content, vocabs[a], text)
                sequences[a] = full; sizes[a] = len(prefix)
        except ValueError as exc:
            failures.append({'prompt_id':row['id'],'error':str(exc)}); continue
        aligned = {a: strict_pairs(spans['S'], spans[a]) for a in ARMS}
        keep = set.intersection(*(set(v) for v in aligned.values()))
        per_prompt[row['id']] = {'row':row,'full_ids':sequences,'prefix_sizes':sizes,'aligned':aligned}
        raw = text.encode()
        for j in sorted(keep):
            if j<5 or j>=1024: continue
            left,right,_ = spans['S'][j]
            cat = category(raw[left:right])
            if cat=='whitespace': continue
            try: raw[:left].decode('utf-8'); raw[left:].decode('utf-8')
            except UnicodeDecodeError: continue
            lengths = {a:sizes[a]+aligned[a][j] for a in ARMS}
            if max(lengths.values())>4096: continue
            length = lengths['S']; bucket = 'short' if length<128 else 'medium' if length<512 else 'long' if length<1024 else 'very_long'
            key = cat+'/'+bucket
            rank = hashlib.sha256(f"42/{row['id']}/{j}".encode()).hexdigest()
            item = {'key':row['id']+'/'+str(j),'prompt_id':row['id'],'position':j,'byte_start':left,
                    'category':cat,'length_bucket':bucket,'stratum':key,'rank':rank,
                    'reference_word':word(raw[left:].decode('utf-8'),terminated=True)}
            counts[key]+=1; strata[key].append(item)
            if len(strata[key])>200:
                strata[key].sort(key=lambda r:r['rank']); del strata[key][100:]
    for items in strata.values(): items.sort(key=lambda r:r['rank'])
    selected=[]
    while len(selected)<N:
        changed=False
        for key in sorted(strata):
            if strata[key] and len(selected)<N:
                selected.append(strata[key].pop(0)); changed=True
        if not changed: raise RuntimeError('Insufficient aligned reference fixture positions')
    sampled=Counter(r['stratum'] for r in selected)
    for item in selected:
        data=per_prompt[item['prompt_id']]; j=item['position']
        item['prefix_ids']={a:data['full_ids'][a][:data['prefix_sizes'][a]+data['aligned'][a][j]] for a in ARMS}
        item['sampling_weight']=counts[item['stratum']]/sampled[item['stratum']]
    fixture={'source_run':SOURCE_RUN,'source_manifest_sha256':hashlib.sha256((source/'prompts.json').read_bytes()).hexdigest(),
        'fixture_type':'held-out performance proxy from train reference solutions, NOT student rollouts',
        'models':list(ARMS),'positions':selected,'sampling':'seed42 hash stratified by category and S prefix-length bucket',
        'population_counts':dict(counts),'sample_counts':dict(sampled),'roundtrip_failures':failures,
        'scientific_data':False,'L_access':assets['access_errors'].get('L')}
    write(OUT/'benchmark-fixtures.json',fixture)
    marker('BENCH_FIXTURES',{'positions':N,'unique_prompts':len({r['prompt_id'] for r in selected}),
        'prefix_lengths':{a:{'min':min(len(r['prefix_ids'][a]) for r in selected),'max':max(len(r['prefix_ids'][a]) for r in selected)} for a in ARMS},
        'reference_roundtrip_failures':len(failures),'scientific_data':False})


def sync_time():
    import torch
    torch.cuda.synchronize(); return time.monotonic()


def proposal(arm):
    import torch
    from hf_frozen_sampler import terminal_token_ids
    rows=json.loads((OUT/'benchmark-fixtures.json').read_text())['positions']
    tick=sync_time(); model,tok=load_hf(arm); load_seconds=sync_time()-tick
    stop=set(terminal_token_ids(tok)); special=set(tok.all_special_ids)
    result=[]
    with torch.inference_mode():
        for n,row in enumerate(rows):
            start=sync_time(); ids=row['prefix_ids'][arm]
            base=model.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=True)
            logits=canonical_head(model,base.last_hidden_state[0,-1:])[0]
            top=[i for i in logits.topk(32).indices.tolist() if i<len(tok) and i not in special][:10]
            prefill=sync_time()-start
            prefix_text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            candidates=[]; incomplete=0; continuation_tokens=0
            for tid in top:
                tail=[tid]; cache=copy.deepcopy(base.past_key_values)
                for depth in range(4):
                    ended=tail[-1] in stop
                    whole=tok.decode(ids+([*tail[:-1]] if ended else tail),skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    if not whole.startswith(prefix_text): raise RuntimeError('Proposal contextual decode changes prefix')
                    candidate=word(whole[len(prefix_text):],terminated=ended)
                    if candidate or ended or depth==3: break
                    ext=model.model(input_ids=torch.tensor([[tail[-1]]],device='cuda'),past_key_values=cache,use_cache=True)
                    cache=ext.past_key_values
                    tail.append(int(canonical_head(model,ext.last_hidden_state[0,-1:]).argmax(-1)))
                    continuation_tokens+=1; del ext
                if candidate: candidates.append(candidate)
                else: incomplete+=1
                del cache
            elapsed=sync_time()-start; del base
            result.append({'key':row['key'],'candidates':sorted(set(candidates)),
                           'prefill_seconds':prefill,'wall_seconds':elapsed,'continuation_tokens':continuation_tokens,
                           'incomplete_proposals':incomplete,'prefix_tokens':len(ids),'stratum':row['stratum']})
            if n%10==0: marker('BENCH_PROPOSAL_PROGRESS',{'arm':arm,'positions':n+1,'last_seconds':elapsed})
    write(OUT/('benchmark-proposal-'+arm+'.json'),{'arm':arm,'rows':result,'load_seconds':load_seconds,
        'gpu':torch.cuda.get_device_name(),'peak_memory_bytes':torch.cuda.max_memory_allocated(),
        'backend':'HF eager BF16 canonical head; top10, at most4 tokens/proposal','scientific_data':False})


def merge():
    fixtures=json.loads((OUT/'benchmark-fixtures.json').read_text())
    sources={a:{r['key']:r for r in json.loads((OUT/('benchmark-proposal-'+a+'.json')).read_text())['rows']} for a in ARMS}
    for row in fixtures['positions']:
        union=set()
        for a in ARMS: union.update(sources[a][row['key']]['candidates'])
        if row['reference_word']: union.add(row['reference_word'])
        row['candidates']=sorted(union)
    write(OUT/'benchmark-candidates.json',fixtures)
    marker('BENCH_CANDIDATES',{'positions':N,'mean_candidates':statistics.mean(len(r['candidates']) for r in fixtures['positions']),
                              'min':min(len(r['candidates']) for r in fixtures['positions']),
                              'max':max(len(r['candidates']) for r in fixtures['positions'])})


def scoring(arm):
    import torch
    from hf_frozen_sampler import terminal_token_ids
    fixtures=json.loads((OUT/'benchmark-candidates.json').read_text())
    tick=sync_time(); model,tok=load_hf(arm); load_seconds=sync_time()-tick
    vocab=vocabulary_bytes(tok,'gemma' if arm in ('S','G9') else 'bpe')
    stop=set(terminal_token_ids(tok))
    boundary_flags=[False]*model.get_output_embeddings().weight.shape[0]
    for i,piece in enumerate(vocab):
        first=(piece or b'').decode('utf-8',errors='ignore')[:1]
        boundary_flags[i]=i in stop or bool(first and (first.isspace() or (not first.isalnum() and first!='_')))
    boundary=torch.tensor(boundary_flags,device='cuda',dtype=torch.bool)
    result=[]
    with torch.inference_mode():
        for n,row in enumerate(fixtures['positions']):
            start=sync_time(); ids=row['prefix_ids'][arm]
            base=model.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=True)
            prefix_h=base.last_hidden_state[0,-1:]
            prefill=sync_time()-start
            prefix_text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            valid=0; rejected=0; suffix_tokens=0; raw_mass=0; calibrated_mass=0
            suffix_time_start=sync_time()
            for candidate in row['candidates']:
                full=tok.encode(prefix_text+candidate,add_special_tokens=False)
                if full[:len(ids)]!=ids or len(full)==len(ids): rejected+=1; continue
                suffix=full[len(ids):]
                if tok.decode(full,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=prefix_text+candidate:
                    rejected+=1; continue
                cache=copy.deepcopy(base.past_key_values)
                ext=model.model(input_ids=torch.tensor([suffix],device='cuda'),past_key_values=cache,use_cache=True)
                # Include one extra predictor for the post-word boundary factor.
                h=torch.cat([prefix_h,ext.last_hidden_state[0]],dim=0)
                logits=canonical_head(model,h); labels=torch.tensor(suffix,device='cuda')
                masses=[]
                for tau in (1.,1.25):
                    lp=(logits/tau).log_softmax(-1)
                    path_lp=lp[:-1].gather(1,labels[:,None]).sum()
                    boundary_p=lp[-1,boundary].exp().sum()
                    masses.append(float(path_lp.exp()*boundary_p))
                raw_mass+=masses[0]; calibrated_mass+=masses[1]
                suffix_tokens+=len(suffix); valid+=1
                del ext,cache
            suffix_seconds=sync_time()-suffix_time_start
            elapsed=sync_time()-start; del base
            result.append({'key':row['key'],'stratum':row['stratum'],'sampling_weight':row['sampling_weight'],
                'prefix_tokens':len(ids),'candidate_count':len(row['candidates']),'valid_candidates':valid,
                'canonical_prefix_rejections':rejected,'candidate_suffix_tokens':suffix_tokens,
                'prefill_seconds':prefill,'candidate_seconds':suffix_seconds,'wall_seconds':elapsed,
                'raw_canonical_path_mass':raw_mass,'calibrated_canonical_path_mass':calibrated_mass})
            if n%10==0: marker('BENCH_SCORING_PROGRESS',{'arm':arm,'positions':n+1,'last_seconds':elapsed,'candidates':valid})
    write(OUT/('benchmark-scoring-'+arm+'.json'),{'arm':arm,'rows':result,'load_seconds':load_seconds,
        'gpu':torch.cuda.get_device_name(),'peak_memory_bytes':torch.cuda.max_memory_allocated(),
        'backend':'HF eager BF16; independent KV clones; raw + representative calibrated temperature; full-vocab boundary softmax',
        'scientific_data':False,'limitations':'literal guide boundary predicate; unresolved word-event math; no fitted global temperatures'})


def weighted_mean(rows,key):
    weights=[r.get('sampling_weight',1.) for r in rows]
    return sum(r[key]*w for r,w in zip(rows,weights))/sum(weights)


def report():
    fixtures=json.loads((OUT/'benchmark-candidates.json').read_text())
    weights={r['key']:r['sampling_weight'] for r in fixtures['positions']}
    measured={}; raw_hours={}; prices={'S':1.179648,'G9':2.395584,'Q':2.395584}
    for a in ARMS:
        p=json.loads((OUT/('benchmark-proposal-'+a+'.json')).read_text())
        s=json.loads((OUT/('benchmark-scoring-'+a+'.json')).read_text())
        for r in p['rows']: r['sampling_weight']=weights[r['key']]
        if len(p['rows'])!=N or len(s['rows'])!=N: raise RuntimeError('Incomplete 100-position benchmark')
        # First5 positions are retained as warmup telemetry, excluded from rate estimates.
        pr=p['rows'][5:]; sr=s['rows'][5:]
        seconds=weighted_mean(pr,'wall_seconds')+weighted_mean(sr,'wall_seconds')
        total_fixed=p['load_seconds']+s['load_seconds']
        raw_hours[a]=(8000*seconds+total_fixed)/3600
        measured[a]={'gpu':s['gpu'],'positions':N,'warmup_positions_excluded':5,
            'mean_proposal_seconds':weighted_mean(pr,'wall_seconds'),'mean_scoring_seconds':weighted_mean(sr,'wall_seconds'),
            'mean_scoring_prefill_seconds':weighted_mean(sr,'prefill_seconds'),
            'mean_scoring_candidate_seconds':weighted_mean(sr,'candidate_seconds'),
            'mean_candidates':weighted_mean(sr,'candidate_count'),'mean_valid_candidates':weighted_mean(sr,'valid_candidates'),
            'prefix_tokens_min':min(r['prefix_tokens'] for r in sr),'prefix_tokens_max':max(r['prefix_tokens'] for r in sr),
            'candidate_rejections':sum(r['canonical_prefix_rejections'] for r in s['rows']),
            'peak_memory_gib':s['peak_memory_bytes']/2**30,'load_seconds_both_stages':total_fixed,
            'projected_8k_hours':raw_hours[a],'projected_8k_cost_usd':raw_hours[a]*prices[a],
            'candidate_seconds_per_attempted_candidate':weighted_mean(sr,'candidate_seconds')/weighted_mean(sr,'candidate_count')}
        valid=weighted_mean(sr,'valid_candidates')
        if not valid: raise RuntimeError('No valid candidates; cannot extrapolate scoring cost')
        measured[a]['candidate_seconds_per_valid_candidate']=weighted_mean(sr,'candidate_seconds')/valid
    cost=sum(raw_hours[a]*prices[a] for a in ARMS)
    all_candidate_hours={a:(8000*(measured[a]['mean_proposal_seconds']+
        measured[a]['mean_scoring_prefill_seconds']+
        measured[a]['candidate_seconds_per_valid_candidate']*measured[a]['mean_candidates'])+
        measured[a]['load_seconds_both_stages'])/3600 for a in ARMS}
    summary={'run_id':str(OUT.name),'fixture_source':fixtures['fixture_type'],'scientific_data':False,
        'measured':measured,'candidate_union_models':list(ARMS),'teacher_scope':['G9','Q'],
        'projection_8k_three_models':{'gpu_hours':sum(raw_hours.values()),'cost_usd':cost,'cost_with_20pct_reserve_usd':cost*1.2},
        'projection_8k_three_models_if_all_candidates_scored':{'gpu_hours':sum(all_candidate_hours.values()),
            'cost_with_20pct_reserve_usd':sum(all_candidate_hours[a]*prices[a] for a in ARMS)*1.2},
        'price_assumption_usd_hour':prices,'excluded':['rollout generation','full-pool teacher scoring','global calibration fit',
            'scientific parity repairs','Modal hydration startup and output IO beyond measured worker times'],
        'limitations':['reference-prefix proxy; student rollout prefix/candidate distributions may differ',
            'sequential candidate implementation; unqualified batching speedups not assumed',
            'linear extrapolation with sampled stratum weights and20pct reserve, not a guaranteed upper bound',
            'observed baseline includes skipped canonical-prefix failures; fixing their event paths may increase cost',
            'performance data must not be used as H2 evidence']}
    write(OUT/'benchmark-summary.json',summary); marker('BENCH_SUMMARY',summary)


if __name__=='__main__':
    stage,arm=sys.argv[1:3]
    {'bench-prepare':prepare,'bench-proposal':lambda:proposal(arm),'bench-merge':merge,
     'bench-scoring':lambda:scoring(arm),'bench-report':report}[stage]()

"""Small HF-eager H2 pilot and exhaustive byte-segmentation audit.

The numerical word distributions are canonical-path proxies, never exhaustive
text marginals. The original SGLang study and failed gates are left untouched.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

from h2_pilot import OUT, ASSETS, render, canonical_head, write, marker, file_sha
from h2_word_events import next_word, needs_boundary, boundary_flags, byte_event_map, is_ascii_atom
from h2_measurements import vocabulary_bytes, byte_spans, strict_pairs
from h2_tokenizer_audit import audit_pair, aggregate_audits
from h2_word_metrics import compare_distributions, aggregate, distribution, js

CFG = json.loads(Path(__file__).with_name('h2_word_pilot_config.json').read_text())
ARMS = ('S', 'B', 'Q')


def load_hf(arm):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    entry=read('assets.json')['models'][arm]
    if not entry.get('verified_weights'): raise RuntimeError('Pinned weights not verified')
    dtype=torch.float32 if CFG['backbone_dtype']=='float32' else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(entry['path'],local_files_only=True,
        dtype=dtype,attn_implementation='eager',device_map={'':'cuda'}).eval()
    model.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(entry['path'],local_files_only=True)
    return model,tok


def stop_ids(model,tok):
    from hf_frozen_sampler import terminal_token_ids
    stop=set(terminal_token_ids(tok))
    for config in (model.config,model.generation_config):
        eos=getattr(config,'eos_token_id',None)
        stop.update(eos if isinstance(eos,(list,tuple)) else [eos] if eos is not None else [])
    added=tok.get_added_vocab()
    stop.update(added[name] for name in ('<|eot_id|>','<|eom_id|>','<|end_of_text|>') if name in added)
    return stop


def read(name):
    return json.loads((OUT / name).read_text())


def tokenizer_stats(tok, family):
    vocab = vocabulary_bytes(tok, family)
    values = {v for v in vocab if v}
    return vocab, values


def lexical_overlap(student, teacher):
    shared = student & teacher
    return {'student_unique_byte_strings':len(student), 'teacher_unique_byte_strings':len(teacher),
            'shared_unique_byte_strings':len(shared), 'student_vocabulary_coverage':len(shared)/len(student),
            'teacher_vocabulary_coverage':len(shared)/len(teacher),
            'jaccard':len(shared)/len(student|teacher),
            'denominator':'unique decoded non-special byte strings, not token occurrence frequency'}


def setup(download_arm=None):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download, get_token
    from transformers import AutoTokenizer
    token = get_token()
    if not token: raise RuntimeError('Existing HF secret has no readable token')
    api = HfApi(token=token)
    manifest = read('assets.json') if (OUT/'assets.json').exists() else {'models':{},'access_errors':{}}
    for arm, spec in CFG['models'].items():
        try:
            prior = manifest['models'].get(arm, {})
            revision = prior.get('revision') if prior.get('repo')==spec['repo'] else None
            revision = revision or api.model_info(spec['repo']).sha
            # A real protected file download, rather than public metadata alone.
            hf_hub_download(spec['repo'], 'config.json', revision=revision, token=token)
            location = ASSETS/'word-pilot-models'/spec['repo'].replace('/','--')/revision
            path = snapshot_download(spec['repo'], revision=revision, token=token, local_dir=location,
                allow_patterns=['*.json','*.jinja','*.model','tokenizer*'])
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            entry = dict(spec, revision=revision, path=str(path),
                tokenizer_sha256=hashlib.sha256(json.dumps(tok.get_vocab(),sort_keys=True).encode()).hexdigest(),
                tokenizer_files_sha256={p.name:file_sha(p) for p in location.iterdir() if p.is_file()},
                verified_weights=prior.get('verified_weights',{}) if prior.get('repo')==spec['repo'] else {})
            if spec.get('official_reference'):
                mirror=api.model_info(spec['repo'],revision=revision,files_metadata=True)
                official=api.model_info(spec['official_reference'],files_metadata=True)
                def weight_hashes(info):
                    return {f.rfilename:getattr(f.lfs,'sha256',None) for f in info.siblings if f.rfilename.endswith('.safetensors')}
                left,right=weight_hashes(mirror),weight_hashes(official)
                official_files={f.rfilename:getattr(f,'blob_id',None) for f in official.siblings}
                small_files={}
                for name in ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','generation_config.json'):
                    local=location/name
                    if not local.exists(): continue
                    data=local.read_bytes(); blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                    small_files[name]={'mirror_git_blob':blob,'official_git_blob':official_files.get(name),
                        'identical':blob==official_files[name] if official_files.get(name) else None}
                entry['official_comparison']={'repo':spec['official_reference'],'revision':official.sha,
                    'mirror_lfs_sha256':left,'official_lfs_sha256':right,
                    'identical_weight_shards_by_name_and_hash':bool(left) and all(left.values()) and left==right,
                    'small_file_comparisons':small_files,
                    'scope':'public LFS metadata; downloaded mirror shards separately verified on disk'}
            if arm == download_arm and not entry['verified_weights']:
                info = api.model_info(spec['repo'],revision=revision,files_metadata=True)
                snapshot_download(spec['repo'],revision=revision,token=token,local_dir=location,allow_patterns=['*.safetensors'])
                for f in info.siblings:
                    if not f.rfilename.endswith('.safetensors'): continue
                    local=location/f.rfilename; digest=file_sha(local)
                    expected=getattr(f.lfs,'sha256',None)
                    if expected and digest!=expected: raise RuntimeError('Weight hash mismatch')
                    entry['verified_weights'][f.rfilename]={'sha256':digest,'size':local.stat().st_size}
            manifest['models'][arm]=entry; manifest['access_errors'].pop(arm,None)
        except Exception as exc:
            manifest['access_errors'][arm]={'type':type(exc).__name__,
                'status_code':getattr(getattr(exc,'response',None),'status_code',None)}
        write(OUT/'assets.json',manifest)
    write(OUT/'word-config.json',CFG)
    marker('WORD_ACCESS',{'models':{a:{'repo':e['repo'],'revision':e['revision'],
        'tokenizer_ready':True,'weights_ready':bool(e['verified_weights'])} for a,e in manifest['models'].items()},
        'access_errors':manifest['access_errors'],'download_arm':download_arm})
    if download_arm and (download_arm not in manifest['models'] or not manifest['models'][download_arm]['verified_weights']):
        raise RuntimeError('Requested pinned model is unavailable')


def tokenizers():
    from transformers import AutoTokenizer
    assets=read('assets.json')
    if any(a not in assets['models'] for a in ARMS): raise RuntimeError('Tokenizer access incomplete')
    return {a:AutoTokenizer.from_pretrained(assets['models'][a]['path'],local_files_only=True) for a in ARMS}


def prompts_prepare():
    toks=tokenizers(); source=OUT.parent/CFG['source_run']
    old=json.loads((source/'prompts.json').read_text())
    pool=[]; excluded=[]
    for kind,count in CFG['dataset_mix'].items():
        eligible=[]
        for row in old['pool']:
            if row['source']!=kind: continue
            lengths={a:len(render(toks[a],row['prompt'])) for a in ARMS}
            if max(lengths.values())+CFG['max_new_tokens']>CFG['context_limit']:
                excluded.append({'prompt_id':row['id'],'lengths':lengths}); continue
            eligible.append(row)
        ranked=sorted(eligible,key=lambda r:hashlib.sha256(f"42/{r['id']}".encode()).hexdigest())
        if len(ranked)<count: raise RuntimeError('Insufficient source prompts')
        pool.extend(ranked[:count])
    write(OUT/'word-prompts.json',{'pool':pool,'datasets':old['datasets'],
        'source_manifest_sha256':file_sha(source/'prompts.json'),'context_exclusions':excluded,
        'template_policy':CFG['template_policy'],
        'template_examples':{a:toks[a].decode(render(toks[a],pool[0]['prompt']),skip_special_tokens=False) for a in ARMS}})
    stats={a:tokenizer_stats(toks[a],CFG['models'][a]['family']) for a in ARMS}
    write(OUT/'vocabulary-overlap.json',{'models':read('assets.json')['models'],
        'pairs':{a:lexical_overlap(stats['S'][1],stats[a][1]) for a in ('B','Q')},
        'warning':'Vocabulary overlap does not imply the same segmentation on occurrences'})
    audit_corpus([dict(r,prompt_id=r['id'],id=r['id']+'-reference',text=r['gold']) for r in pool],'reference')
    marker('WORD_PROMPTS',{'prompts':len(pool),'mix':CFG['dataset_mix']})


def encode_response(tok,vocab,prompt,text,content=None):
    prefix=render(tok,prompt); prefix_text=tok.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    full=prefix+content if content is not None else tok.encode(prefix_text+text,add_special_tokens=False)
    if full[:len(prefix)]!=prefix: raise ValueError('Native template boundary changed')
    if tok.decode(full,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=prefix_text+text:
        raise ValueError('Contextual decoding roundtrip failed')
    return full,len(prefix),byte_spans(full[len(prefix):],vocab,text)


def audit_corpus(rows,label):
    from transformers import AutoTokenizer
    toks=tokenizers(); stats={a:tokenizer_stats(toks[a],CFG['models'][a]['family']) for a in ARMS}
    old=json.loads((OUT.parent/CFG['source_run']/'assets.json').read_text())['models']
    ot={a:AutoTokenizer.from_pretrained(old[k]['path'],local_files_only=True) for a,k in CFG['historical_tokenizer_controls'].items()}
    ov={a:vocabulary_bytes(t,'gemma' if a in ('S','B') else 'bpe') for a,t in ot.items()}
    old_sets={a:{piece for piece in values if piece} for a,values in ov.items()}
    groups={}; failures=[]; index=[]
    same_native_vocabulary=read('assets.json')['models']['S']['tokenizer_sha256']==read('assets.json')['models']['B']['tokenizer_sha256']
    trace_dir=OUT/'tokenizer-audit'/label; trace_dir.mkdir(parents=True,exist_ok=True)
    for row in rows:
        spans={}; old_spans={}; errors={}; encoded={}
        for a in ARMS:
            try:
                full,prefix,spans[a]=encode_response(toks[a],stats[a][0],row['prompt'],row['text'],
                    row.get('content_ids') if label=='rollout' and (a=='S' or a=='B' and same_native_vocabulary) else None)
                encoded[a]={'full_ids':full,'prefix_size':prefix,'spans':spans[a]}
            except ValueError as exc: errors['new-'+a]=str(exc)
            try: _,_,old_spans[a]=encode_response(ot[a],ov[a],row['prompt'],row['text'])
            except ValueError as exc: errors['old-'+a]=str(exc)
        payload={'prompt_id':row['prompt_id'],'rollout_id':row['id'],'text':row['text'],
            'source':row['source'],'errors':errors,'pairs':{},'encoded':encoded,
            'generation_metadata':{k:row.get(k) for k in ('response_index','finish_reason','sampling_seed','batch_offset')},
            'excluded_terminal_token_ids':row.get('output_ids',[])[len(row.get('content_ids',[])):] if label=='rollout' else [],
            'template_policy':CFG['template_policy']}
        canonical_spans={}
        if label=='rollout':
            for a in ARMS:
                try: _,_,canonical_spans[a]=encode_response(toks[a],stats[a][0],row['prompt'],row['text'])
                except ValueError as exc: errors['canonical-'+a]=str(exc)
            if 'S' in spans and 'S' in canonical_spans:
                payload['native_vs_canonical_student']=audit_pair(spans['S'],canonical_spans['S'],row['text'])
                payload['student_path_is_canonical']=spans['S']==canonical_spans['S']
            for a in ('B','Q'):
                if 'S' in canonical_spans and a in canonical_spans:
                    name='canonical-'+a
                    result=audit_pair(canonical_spans['S'],canonical_spans[a],row['text'])
                    payload['pairs'][name]=result
                    groups.setdefault(name,[]).append({'prompt_id':row['prompt_id'],'rollout_id':row['id'],'audit':result})
        payload['baseline_content_policy']='student-native IDs after asserting identical vocabulary hash' if label=='rollout' and same_native_vocabulary else 'independent canonical encoding'
        for family,pieces in [('new',spans),('historical',old_spans)]:
            for a in ('B','Q'):
                if 'S' not in pieces or a not in pieces: continue
                name=family+'-'+a; audit=audit_pair(pieces['S'],pieces[a],row['text'])
                sets={k:stats[k][1] for k in ARMS} if family=='new' else old_sets
                source=row['text'].encode()
                s_available=[source[x:y] in sets[a] for x,y,_ in pieces['S']]
                t_available=[source[x:y] in sets['S'] for x,y,_ in pieces[a]]
                s_bytes=sum(y-x for (x,y,_),ok in zip(pieces['S'],s_available) if ok)
                t_bytes=sum(y-x for (x,y,_),ok in zip(pieces[a],t_available) if ok)
                audit['counts'].update(student_tokens_available_teacher_vocab=sum(s_available),
                    teacher_tokens_available_student_vocab=sum(t_available),
                    student_bytes_available_teacher_vocab=s_bytes,teacher_bytes_available_student_vocab=t_bytes)
                eligible=[t for t in audit['student_token_trace'] if t['student_index']>=5 and
                    t['category']!='whitespace' and t['utf8_start_boundary']]
                matched=[t for t in eligible if t['strict']]
                audit['counts'].update(eligible_student_tokens=len(eligible),strict_eligible_student_tokens=len(matched))
                audit['h2_eligibility']={'eligible_student_tokens':len(eligible),
                    'strict_eligible_student_tokens':len(matched),
                    'strict_eligible_student_fraction':len(matched)/len(eligible) if eligible else None,
                    'excluded_first_response_tokens':5,'whitespace_only_excluded':True,
                    'incomplete_utf8_prefix_excluded':True}
                audit['lexical_occurrence_coverage']={'student_fraction':sum(s_available)/len(s_available) if s_available else None,
                    'teacher_fraction':sum(t_available)/len(t_available) if t_available else None,
                    'student_byte_fraction':s_bytes/len(source) if source else None,
                    'teacher_byte_fraction':t_bytes/len(source) if source else None,
                    'contract':'Observed decoded token byte string exists in the other vocabulary, independently of chosen segmentation'}
                for trace,available in zip(audit['student_token_trace'],s_available):
                    trace['byte_string_available_in_teacher_vocab']=available
                payload['pairs'][name]=audit
                groups.setdefault(name,[]).append({'prompt_id':row['prompt_id'],'rollout_id':row['id'],'audit':audit})
        path=trace_dir/(row['id']+'.json'); write(path,payload)
        index.append({'prompt_id':row['prompt_id'],'rollout_id':row['id'],'path':str(path),'errors':errors})
        if errors: failures.append({'rollout_id':row['id'],'errors':errors})
    paired={}
    for arm in ('B','Q'):
        by_family={family:{r['rollout_id']:r for r in groups.get(family+'-'+arm,[])} for family in ('new','historical')}
        common=set(by_family['new'])&set(by_family['historical'])
        paired[arm]={'common_response_ids':sorted(common),'common_count':len(common),
            'omitted_from_paired_count':len(rows)-len(common),
            'new':aggregate_audits([by_family['new'][k] for k in sorted(common)]),
            'historical':aggregate_audits([by_family['historical'][k] for k in sorted(common)])}
        if label=='rollout':
            canonical={r['rollout_id']:r for r in groups.get('canonical-'+arm,[])}
            common_canonical=set(canonical)&set(by_family['historical'])
            paired[arm]['canonical_control']={'common_response_ids':sorted(common_canonical),
                'common_count':len(common_canonical),'omitted_count':len(rows)-len(common_canonical),
                'new':aggregate_audits([canonical[k] for k in sorted(common_canonical)]),
                'historical':aggregate_audits([by_family['historical'][k] for k in sorted(common_canonical)]),
                'contract':'Matched-text canonical-vs-canonical comparison, excluding stochastic token-path effects'}
    summary={'corpus':label,'rows':len(rows),'failures':failures,
        'paired_comparison':paired,
        'paired_corpus_policy':'Use paired_comparison for old/new comparisons: common successful response IDs and identical text. Independent summaries retain all successes and explicit failures.',
        'pairs':{name:aggregate_audits(records) for name,records in groups.items()},'index':index,
        'claim_90_percent':'No claim source supplied. Compare only an explicit numerator/denominator on this corpus, not truthfulness of an unknown report.'}
    write(OUT/('tokenizer-audit-'+label+'.json'),summary)
    marker('TOKENIZER_AUDIT',{'corpus':label,'rows':len(rows),'failures':len(failures),'pairs':list(groups)})


def qualify(arm):
    import torch
    from hf_frozen_sampler import terminal_token_ids
    model,tok=load_hf(arm); tok.padding_side='left'
    probes=['2+2=4.\n','A fraction: \\frac{1}{2}. 😀']
    prefixes=[render(tok,text) for text in probes]
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    width=max(map(len,prefixes))
    ids=torch.tensor([[pad]*(width-len(p))+p for p in prefixes],device='cuda')
    mask=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prefixes],device='cuda')
    diffs=[]; heads=[]; cache_errors=[]; word_self_js=[]
    vocab=vocabulary_bytes(tok,CFG['models'][arm]['family'])
    stop=stop_ids(model,tok); head_size=model.get_output_embeddings().weight.shape[0]
    masks={kind:torch.tensor(boundary_flags(vocab,head_size,stop,kind=='latex'),device='cuda',dtype=torch.bool)
           for kind in ('letter','latex')}
    with torch.inference_mode():
        torch.manual_seed(42)
        generated=model.generate(input_ids=ids,attention_mask=mask,do_sample=True,temperature=1.,top_p=1.,top_k=0,
            repetition_penalty=1.,max_new_tokens=2,pad_token_id=pad,eos_token_id=sorted(stop),
            return_dict_in_generate=True,output_scores=True,use_cache=True)
        for k,p in enumerate(prefixes):
            x=torch.tensor([p],device='cuda')
            full=model(input_ids=x,output_hidden_states=True,use_cache=False)
            all_head=canonical_head(model,full.hidden_states[-1])
            h=all_head[0,-1]
            heads.append(float((all_head-full.logits).abs().max()))
            diffs.extend((h.log_softmax(-1)-generated.scores[0][k].float().log_softmax(-1)).abs().cpu().tolist())
            prefix=model.model(input_ids=x,use_cache=True)
            tail=torch.tensor([[int(generated.sequences[k,width])]],device='cuda')
            cached=model.model(input_ids=tail,past_key_values=prefix.past_key_values,use_cache=True)
            direct=model.model(input_ids=torch.cat([x,tail],1),use_cache=False)
            a=canonical_head(model,cached.last_hidden_state[0,-1:])[0].log_softmax(-1)
            b=canonical_head(model,direct.last_hidden_state[0,-1:])[0].log_softmax(-1)
            cache_errors.extend((a-b).abs().cpu().tolist())
            prefix_text=tok.decode(p,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            reference=[]; reconstructed=[]
            for word in (' the',' then','1','.',r'\frac'):
                encoded=tok.encode(prefix_text+word,add_special_tokens=False)
                if encoded[:len(p)]!=p: continue
                suffix=encoded[len(p):]
                if not suffix: continue
                base=model.model(input_ids=x,use_cache=True)
                ext=model.model(input_ids=torch.tensor([suffix],device='cuda'),past_key_values=base.past_key_values,use_cache=True)
                direct=model.model(input_ids=torch.tensor([encoded],device='cuda'),use_cache=False)
                cached_h=torch.cat([base.last_hidden_state[0,-1:],ext.last_hidden_state[0]],0)
                direct_h=direct.last_hidden_state[0,len(p)-1:]
                for hidden,bucket in ((cached_h,reconstructed),(direct_h,reference)):
                    lp=canonical_head(model,hidden).log_softmax(-1)
                    labels=torch.tensor(suffix,device='cuda')
                    mass=lp[:-1].gather(1,labels[:,None]).sum().exp()
                    if needs_boundary(word): mass*=lp[-1,masks['latex' if word.lstrip().startswith('\\') else 'letter']].exp().sum()
                    bucket.append(float(mass))
                del base,ext,direct
            if len(reference)<2: raise RuntimeError('Insufficient native word self-check paths')
            word_self_js.append(js(distribution(reference),distribution(reconstructed)))
    result={'arm':arm,'backend':'HF eager '+CFG['backbone_dtype'],'canonical_head_max_abs':max(heads),
        'generation_mean_abs_logprob':sum(diffs)/len(diffs),'generation_max_abs_logprob':max(diffs),
        'cached_mean_abs_logprob':sum(cache_errors)/len(cache_errors),'cached_max_abs_logprob':max(cache_errors),
        'same_model_cached_vs_uncached_word_js':word_self_js,
        'limits':{'head':1e-5,'mean_logprob':.02,'max_logprob':.1},
        'model_revision':read('assets.json')['models'][arm]['revision'],
        'stop_token_ids':sorted(stop), 'config_sha256':file_sha(Path(__file__).with_name('h2_word_pilot_config.json')),
        'versions':{'torch':torch.__version__,'cuda':torch.version.cuda},'gpu':torch.cuda.get_device_name()}
    result['pass']=max(heads)<=1e-5 and max(diffs)<=.1 and sum(diffs)/len(diffs)<=.02 and max(cache_errors)<=.1 and sum(cache_errors)/len(cache_errors)<=.02 and max(word_self_js)<=1e-5
    write(OUT/('word-qualification-'+arm+'-a'+os.environ.get('H2_STAGE_ATTEMPT','1')+'.json'),result); write(OUT/('word-qualification-'+arm+'.json'),result); marker('WORD_QUALIFICATION',result)
    if not result['pass']: raise RuntimeError('HF generation/cache qualification failed')


def require_qualified(arm):
    result=read('word-qualification-'+arm+'.json')
    if not result['pass'] or result['model_revision']!=read('assets.json')['models'][arm]['revision'] or result.get('config_sha256')!=file_sha(Path(__file__).with_name('h2_word_pilot_config.json')):
        raise RuntimeError('HF eager qualification missing or stale')


def rollout():
    import torch
    from hf_frozen_sampler import terminal_token_ids
    require_qualified('S'); model,tok=load_hf('S')
    pool=read('word-prompts.json')['pool']; tasks=[(p,j) for p in pool for j in range(CFG['samples_per_prompt'])]
    path=OUT/'word-rollouts.json'; rows=read(path.name) if path.exists() else []
    expected=[p['id']+'-rollout-'+str(j) for p,j in tasks]
    if [r['id'] for r in rows]!=expected[:len(rows)]: raise RuntimeError('Resume manifest differs')
    stop=stop_ids(model,tok); pad=tok.pad_token_id if tok.pad_token_id is not None else next(iter(stop))
    batch_size=CFG['generation_batch_size']; began=time.monotonic()
    for off in range(len(rows),len(tasks),batch_size):
        batch=tasks[off:off+batch_size]; prefixes=[render(tok,p['prompt']) for p,j in batch]; width=max(map(len,prefixes))
        ids=torch.tensor([[pad]*(width-len(p))+p for p in prefixes],device='cuda')
        mask=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prefixes],device='cuda')
        seed=CFG['seed']+off; torch.manual_seed(seed)
        with torch.inference_mode():
            outputs=model.generate(input_ids=ids,attention_mask=mask,do_sample=True,temperature=1.,top_p=1.,top_k=0,
                repetition_penalty=1.,max_new_tokens=CFG['max_new_tokens'],pad_token_id=pad,eos_token_id=sorted(stop),use_cache=True)
        for (p,j),prefix,out in zip(batch,prefixes,outputs[:,width:].tolist()):
            end=next((i for i,v in enumerate(out) if v in stop),len(out)); content=out[:end]
            prefix_text=tok.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            text=tok.decode(prefix+content,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if not text.startswith(prefix_text): raise RuntimeError('Sampled contextual decode changed prefix')
            rows.append(dict(p,id=p['id']+'-rollout-'+str(j),prompt_id=p['id'],response_index=j,
                prefix_ids=prefix,content_ids=content,output_ids=out[:end+1] if end<len(out) else content,
                text=text[len(prefix_text):],finish_reason='terminal' if end<len(out) else 'max_new_tokens',
                sampling_seed=seed,batch_offset=off,backend='HF eager '+CFG['backbone_dtype']))
        write(path,rows); marker('WORD_ROLLOUT_PROGRESS',{'responses':len(rows),'total':len(tasks),'seconds':time.monotonic()-began})
    write(OUT/'word-rollout-summary.json',{'responses':len(rows),'prompts':len(pool),
        'generated_tokens':sum(len(r['content_ids']) for r in rows),'cap_hits':sum(r['finish_reason']=='max_new_tokens' for r in rows),
        'sampling':{k:CFG[k] for k in ('seed','temperature','top_p','top_k','max_new_tokens','generation_batch_size')},
        'rng_policy':'CUDA RNG fixed per stable batch offset; fixed model/runtime/batch ordering',
        'template_policy':CFG['template_policy']})


def audit_rollouts():
    rows=read('word-rollouts.json')
    if len(rows)!=CFG['prompts']*CFG['samples_per_prompt']: raise RuntimeError('Incomplete rollout pool')
    audit_corpus(rows,'rollout')


def fixtures():
    audit=read('tokenizer-audit-rollout.json'); groups=defaultdict(list); counts=Counter(); failures=[]; encodings={}
    for item in audit['index']:
        payload=json.loads(Path(item['path']).read_text()); enc=payload['encoded']
        if any(a not in enc for a in ARMS): failures.append(item['rollout_id']); continue
        pairs={a:strict_pairs(enc['S']['spans'],enc[a]['spans']) for a in ARMS}
        encodings[item['rollout_id']]=(enc,pairs,payload['text'])
        keep=set.intersection(*(set(m) for m in pairs.values()))
        trace=payload['pairs']['new-B']['student_token_trace']; raw=payload['text'].encode()
        for j in sorted(keep):
            left,right,_=enc['S']['spans'][j]
            if j<5 or raw[left:right].decode('utf-8',errors='ignore').isspace(): continue
            try: continuation=raw[left:].decode('utf-8'); raw[:left].decode('utf-8')
            except UnicodeDecodeError: continue
            word=next_word(continuation,terminated=True)
            if not word: continue
            cat=trace[j]['category']; stratum=item['prompt_id']+'/'+cat
            row={'key':item['rollout_id']+'/'+str(j),'prompt_id':item['prompt_id'],'response_id':item['rollout_id'],
                'position':j,'byte_start':left,'byte_end':right,'category':cat,'stratum':stratum,'reference_word':word,
                'rank':hashlib.sha256(f"42/{item['rollout_id']}/{j}".encode()).hexdigest()}
            groups[stratum].append(row); counts[stratum]+=1
    for values in groups.values(): values.sort(key=lambda r:r['rank'])
    selected=[]; keys=sorted(groups)
    while len(selected)<CFG['e3_positions']:
        changed=False
        for k in keys:
            if groups[k] and len(selected)<CFG['e3_positions']: selected.append(groups[k].pop(0)); changed=True
        if not changed: raise RuntimeError('Insufficient eligible aligned positions; do not duplicate samples')
    sample=Counter(r['stratum'] for r in selected)
    for r in selected:
        r['sampling_weight']=counts[r['stratum']]/sample[r['stratum']]
        enc,pairs,text=encodings[r['response_id']]; j=r['position']
        r['prefix_ids']={a:enc[a]['full_ids'][:enc[a]['prefix_size']+pairs[a][j]] for a in ARMS}
        r['realized_token_ids']={a:enc[a]['spans'][pairs[a][j]][2] for a in ARMS}
        r['prefix_text']=text.encode()[:r['byte_start']].decode('utf-8')
    write(OUT/'word-fixtures.json',{'positions':selected,'population_counts':dict(counts),'sample_counts':dict(sample),
        'sampling':'seed42 hash, round-robin prompt/category strata; weights N_stratum/n_stratum',
        'failed_responses':failures,'eligible_positions':sum(counts.values())})
    marker('WORD_FIXTURES',{'positions':len(selected),'unique_prompts':len({r['prompt_id'] for r in selected}),
        'categories':dict(Counter(r['category'] for r in selected)),'eligible_positions':sum(counts.values())})


def proposal(arm):
    import torch
    from hf_frozen_sampler import terminal_token_ids
    require_qualified(arm); model,tok=load_hf(arm); stop=stop_ids(model,tok); special=set(tok.all_special_ids)
    vocab=vocabulary_bytes(tok,CFG['models'][arm]['family'])
    rows=read('word-fixtures.json')['positions']; path=OUT/('word-proposals-'+arm+'.json')
    result=read(path.name)['rows'] if path.exists() else []; done={r['key'] for r in result}; began=time.monotonic()
    with torch.inference_mode():
        for n,row in enumerate(rows):
            if row['key'] in done: continue
            tick=time.monotonic(); ids=row['prefix_ids'][arm]
            base=model.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=True)
            logits=canonical_head(model,base.last_hidden_state[0,-1:])[0]
            top=[i for i in logits.topk(32).indices.tolist() if i<len(tok) and i not in special][:CFG['proposal_top_k']]
            prefix=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False); candidates=[]; incomplete=0
            for tid in top:
                tail=[tid]; cache=copy.deepcopy(base.past_key_values)
                for depth in range(CFG['proposal_max_tokens']):
                    ended=tail[-1] in stop
                    whole=tok.decode(ids+(tail[:-1] if ended else tail),skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    if not whole.startswith(prefix): raise RuntimeError('Proposal changed decoded prefix')
                    lexical_tail=tail[:-1] if ended else tail
                    if any(vocab[i] is None for i in lexical_tail): candidate=None; ended=True
                    else:
                        try: tail_text=b''.join(vocab[i] for i in lexical_tail).decode('utf-8'); candidate=next_word(tail_text,terminated=ended)
                        except UnicodeDecodeError: candidate=None
                    if candidate or ended or depth==CFG['proposal_max_tokens']-1: break
                    ext=model.model(input_ids=torch.tensor([[tail[-1]]],device='cuda'),past_key_values=cache,use_cache=True)
                    cache=ext.past_key_values; tail.append(int(canonical_head(model,ext.last_hidden_state[0,-1:]).argmax(-1))); del ext
                if candidate: candidates.append(candidate)
                else: incomplete+=1
                del cache
            torch.cuda.synchronize(); result.append({'key':row['key'],'candidates':sorted(set(candidates)),
                'incomplete_proposals':incomplete,'wall_seconds':time.monotonic()-tick})
            del base
            if len(result)%10==0 or n==len(rows)-1:
                write(path,{'arm':arm,'rows':result,'elapsed_seconds':time.monotonic()-began,'gpu':torch.cuda.get_device_name(),
                    'peak_memory_bytes':torch.cuda.max_memory_allocated(),'dtype':CFG['backbone_dtype']})
                marker('WORD_PROPOSAL_PROGRESS',{'arm':arm,'positions':len(result),'total':len(rows),'seconds':time.monotonic()-began})


def merge():
    fixtures=read('word-fixtures.json'); sources={a:{r['key']:r for r in read('word-proposals-'+a+'.json')['rows']} for a in ARMS}
    for row in fixtures['positions']:
        union={row['reference_word']}
        for a in ARMS: union.update(sources[a][row['key']]['candidates'])
        row['candidates']=sorted(union)
    write(OUT/'word-candidates.json',fixtures)


def scoring(arm):
    import torch
    from hf_frozen_sampler import terminal_token_ids
    require_qualified(arm); model,tok=load_hf(arm); vocab=vocabulary_bytes(tok,CFG['models'][arm]['family'])
    stop=stop_ids(model,tok); head_size=model.get_output_embeddings().weight.shape[0]
    boundary={kind:torch.tensor(boundary_flags(vocab,head_size,stop,kind=='latex'),device='cuda',dtype=torch.bool)
              for kind in ('letter','latex')}
    byte_map=torch.tensor(byte_event_map(vocab,head_size,stop),device='cuda',dtype=torch.long)
    rows=read('word-candidates.json')['positions']; path=OUT/('word-scores-'+arm+'.json')
    result=read(path.name)['rows'] if path.exists() else []; done={r['key'] for r in result}; began=time.monotonic()
    with torch.inference_mode():
        for n,row in enumerate(rows):
            if row['key'] in done: continue
            tick=time.monotonic(); ids=row['prefix_ids'][arm]
            base=model.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=True)
            first_lp=canonical_head(model,base.last_hidden_state[0,-1:])[0].double().log_softmax(-1)
            first_probs=first_lp.double().exp()
            byte_probs=torch.zeros(258,device='cuda',dtype=torch.float64)
            byte_probs.scatter_add_(0,byte_map,first_probs)
            byte_values=byte_probs.cpu().tolist()
            top_values,top_ids=first_lp.topk(64)
            token_telemetry={'realized_token_id':row['realized_token_ids'][arm],
                'logprob_realized_token':float(first_lp[row['realized_token_ids'][arm]]),
                'top64_ids':top_ids.cpu().tolist(),'top64_logprobs':top_values.cpu().tolist(),
                'top64_retained_mass':float(top_values.double().exp().sum()),
                'native_token_entropy_nats':float(-(first_probs*first_lp.double()).sum())}
            prefix=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            masses=[]; reasons=[]; native_paths=[]; methods=[]; uncached_check=[]
            check_this_position=n<2 and len(ids)<=1024
            if check_this_position:
                check_prefix=model.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False)
                check_lp=canonical_head(model,check_prefix.last_hidden_state[0,-1:])[0].double().log_softmax(-1)
                check_bytes=torch.zeros(258,device='cuda',dtype=torch.float64)
                check_bytes.scatter_add_(0,byte_map,check_lp.exp()); check_bytes=check_bytes.cpu().tolist()
                del check_prefix
            for candidate in row['candidates']:
                # Hold the sampled native prefix fixed. Canonicalizing the full
                # prefix+word would change earlier sampled segmentation paths.
                suffix=tok.encode(candidate,add_special_tokens=False); full=ids+suffix
                if not suffix:
                    masses.append(None); reasons.append('empty_candidate_encoding'); native_paths.append(None); methods.append(None); continue
                if tok.decode(full,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=prefix+candidate:
                    masses.append(None); reasons.append('contextual_roundtrip_failed'); native_paths.append(None); methods.append(None); continue
                # A single ASCII digit/symbol is a first-byte event. Include
                # tokens such as '123' or '...' that cross its word boundary.
                if is_ascii_atom(candidate):
                    masses.append(byte_values[ord(candidate)]); reasons.append(None); native_paths.append(suffix)
                    if check_this_position: uncached_check.append(check_bytes[ord(candidate)])
                    methods.append('exact_next_byte_marginal_for_single_ASCII_atom'); continue
                cache=copy.deepcopy(base.past_key_values)
                ext=model.model(input_ids=torch.tensor([suffix],device='cuda'),past_key_values=cache,use_cache=True)
                h=torch.cat([base.last_hidden_state[0,-1:],ext.last_hidden_state[0]],0)
                lp=canonical_head(model,h).double().log_softmax(-1); labels=torch.tensor(suffix,device='cuda')
                probability=lp[:-1].double().gather(1,labels[:,None]).sum().exp()
                if needs_boundary(candidate): probability*=lp[-1,boundary['latex' if candidate.lstrip().startswith('\\') else 'letter']].double().exp().sum()
                masses.append(float(probability)); reasons.append(None); native_paths.append(suffix)
                if check_this_position:
                    direct=model.model(input_ids=torch.tensor([full],device='cuda'),use_cache=False)
                    reference_lp=canonical_head(model,direct.last_hidden_state[0,len(ids)-1:]).double().log_softmax(-1)
                    reference=reference_lp[:-1].gather(1,labels[:,None]).sum().exp()
                    if needs_boundary(candidate): reference*=reference_lp[-1,boundary['latex' if candidate.lstrip().startswith('\\') else 'letter']].exp().sum()
                    uncached_check.append(float(reference)); del direct
                methods.append('prefix_preserving_canonical_suffix_plus_consistent_boundary'); del ext,cache
            self_js=js(distribution([p for p in masses if p is not None]),distribution(uncached_check)) if check_this_position else None
            if self_js is not None and self_js>1e-5: raise RuntimeError('Actual sampled-prefix cache/uncached word self-check failed')
            torch.cuda.synchronize(); result.append({'key':row['key'],'masses':masses,'path_errors':reasons,
                'native_candidate_paths':native_paths,'mass_methods':methods,'byte_probs':byte_values,
                'sampled_prefix_cached_uncached_word_js':self_js,
                'token_telemetry':token_telemetry,'wall_seconds':time.monotonic()-tick})
            del base
            if len(result)%10==0 or n==len(rows)-1:
                write(path,{'arm':arm,'rows':result,'elapsed_seconds':time.monotonic()-began,'gpu':torch.cuda.get_device_name(),
                    'peak_memory_bytes':torch.cuda.max_memory_allocated(),'dtype':CFG['backbone_dtype'],
                    'normalization_dtype':'float64',
                    'byte_event_cells':'bytes0..255,EOS256,nontext/padded257'})
                marker('WORD_SCORING_PROGRESS',{'arm':arm,'positions':len(result),'total':len(rows),'seconds':time.monotonic()-began})


def report():
    fixtures=read('word-candidates.json')['positions']; scores={a:{r['key']:r for r in read('word-scores-'+a+'.json')['rows']} for a in ARMS}
    if any(len(s)!=CFG['e3_positions'] for s in scores.values()): raise RuntimeError('Incomplete word scoring')
    rows=[]; failures=[]
    for row in fixtures:
        values={a:scores[a][row['key']] for a in ARMS}
        keep=[i for i in range(len(row['candidates'])) if all(values[a]['masses'][i] is not None for a in ARMS)]
        candidates=[row['candidates'][i] for i in keep]; vectors={a:[values[a]['masses'][i] for i in keep] for a in ARMS}
        if len(keep)<2: failures.append({'key':row['key'],'reason':'fewer_than_two_joint_valid_candidates'}); continue
        try: metric=compare_distributions(vectors['S'],vectors['B'],vectors['Q'])
        except ValueError as exc: failures.append({'key':row['key'],'reason':str(exc)}); continue
        byte=compare_distributions(*(values[a]['byte_probs'][:-1] for a in ARMS))
        conditional={a:[v/sum(vectors[a]) for v in vectors[a]] if sum(vectors[a])>0 else None for a in ARMS}
        conditional_excess=(js(conditional['S'],conditional['Q'])-js(conditional['S'],conditional['B'])) if all(conditional.values()) else None
        metric['probability_contract']='hybrid exact ASCII atomic marginal and canonical-word boundary proxy'
        rows.append(dict(row,**metric,candidates=candidates,dropped_candidates=len(row['candidates'])-len(keep),
            masses=vectors,probabilities={a:distribution(vectors[a]) for a in ARMS},
            X_byte_raw=byte['X_raw'],X_byte_shape=byte['X_shape'],X_candidate_conditional=conditional_excess,
            native_token_telemetry={a:values[a]['token_telemetry'] for a in ARMS},
            cross_disagrees_when_baseline_agrees=not metric['raw']['cross']['candidate_top1_overlap'] and metric['raw']['baseline']['candidate_top1_overlap'],
            model_self_checks={a:js(distribution(vectors[a]),distribution(vectors[a])) for a in ARMS}))
    write(OUT/'word-metrics.json',{'rows':rows,'failures':failures})
    summary={'run_id':OUT.name,'config':CFG,'models':read('assets.json')['models'],
        'requested_positions':CFG['e3_positions'],'valid_positions':len(rows),'failures':failures,
        'X_raw':aggregate(rows,'X_raw',CFG['bootstrap_replicates']),
        'X_shape':aggregate(rows,'X_shape',CFG['bootstrap_replicates']),
        'X_byte_raw':aggregate(rows,'X_byte_raw',CFG['bootstrap_replicates']),
        'X_byte_shape':aggregate(rows,'X_byte_shape',CFG['bootstrap_replicates']),
        'X_candidate_conditional':aggregate(rows,'X_candidate_conditional',CFG['bootstrap_replicates']),
        'diagnostic_contract':'Byte marginals preserve all native-token probability; conditional-candidate JS removes OTHER and is a sensitivity check, not the primary estimand',
        'effective_probability_contract':'hybrid exact single-ASCII atomic event marginals and prefix-preserving canonical-word boundary proxy; not exhaustive word text marginals',
        'dropped_candidates':sum(r['dropped_candidates'] for r in rows),
        'other_dominant_counts':{a:sum(r['top_cells'][a]['other_dominant'] for r in rows) for a in ('S','baseline','cross')},
        'cross_disagrees_when_baseline_agrees':{'count':sum(r['cross_disagrees_when_baseline_agrees'] for r in rows),
            'denominator':len(rows),'top1_policy':'candidate tie-set overlap; report OTHER dominance separately'},
        'global_calibration':'not run','scientific_status':'exploratory canonical-path word-boundary proxy pilot; no original Gate1 claim',
        'limitations':['finite candidate union and canonical path approximation','native templates differ across models',
            'boundary mask is defined consistently with single-digit/symbol events, differs from literal guide mask',
            'no teacher-quality/capacity/placebo controls','shape temperature operates on word cells, not token logits',
            'bootstrap resamples prompts; positions are not independent','unknown90pct claim has no source or denominator'],
        'highest_X_shape_examples':sorted([r for r in rows if r['X_shape'] is not None],key=lambda r:r['X_shape'],reverse=True)[:30]}
    write(OUT/'word-summary.json',summary); marker('WORD_SUMMARY',{'valid_positions':len(rows),'X_raw':summary['X_raw'],'X_shape':summary['X_shape']})


def export_bundle():
    import zipfile
    if not (OUT/'word-summary.json').exists(): raise RuntimeError('No completed word summary')
    historical=OUT.parent/CFG['source_run']/'assets.json'
    write(OUT/'historical-assets.json',json.loads(historical.read_text()))
    pinned=dict(CFG,models={a:dict(spec,revision=read('assets.json')['models'][a]['revision']) for a,spec in CFG['models'].items()})
    write(OUT/'word-config-pinned-for-reproduction.json',pinned)
    # Only run artifacts, never model cache or credential mounts.
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.suffix in {'.json','.log','.txt'} and p.name!='delivery-file-manifest.json')
    manifest={str(p.relative_to(OUT)):{'sha256':file_sha(p),'size':p.stat().st_size} for p in files}
    write(OUT/'delivery-file-manifest.json',manifest)
    with zipfile.ZipFile(OUT/'word-delivery.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[OUT/'delivery-file-manifest.json']: z.write(p,p.relative_to(OUT))
    marker('WORD_EXPORT',{'files':len(files)+1,'bytes':(OUT/'word-delivery.zip').stat().st_size})


if __name__=='__main__':
    stage,arm=sys.argv[1:3]
    {'word-access':lambda:setup(),'word-prepare':lambda:setup(arm),'word-prompts':prompts_prepare,
     'word-qualify':lambda:qualify(arm),'word-rollout':rollout,'word-audit':audit_rollouts,
     'word-fixtures':fixtures,'word-proposal':lambda:proposal(arm),'word-merge':merge,
     'word-scoring':lambda:scoring(arm),'word-report':report,'word-export':export_bundle}[stage]()

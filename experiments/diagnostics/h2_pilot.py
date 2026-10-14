"""Inference-only H2 stages. All weights, datasets and outputs live on Modal."""
from collections import Counter
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
from functools import lru_cache

CONFIG = json.loads(Path(__file__).with_name('h2_pilot_config.json').read_text())
RUN = os.environ.get('H2_RUN_ID', CONFIG['run_id'])
ASSETS = Path('/assets')
OUT = Path('/results') / RUN
OUT.mkdir(parents=True, exist_ok=True)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def marker(name, value):
    print('H2_' + name + '=' + json.dumps(value, ensure_ascii=False), flush=True)


def file_sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def dataset_rows(api, repo, patterns, token):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    info = api.dataset_info(repo, files_metadata=True)
    files = sorted(f.rfilename for f in info.siblings
                   if f.rfilename.endswith(('.parquet', '.jsonl')) and any(p in f.rfilename for p in patterns))
    if not files: raise RuntimeError('Pinned dataset split missing: ' + repo)
    rows = []; hashes = {}
    for filename in files:
        path = hf_hub_download(repo, filename, repo_type='dataset', revision=info.sha, token=token)
        if filename.endswith('.parquet'): rows.extend(pq.read_table(path).to_pylist())
        else: rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
        hashes[filename] = file_sha(path)
    return rows, {'repo': repo, 'revision': info.sha, 'files_sha256': hashes, 'row_count': len(rows)}


def prepare(arm):
    from huggingface_hub import HfApi, snapshot_download, get_token
    from transformers import AutoTokenizer
    token = get_token()
    if not token: raise RuntimeError('Existing HF secret has no readable token')
    api = HfApi(token=token)
    old = json.loads((ASSETS / 'model-manifest.json').read_text())
    manifest_path = OUT / 'assets.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'models': {}, 'access_errors': {}}
    for key, spec in CONFIG['models'].items():
        try:
            # Downloading a config is the access check, not public model metadata.
            revision = spec['revision'] or api.model_info(spec['repo']).sha
            path = snapshot_download(spec['repo'], revision=revision, token=token,
                local_dir=ASSETS / 'models' / key,
                allow_patterns=['*.json', '*.jinja', '*.model', 'tokenizer*'])
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            prior = manifest['models'].get(key, old['models'].get(key, {}))
            entry = dict(spec, revision=revision, path=str(path),
                         tokenizer_sha256=hashlib.sha256(json.dumps(tok.get_vocab(), sort_keys=True).encode()).hexdigest())
            if prior.get('revision') == revision: entry['verified_weights'] = prior.get('verified_weights', {})
            manifest['models'][key] = entry
            if key == arm and not entry.get('verified_weights'):
                info = api.model_info(spec['repo'], revision=revision, files_metadata=True)
                snapshot_download(spec['repo'], revision=revision, token=token, local_dir=path,
                                  allow_patterns=['*.safetensors'])
                verified = {}
                for f in info.siblings:
                    if not f.rfilename.endswith('.safetensors'): continue
                    local = Path(path) / f.rfilename; digest = file_sha(local)
                    expected = getattr(f.lfs, 'sha256', None)
                    if expected and digest != expected: raise RuntimeError('Weight SHA mismatch')
                    verified[f.rfilename] = {'sha256': digest, 'size': local.stat().st_size}
                entry['verified_weights'] = verified
            manifest['access_errors'].pop(key, None)
        except Exception as exc:
            manifest['access_errors'][key] = {'type': type(exc).__name__,
                'status_code': getattr(getattr(exc, 'response', None), 'status_code', None)}
        write(manifest_path, manifest)
    if not (OUT / 'prompts.json').exists():
        datasets = {}
        gsm, datasets['gsm_train'] = dataset_rows(api, 'openai/gsm8k', ['main/train'], token)
        math, datasets['math_train'] = dataset_rows(api, 'DigitalLearningGmbH/MATH-lighteval', ['data/train'], token)
        gsm_test, datasets['gsm_test'] = dataset_rows(api, 'openai/gsm8k', ['main/test'], token)
        math_test, datasets['math_test'] = dataset_rows(api, 'HuggingFaceH4/MATH-500', ['test'], token)
        def sample(rows, count, kind, split, seed):
            indices = random.Random(seed).sample(range(len(rows)), count)
            return [{'id': f'{kind}-{split}-{i}', 'source': kind, 'split': split,
                     'prompt': rows[i].get('question', rows[i].get('problem')),
                     'gold': rows[i].get('answer', rows[i].get('solution'))} for i in indices]
        # Sampling 2,000 first makes the pilot an immutable subset of the full pool.
        full = sample(gsm, 1000, 'gsm8k', 'train', 42) + sample(math, 1000, 'math', 'train', 43)
        pool = full[:256] + full[1000:1256]
        quality = pool[:64] + pool[256:320] + sample(gsm_test, 64, 'gsm8k', 'test', 44) + sample(math_test, 64, 'math', 'test', 45)
        calibration = sorted(random.Random(46).sample([p['id'] for p in pool], 102))
        write(OUT / 'prompts.json', {'pool': pool, 'quality': quality, 'full_pool': full,
              'calibration_prompt_ids': calibration, 'datasets': datasets, 'seed': 42})
    from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
    source = inspect.getsource(Gemma2ForCausalLM.forward)
    (OUT / 'canonical_gemma_forward.txt').write_text(source)
    marker('PREPARE_JSON', {'models': {k: {'revision': v['revision'], 'weights_ready': bool(v.get('verified_weights'))}
                                  for k, v in manifest['models'].items()}, 'access_errors': manifest['access_errors'],
                            'prompts': len(json.loads((OUT / 'prompts.json').read_text())['pool'])})
    if arm not in manifest['models'] or not manifest['models'][arm].get('verified_weights'):
        raise RuntimeError('Requested model weights unavailable')


def model_entry(arm):
    entry = json.loads((OUT / 'assets.json').read_text())['models'][arm]
    if not entry.get('verified_weights'): raise RuntimeError('Prepare pinned weights before GPU allocation')
    return entry


def load_hf(arm):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    entry = model_entry(arm)
    tok = AutoTokenizer.from_pretrained(entry['path'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(entry['path'], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
    model.requires_grad_(False)
    return model, tok


def canonical_head(model, hidden):
    import torch
    logits = model.get_output_embeddings()(hidden)
    softcap = getattr(model.config, 'final_logit_softcapping', None)
    if softcap: logits = torch.tanh(logits / softcap) * softcap
    return logits.float()


def parity(arm):
    import torch
    import sglang
    from transformers import AutoTokenizer
    from mp_opd_parity_worker import _engine_kwargs, _server_info
    entry = model_entry(arm); tok = AutoTokenizer.from_pretrained(entry['path'], local_files_only=True)
    probes = ['2 + 2 = 4. Therefore, the answer is 4.', '\\frac{1}{2} + 0.5 = 1.\n\n  Next step: 😀 → x.']
    ids = [tok.encode(text, add_special_tokens=False) for text in probes]
    kwargs = _engine_kwargs(entry['path'], torch.cuda.get_device_name(), 'default', None)
    kwargs.update(dtype='bfloat16', context_length=4096, mem_fraction_static=0.70,
                  disable_cuda_graph=True, random_seed=42)
    engine = sglang.Engine(**kwargs)
    try:
        outputs = engine.generate(input_ids=ids, sampling_params={'max_new_tokens': 0, 'temperature': 1.0},
                                  return_logprob=True, logprob_start_len=0)
        engine_args = _server_info(engine)
    finally: engine.shutdown()
    gc.collect(); torch.cuda.empty_cache()
    model, tok = load_hf(arm)
    differences = []; head_differences = []; details = []
    with torch.inference_mode():
        for token_ids, out in zip(ids, outputs):
            tensor = torch.tensor([token_ids], device='cuda')
            forward = model(input_ids=tensor, output_hidden_states=True, use_cache=False)
            reconstructed = canonical_head(model, forward.hidden_states[-1])
            head_differences.append(float((reconstructed - forward.logits).abs().max()))
            hf = forward.logits[0, :-1].float().log_softmax(-1).gather(-1, tensor[0, 1:, None]).squeeze(-1)
            entries = out['meta_info']['input_token_logprobs']
            if [int(v[1]) for v in entries] != token_ids: raise RuntimeError('SGLang input logprob token IDs differ')
            engine_lp = torch.tensor([float(v[0]) for v in entries[1:]], device='cuda')
            differences.extend((hf - engine_lp).abs().cpu().tolist())
            details.append({'token_ids':token_ids,'reference_logprobs':hf.cpu().tolist(),
                            'engine_logprobs':engine_lp.cpu().tolist()})
    limits = CONFIG['parity']
    summary = {'arm': arm, 'tokens': len(differences), 'max_abs_logprob': max(differences),
               'mean_abs_logprob': sum(differences) / len(differences),
               'canonical_head_max_abs': max(head_differences), 'engine_args': engine_args,
               'attention_reference': 'eager', 'gpu': torch.cuda.get_device_name(),
               'reference_logsoftmax_dtype': 'float32', 'details':details,
               'pass': max(differences) <= limits['max_abs_logprob'] and
                       sum(differences) / len(differences) <= limits['mean_abs_logprob'] and
                       max(head_differences) <= limits['canonical_head_max_abs']}
    write(OUT / ('parity-' + arm + '-a' + os.environ.get('H2_STAGE_ATTEMPT', '1') + '.json'), summary)
    write(OUT / ('parity-' + arm + '.json'), summary); marker('PARITY_JSON', summary)
    if not summary['pass']: raise RuntimeError('Parity gate failed; do not generate a scientific pool')


def require_parity(arm):
    path = OUT / ('parity-' + arm + '.json')
    if not path.exists() or not json.loads(path.read_text())['pass']:
        raise RuntimeError('Required eager/SGLang/canonical-head parity not passed for ' + arm)


def render(tok, prompt, policy='native'):
    if policy == 'plain': return tok.encode(prompt + '\n', add_special_tokens=True)
    from mp_opd_parity_worker import _token_ids
    return _token_ids(tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                      tokenize=True, add_generation_prompt=True))


def responses(kind='rollout', arm='S'):
    path = OUT / (kind + '-' + arm + '.json')
    value = json.loads(path.read_text())
    expected = CONFIG['prompts'] * CONFIG['samples_per_prompt'] if kind == 'rollout' else CONFIG['quality_problems'] * CONFIG['quality_samples']
    if len(value) != expected or len({r['id'] for r in value}) != expected:
        raise RuntimeError('Response pool count/uniqueness contract violated')
    return value


def generation(arm, kind):
    import torch
    import sglang
    from transformers import AutoTokenizer
    from hf_frozen_sampler import terminal_token_ids
    from mp_opd_parity_worker import _engine_kwargs, _server_info
    require_parity(arm)
    entry = model_entry(arm); tok = AutoTokenizer.from_pretrained(entry['path'], local_files_only=True)
    selected = json.loads((OUT / 'prompts.json').read_text())['pool' if kind == 'rollout' else 'quality']
    n = CONFIG['samples_per_prompt'] if kind == 'rollout' else CONFIG['quality_samples']
    if kind == 'rollout' and arm != 'S': raise ValueError('Only frozen student generates the common rollout pool')
    tasks = [(p, j) for p in selected for j in range(n)]
    path = OUT / (kind + '-' + arm + '.json')
    records = json.loads(path.read_text()) if path.exists() else []
    if len(records) == len(tasks):
        responses(kind, arm); marker('GENERATION_ALREADY_COMPLETE', {'kind': kind, 'arm': arm, 'responses': len(records)}); return
    if [r['id'] for r in records] != [f"{p['id']}-{kind}-{j}" for p, j in tasks[:len(records)]]:
        raise RuntimeError('Resume IDs are not a prefix of the fixed task manifest')
    stop = terminal_token_ids(tok)
    kwargs = _engine_kwargs(entry['path'], torch.cuda.get_device_name(), 'default', None)
    kwargs.update(dtype='bfloat16', context_length=8192, mem_fraction_static=.80,
                  random_seed=42, max_running_requests=64, disable_cuda_graph=True)
    engine = sglang.Engine(**kwargs); start = time.monotonic(); times = []
    try:
        for offset in range(len(records), len(tasks), 64):
            batch = tasks[offset:offset + 64]
            prefixes = [render(tok, p['prompt']) for p, j in batch]
            if any(len(p) + CONFIG['max_new_tokens'] > 8192 for p in prefixes):
                raise RuntimeError('Prompt exceeds declared context; no silent truncation')
            seeds = [int.from_bytes(hashlib.sha256(f"42/{kind}/{p['id']}/{j}".encode()).digest()[:4], 'little') & 0x7fffffff for p, j in batch]
            sampling = [{'temperature': 1.0, 'top_p': 1.0, 'top_k': -1,
                         'max_new_tokens': 1024, 'sampling_seed': seed,
                         'stop_token_ids': list(stop), 'ignore_eos': False} for seed in seeds]
            tick = time.monotonic()
            outputs = engine.generate(input_ids=prefixes, sampling_params=sampling)
            count = 0
            for (p, j), prefix, seed, out in zip(batch, prefixes, seeds, outputs):
                native = out.get('output_ids')
                if native is None: raise RuntimeError('Native output IDs absent; decoding/re-encoding cannot substitute')
                end = next((i for i, v in enumerate(native) if v in stop), len(native))
                content = native[:end]
                decoded_prefix = tok.decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                whole = tok.decode(prefix + content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                if not whole.startswith(decoded_prefix): raise RuntimeError('Contextual response decoding changes prefix')
                text = whole[len(decoded_prefix):]
                records.append(dict(p, id=f"{p['id']}-{kind}-{j}", prompt_id=p['id'], sample=j,
                                    prefix_ids=prefix, content_ids=content, output_ids=native,
                                    text=text, engine_text=out['text'], sampling_seed=seed,
                                    finish_reason=out['meta_info'].get('finish_reason'),
                                    model_revision=entry['revision']))
                count += len(native)
            times.append({'offset': offset, 'responses': len(batch), 'tokens': count, 'seconds': time.monotonic() - tick})
            write(path, records); marker('GENERATION_BATCH', times[-1])
    finally: engine.shutdown()
    checked = responses(kind, arm)
    summary = {'arm': arm, 'kind': kind, 'responses': len(checked), 'generated_tokens': sum(len(r['output_ids']) for r in checked),
               'timings': times, 'elapsed_seconds': time.monotonic() - start, 'engine_args': _server_info(engine),
               'template_policy': 'native user-only chat template; exact rendered prefix IDs persisted',
               'sampling': {k: CONFIG[k] for k in ('seed','temperature','top_p','top_k','max_new_tokens')}}
    write(OUT / (kind + '-' + arm + '-summary.json'), summary); marker('GENERATION_SUMMARY', summary)
    if kind == 'quality': quality_verify(arm, checked)


def quality_verify(arm, records):
    from answer_verifier import verify_response, extract_boxed_answer
    scored = []
    for row in records:
        source = 'gsm8k' if row['source'] == 'gsm8k' else 'math_minus_500'
        gold = row['gold']
        if source == 'math_minus_500' and '\\boxed' in gold: gold = extract_boxed_answer(gold)
        try: correct = bool(verify_response(row['text'], gold, source)); error = None
        except Exception as exc: correct = None; error = type(exc).__name__
        scored.append({'id': row['id'], 'prompt_id': row['prompt_id'], 'split': row['split'],
                       'source': row['source'], 'correct': correct, 'error': error})
    write(OUT / ('quality-' + arm + '-scores.json'), {'scorer': 'SimCT scripts/sft/answer_verifier.py; heuristic pilot control',
                                                    'rows': scored, 'errors_are_not_incorrect': True})


def score(arm):
    import torch
    from h2_measurements import vocabulary_bytes, byte_spans
    require_parity(arm); pool = responses()
    model, tok = load_hf(arm)
    family = 'gemma' if arm in {'S','G9'} else 'bpe'
    vocab = vocabulary_bytes(tok, family)
    root = OUT / 'hidden' / arm; root.mkdir(parents=True, exist_ok=True)
    index_path = root / 'index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    started = time.monotonic()
    with torch.inference_mode():
        for number, row in enumerate(pool):
            if row['id'] in index: continue
            prefix = render(tok, row['prompt'])
            if arm == 'S': content = row['content_ids']; full = prefix + content
            else:
                text_prefix = tok.decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                full = tok.encode(text_prefix + row['text'], add_special_tokens=False)
                if full[:len(prefix)] != prefix:
                    index[row['id']] = {'error': 'template/response retokenization changed prompt boundary'}
                    write(index_path, index); continue
                content = full[len(prefix):]
            try: spans = byte_spans(content, vocab, row['text'])
            except ValueError as exc:
                index[row['id']] = {'error': str(exc)}; write(index_path, index); continue
            if not content:
                index[row['id']] = {'error': 'empty response'}; write(index_path, index); continue
            tensor = torch.tensor([full], device='cuda')
            hidden = model.model(input_ids=tensor, use_cache=False).last_hidden_state[0, len(prefix)-1:len(full)-1]
            # Preserve BF16 bits. Reconstruct logits later from the qualified head.
            destination = root / (row['id'] + '.pt')
            torch.save(hidden.cpu().contiguous(), destination)
            top_ids = []; top_logprobs = []; retained = []; lses = []; entropies = []
            for h in hidden.split(16):
                logits = canonical_head(model, h); lp = logits.log_softmax(-1); probs = lp.exp()
                values, ids = lp.topk(64, -1)
                top_ids.extend(ids.cpu().tolist()); top_logprobs.extend(values.cpu().tolist())
                retained.extend(values.exp().sum(-1).cpu().tolist())
                lses.extend(logits.logsumexp(-1).cpu().tolist())
                entropies.extend((-torch.xlogy(probs, probs).sum(-1)).cpu().tolist())
            record = {'content_ids': content, 'spans': spans, 'hidden_path': str(destination),
                                'top64_ids': top_ids, 'top64_logprobs': top_logprobs,
                                'retained_mass': retained, 'logsumexp': lses, 'entropy': entropies}
            record_path = root / (row['id'] + '.json')
            write(record_path, record)
            index[row['id']] = {'record_path': str(record_path)}
            write(index_path, index)
            if number % 32 == 0: marker('SCORING_PROGRESS', {'arm': arm, 'done': number + 1, 'total': len(pool), 'seconds': time.monotonic()-started})
    marker('SCORING_SUMMARY', {'arm': arm, 'responses': len(index), 'failures': sum('error' in r for r in index.values()),
                              'elapsed_seconds': time.monotonic()-started})


@lru_cache(maxsize=16)
def scoring_record(arm, rid):
    entry = json.loads((OUT / 'hidden' / arm / 'index.json').read_text())[rid]
    return json.loads(Path(entry['record_path']).read_text()) if 'record_path' in entry else entry


def load_heads(arms):
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, AutoConfig
    from hf_frozen_sampler import terminal_token_ids
    from h2_measurements import vocabulary_bytes, prefix_map
    heads = {}; toks = {}; vocabs = {}; stops = {}; configs = {}
    for arm in arms:
        path = Path(model_entry(arm)['path'])
        toks[arm] = AutoTokenizer.from_pretrained(path, local_files_only=True)
        configs[arm] = AutoConfig.from_pretrained(path, local_files_only=True)
        vocabs[arm] = vocabulary_bytes(toks[arm], 'gemma' if arm in {'S','G9'} else 'bpe')
        stops[arm] = set(terminal_token_ids(toks[arm]))
        weights = None
        candidates = ['lm_head.weight', 'model.embed_tokens.weight']
        index_file = path / 'model.safetensors.index.json'
        if index_file.exists():
            weight_map = json.loads(index_file.read_text())['weight_map']
            key = next(k for k in candidates if k in weight_map)
            with safe_open(str(path / weight_map[key]), framework='pt', device='cpu') as f: weights = f.get_tensor(key)
        else:
            with safe_open(str(path / 'model.safetensors'), framework='pt', device='cpu') as f:
                key = next(k for k in candidates if k in f.keys()); weights = f.get_tensor(key)
        heads[arm] = weights.to(device='cuda', dtype=torch.bfloat16)
        if heads[arm].shape[0] < len(vocabs[arm]): raise RuntimeError('Head vocabulary smaller than tokenizer')
    maps = {a: torch.tensor(prefix_map(vocabs['S'], vocabs[a], stops['S'], stops[a])[0],
                            device='cuda', dtype=torch.long) for a in arms}
    for a in arms:
        extra = heads[a].shape[0] - maps[a].numel()
        if extra: maps[a] = torch.cat([maps[a],torch.full((extra,),len(vocabs['S'])+1,device='cuda',dtype=torch.long)])
    common = set(p for p in vocabs['S'] if p)
    for a in arms: common &= set(p for p in vocabs[a] if p)
    canonical = {}
    for i, p in enumerate(vocabs['S']):
        if p and p in common: canonical.setdefault(p, i)
    e1_maps = {a: torch.tensor([canonical.get(p, len(vocabs['S'])+1) if i not in stops[a] else len(vocabs['S'])
                               for i,p in enumerate(vocabs[a])], device='cuda') for a in arms}
    for a in arms:
        extra = heads[a].shape[0] - e1_maps[a].numel()
        if extra: e1_maps[a] = torch.cat([e1_maps[a],torch.full((extra,),len(vocabs['S'])+1,device='cuda',dtype=torch.long)])
    return heads, toks, vocabs, maps, e1_maps, configs


def head_logits(head, hidden, config):
    import torch
    logits = hidden.to('cuda') @ head.T
    cap = getattr(config, 'final_logit_softcapping', None)
    if cap: logits = torch.tanh(logits / cap) * cap
    return logits.float()


def tensor_js(p, q):
    import torch
    m = (p + q) / 2
    return .5 * (torch.xlogy(p, p / m.clamp_min(1e-38)).sum(-1) +
                 torch.xlogy(q, q / m.clamp_min(1e-38)).sum(-1))


def events():
    import torch
    from h2_measurements import strict_pairs, category
    pool = responses()
    manifest = json.loads((OUT / 'assets.json').read_text())
    arms = [a for a in ('S','G9','Q','L') if a in manifest['models'] and
            (OUT / 'hidden' / a / 'index.json').exists()]
    if 'S' not in arms or 'G9' not in arms or not any(a in arms for a in ('Q','L')):
        raise RuntimeError('Need completed S, G9 and at least one cross-family scoring stage')
    for a in arms: require_parity(a)
    indices = {a: json.loads((OUT / 'hidden' / a / 'index.json').read_text()) for a in arms}
    if any(len(v) != len(pool) for v in indices.values()): raise RuntimeError('Incomplete scoring index')
    heads, toks, vocabs, maps, e1_maps, configs = load_heads(arms)
    size = len(vocabs['S']) + 2
    rows = []; e3_candidates = []; census = {a: Counter() for a in arms}; failures = []
    calibration = set(json.loads((OUT / 'prompts.json').read_text())['calibration_prompt_ids'])
    with torch.inference_mode():
        for ri, response in enumerate(pool):
            rid = response['id']
            if any('error' in indices[a][rid] for a in arms):
                failures.append(rid); continue
            records = {a: scoring_record(a, rid) for a in arms}
            spans = {a: records[a]['spans'] for a in arms}
            aligned = {a: strict_pairs(spans['S'], spans[a]) for a in arms}
            for a in arms:
                census[a]['tokens'] += len(spans['S']); census[a]['strict'] += len(aligned[a])
            common_positions = sorted(set.intersection(*(set(v) for v in aligned.values())))
            h = {a: torch.load(records[a]['hidden_path'], weights_only=True) for a in arms}
            raw = response['text'].encode()
            for off in range(0, len(common_positions), 16):
                positions = common_positions[off:off+16]
                p = {}; e1 = {}; mass = {}; entropy = {}
                for a in arms:
                    gathered = h[a][[aligned[a][j] for j in positions]]
                    logits = head_logits(heads[a], gathered, configs[a]); probs = logits.softmax(-1)
                    target = torch.zeros((len(positions), size), device='cuda')
                    target.scatter_add_(1, maps[a].expand(len(positions), -1), probs)
                    if float((target.sum(-1)-1).abs().max()) > 1e-5: raise RuntimeError('E2 mass preservation failed')
                    p[a] = target
                    shared = torch.zeros_like(target); shared.scatter_add_(1, e1_maps[a].expand(len(positions), -1), probs)
                    shared[:, -1] = 0; mass[a] = shared.sum(-1)
                    e1[a] = shared / mass[a][:,None].clamp_min(1e-38)
                    entropy[a] = -torch.xlogy(probs, probs).sum(-1)
                if float(tensor_js(p['S'], p['S']).abs().max()) > 1e-7: raise RuntimeError('E2 identity failed')
                b = {a: tensor_js(p['S'], p[a]).cpu().tolist() for a in arms}
                b1 = {a: tensor_js(e1['S'], e1[a]).cpu().tolist() for a in arms}
                for k,j in enumerate(positions):
                    left,right,_ = spans['S'][j]; cat = category(raw[left:right])
                    record = {'response_id':rid,'prompt_id':response['prompt_id'],'position':j,
                              'byte_start':left,'byte_end':right,'category':cat,
                              'first_five':j<5,'calibration':response['prompt_id'] in calibration,
                              'E2_JS':{a:b[a][k] for a in arms},'E1_JS':{a:b1[a][k] for a in arms},
                              'E1_retained':{a:float(mass[a][k]) for a in arms},
                              'E2_residual':{a:float(p[a][k,-1]) for a in arms},
                              'entropy':{a:float(entropy[a][k]) for a in arms}}
                    for a in ('Q','L'):
                        if a in arms: record['X_E2_'+a] = b[a][k] - b['G9'][k]
                    rows.append(record)
                    # Stable hash selects the sample independently of observed divergence.
                    if j >= 5 and cat != 'whitespace':
                        rank = hashlib.sha256(f"42/{rid}/{j}".encode()).hexdigest()
                        e3_candidates.append((rank, record))
            if ri % 32 == 0: marker('EVENTS_PROGRESS', {'responses':ri+1,'positions':len(rows)})
    # Select by category and prompt calibration split, independently of outcomes.
    strata = {}
    for rank,row in e3_candidates: strata.setdefault((row['category'],row['calibration']),[]).append((rank,row))
    chosen = []
    for key, group in sorted(strata.items()):
        group.sort(key=lambda x:x[0]); chosen.extend(group[:CONFIG['e3_positions']//max(1,len(strata))])
    chosen_ids = {(r['response_id'],r['position']) for _,r in chosen}
    for pair in sorted(e3_candidates):
        if len(chosen) >= CONFIG['e3_positions']: break
        key = (pair[1]['response_id'],pair[1]['position'])
        if key not in chosen_ids: chosen.append(pair); chosen_ids.add(key)
    with (OUT / 'e2-records.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
    write(OUT/'e3-position-manifest.json',[r for _,r in chosen])
    write(OUT/'structural-census.json',{'arms':{a:dict(c) for a,c in census.items()},'failed_response_ids':failures,
          'position_set':'strict for every available scored model; does not imply L strictness when L unavailable'})
    marker('EVENTS_SUMMARY',{'arms':arms,'positions':len(rows),'e3_positions':len(chosen),'failed_responses':len(failures)})


def main():
    stage, arm = sys.argv[1:3]
    if stage not in {'prepare','report','e3-merge'}:
        import sitecustomize
        if not sitecustomize.H2_PATCH_APPLIED:
            raise RuntimeError('Study-scoped Gemma canonical softcap adapter not active')
    if stage == 'prepare': prepare(arm)
    elif stage == 'parity': parity(arm)
    elif stage == 'rollout': generation(arm, 'rollout')
    elif stage == 'quality': generation(arm, 'quality')
    elif stage == 'score': score(arm)
    elif stage == 'events': events()
    elif stage in {'e3-candidates','e3-merge','e3-score'}:
        import h2_e3
        {'e3-candidates':lambda:h2_e3.candidates(arm),'e3-merge':h2_e3.merge,
         'e3-score':lambda:h2_e3.score_candidates(arm)}[stage]()
    elif stage == 'report':
        from h2_report import report
        report()
    else: raise RuntimeError('Stage implementation not yet enabled: ' + stage)


if __name__ == '__main__': main()

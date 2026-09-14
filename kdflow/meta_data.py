"""Same-corpus B/M sampling with a separate, stateless RNG stream."""
import hashlib
import random
import re
import unicodedata

POLICY = 'uniform-prompt-one-reference-v2'


def prompt_id(text):
    normalized=re.sub(r"\s+", " ", unicodedata.normalize("NFKC",text)).strip().casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


class MetaSampler:
    def __init__(self, rows, train_prompts, seed, batch_size=16):
        train_ids={prompt_id(x) for x in train_prompts}
        unique={};count=0
        for row in rows:
            key=prompt_id(row['prompt'])
            if key not in train_ids:
                raise ValueError('Meta prompt is outside the pinned training corpus: '+key)
            if not isinstance(row['reference'],str) or not row['reference'].strip():
                raise ValueError('Empty/nontext meta reference: '+key)
            count+=1
            if key not in unique:
                unique[key]=dict(prompt=row['prompt'],id=key,references=set())
            if unique[key]['prompt']!=row['prompt']:
                raise ValueError('Ambiguous normalized meta prompts: '+key)
            unique[key]['references'].add(row['reference'])
        self.rows=sorted(unique.values(),key=lambda r:r['id'])
        for row in self.rows:row['references']=tuple(sorted(row['references']))
        self.audit=dict(policy=POLICY,input_rows=count,unique_prompts=len(self.rows),
            unique_references=sum(len(r['references']) for r in self.rows),
            multi_reference_prompts=sum(len(r['references'])>1 for r in self.rows),
            groups_sha256=hashlib.sha256(repr([(r['id'],r['references']) for r in self.rows]).encode()).hexdigest())
        self.seed=seed;self.batch_size=batch_size
        if len(self.rows)<batch_size:
            raise ValueError('Insufficient distinct meta references')

    def batch(self, step, rollout_prompts):
        excluded={prompt_id(x) for x in rollout_prompts}
        candidates=[r for r in self.rows if r['id'] not in excluded]
        if len(candidates)<self.batch_size:
            raise ValueError('Insufficient meta references disjoint from this rollout batch')
        # Stable sampling does not disturb B shuffling or student generation RNG.
        rng=random.Random(f'kdflow-meta-v2:{self.seed}:{step}')
        selected=rng.sample(candidates,self.batch_size)
        # Per-prompt reference RNG: duplicate rows and other prompts' reference
        # counts never alter the selected prompt set or another prompt's draw.
        return [dict(id=r['id'],prompt=r['prompt'],reference=random.Random(
            f'kdflow-reference-v2:{self.seed}:{step}:{r["id"]}').choice(r['references']))
            for r in selected]


def prepare_meta(rows, render, tokenizer, max_len):
    prepared=[];excluded=0
    for row in rows:
        prompt=render(row);reference=row.get('label')
        if not isinstance(reference,str) or not reference.strip():
            raise ValueError('Meta selected.parquet must contain nonempty teacher references in label')
        length=len(tokenizer.encode(prompt,add_special_tokens=False))+len(tokenizer.encode(reference,add_special_tokens=False))+1
        if length>max_len:
            excluded+=1;continue
        prepared.append(dict(prompt=prompt,reference=reference))
    return prepared,excluded

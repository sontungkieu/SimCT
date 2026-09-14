"""Same-corpus B/M sampling with a separate, stateless RNG stream."""
import hashlib
import random
import re
import unicodedata


def prompt_id(text):
    normalized=re.sub(r"\s+", " ", unicodedata.normalize("NFKC",text)).strip().casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


class MetaSampler:
    def __init__(self, rows, train_prompts, seed, batch_size=16):
        train_ids={prompt_id(x) for x in train_prompts}
        unique={}
        for row in rows:
            key=prompt_id(row['prompt'])
            if key not in train_ids:
                raise ValueError('Meta prompt is outside the pinned training corpus: '+key)
            if not isinstance(row['reference'],str) or not row['reference'].strip():
                raise ValueError('Empty/nontext meta reference: '+key)
            if key in unique and unique[key]['reference']!=row['reference']:
                raise ValueError('Conflicting meta references: '+key)
            unique[key]=dict(row,id=key)
        self.rows=sorted(unique.values(),key=lambda r:r['id'])
        self.seed=seed;self.batch_size=batch_size
        if len(self.rows)<batch_size:
            raise ValueError('Insufficient distinct meta references')

    def batch(self, step, rollout_prompts):
        excluded={prompt_id(x) for x in rollout_prompts}
        candidates=[r for r in self.rows if r['id'] not in excluded]
        if len(candidates)<self.batch_size:
            raise ValueError('Insufficient meta references disjoint from this rollout batch')
        # Stable sampling does not disturb B shuffling or student generation RNG.
        rng=random.Random(f'kdflow-meta-v1:{self.seed}:{step}')
        return rng.sample(candidates,self.batch_size)

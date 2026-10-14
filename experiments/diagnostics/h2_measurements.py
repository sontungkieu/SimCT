"""Byte events and prompt-clustered statistics for the H2 pilot."""
import math
import random
import re


def byte_decoder():
    base = list(range(33, 127)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars = list(base); n = 0
    for b in range(256):
        if b not in base: base.append(b); chars.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(base, chars)}


def vocabulary_bytes(tokenizer, family):
    decoder = byte_decoder(); special = set(tokenizer.all_special_ids); values = []
    for i in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(i)
        if i in special: values.append(None)
        elif family == 'gemma':
            match = re.fullmatch(r'<0x([0-9A-Fa-f]{2})>', piece)
            values.append(bytes([int(match.group(1), 16)]) if match else piece.replace('▁', ' ').encode())
        else:
            try: values.append(bytes(decoder[c] for c in piece))
            except KeyError: raise ValueError('Unsupported added non-special token')
    return values


def byte_spans(ids, vocab, text):
    pieces = [vocab[i] for i in ids]
    if any(p is None or not p for p in pieces): raise ValueError('Special/empty response token')
    if b''.join(pieces) != text.encode(): raise ValueError('Native token bytes differ from response context bytes')
    result = []; start = 0
    for i, p in zip(ids, pieces):
        result.append((start, start + len(p), i)); start += len(p)
    return result


def strict_pairs(student, teacher):
    teacher_by_span = {(a, b): i for i, (a, b, _) in enumerate(teacher)}
    return {j: teacher_by_span[a, b] for j, (a, b, _) in enumerate(student) if (a, b) in teacher_by_span}


def category(raw):
    text = raw.decode('utf-8', errors='replace')
    if text.isspace(): return 'whitespace'
    if text.isdigit(): return 'digit'
    if any(c in text for c in ('\\', '$', '{', '}')): return 'latex'
    if re.fullmatch(r'[^\w\s]+', text): return 'punctuation'
    return 'content'


def prefix_map(student_vocab, teacher_vocab, student_stop, teacher_stop):
    """Mass-preserving map: duplicate strings merge; EOS and residual are explicit."""
    canonical = {}
    for i, piece in enumerate(student_vocab):
        if piece and i not in student_stop: canonical.setdefault(piece, i)
    trie = {}
    for piece, i in canonical.items():
        node = trie
        for byte in piece: node = node.setdefault(byte, {})
        node[None] = i
    end = len(student_vocab); residual = end + 1
    result = []
    for i, piece in enumerate(teacher_vocab):
        if i in teacher_stop: result.append(end); continue
        node = trie; best = residual
        for byte in piece or b'':
            if byte not in node: break
            node = node[byte]
            if None in node: best = node[None]
        result.append(best)
    return result, end + 2


def js(p, q):
    if len(p) != len(q): raise ValueError('Different event supports')
    if any(x < 0 or not math.isfinite(x) for x in [*p, *q]): raise ValueError('Invalid mass')
    if not math.isclose(sum(p), 1, abs_tol=1e-6) or not math.isclose(sum(q), 1, abs_tol=1e-6):
        raise ValueError('Mass must sum to one')
    value = 0
    for a, b in zip(p, q):
        m = (a + b) / 2
        if a: value += .5 * a * math.log(a / m)
        if b: value += .5 * b * math.log(b / m)
    return value


def bootstrap_prompt_mean(rows, value_key, *, seed=42, repeats=2000):
    groups = {}
    for row in rows: groups.setdefault(row['prompt_id'], []).append(row[value_key])
    means = [sum(v) / len(v) for _, v in sorted(groups.items())]
    if not means: return {'prompts': 0, 'mean': None, 'ci95': None}
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(repeats))
    return {'prompts': len(means), 'positions': len(rows), 'mean': sum(means) / len(means),
            'ci95': [draws[int(.025 * repeats)], draws[min(repeats - 1, int(.975 * repeats))]],
            'estimand': 'equal prompt weight; mean of position means within each prompt'}


def next_word_event(text):
    """Finite candidate convention: include the first terminating boundary char.

    Including the boundary prevents e.g. '1' and '12' being overlapping events.
    A string without an observed boundary is incomplete and is not a candidate.
    Canonical chain-rule scoring remains a segmentation-dependent approximation.
    """
    leading = len(text) - len(text.lstrip())
    if leading == len(text): return None
    first = text[leading]
    if not first.isalnum() and first != '_': return text[:leading + 1]
    digit = first.isdigit()
    for j in range(leading + 1, len(text)):
        c = text[j]
        same = c.isdigit() if digit else (c.isalpha() or c == '_')
        if not same: return text[:j + 1]
    return None


def assert_prefix_free(candidates):
    ordered = sorted(s.encode() for s in candidates)
    if len(set(ordered)) != len(ordered): raise ValueError('Duplicate candidates')
    if any(b.startswith(a) for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Candidate events overlap by byte prefix')

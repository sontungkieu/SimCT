"""One consistent next-event grammar for candidate parsing and boundary masks."""

ASCII_LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def next_word(text, terminated=False):
    start=0
    while start<len(text) and text[start].isspace(): start+=1
    if start==len(text): return None
    first=text[start]; end=start+1
    if first=='\\':
        if end==len(text) and not terminated: return None
        while end<len(text) and text[end] in ASCII_LETTERS: end+=1
        if end>start+1 and end==len(text) and not terminated: return None
    elif first.isalpha():
        while end<len(text) and text[end].isalpha(): end+=1
        if end==len(text) and not terminated: return None
    return text[:end]


def needs_boundary(value):
    value=value.lstrip()
    return bool(value and (value[0]=='\\' or value[0].isalpha()))


def first_character(piece):
    if not piece: return ''
    for end in range(1,min(4,len(piece))+1):
        try:
            value=piece[:end].decode('utf-8')
            if value: return value[0]
        except UnicodeDecodeError as exc:
            if exc.reason!='unexpected end of data': return ''
    return ''


def boundary_flags(vocab,head_size,stop,latex=False):
    flags=[False]*head_size
    for i,piece in enumerate(vocab):
        first=first_character(piece)
        continues=bool(first and first in ASCII_LETTERS) if latex else first.isalpha()
        flags[i]=i in stop or bool(first and not continues)
    return flags


def byte_event_map(vocab,head_size,stop):
    """Exact first-native-byte projection, plus EOS and non-text/padded cells."""
    mapping=[256 if i in stop else piece[0] if piece else 257 for i,piece in enumerate(vocab)]
    return mapping+[257]*(head_size-len(mapping))


def is_ascii_atom(value):
    return len(value)==1 and ord(value)<128 and not value.isalpha() and not value.isspace() and not needs_boundary(value)

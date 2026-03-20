#!/usr/bin/env python3
"""Export vocab.bin from tokenizer.json for the C inference engine.

Binary format (matches load_vocab in infer.m):
  uint32 num_entries
  uint32 max_id  (unused, set to num_entries)
  For each entry (0..num_entries-1):
    uint16 byte_len
    char[byte_len] UTF-8 string  (empty if byte_len==0)

Usage:
    python export_vocab.py /path/to/tokenizer.json [output.bin]
"""
import json
import struct
import sys


def build_byte_decoder():
    """Inverse of the GPT-2 bytes_to_unicode mapping.
    Converts BPE visual chars (Ġ=space, Ċ=newline, etc.) back to raw bytes."""
    bs = list(range(ord('!'), ord('~') + 1)) + \
         list(range(ord('¡'), ord('¬') + 1)) + \
         list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def decode_bpe_token(token_str, byte_decoder):
    """Convert a BPE token string to its actual UTF-8 bytes."""
    try:
        raw_bytes = bytes([byte_decoder[c] for c in token_str])
        return raw_bytes.decode('utf-8', errors='replace')
    except (KeyError, UnicodeDecodeError):
        return token_str


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_vocab.py <tokenizer.json> [output.bin]", file=sys.stderr)
        sys.exit(1)

    tok_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'vocab.bin'

    with open(tok_path) as f:
        tok = json.load(f)

    # Build id->string mapping from vocab + added_tokens
    vocab = {}
    if 'model' in tok and 'vocab' in tok['model']:
        for token_str, token_id in tok['model']['vocab'].items():
            vocab[token_id] = token_str

    if 'added_tokens' in tok:
        for entry in tok['added_tokens']:
            vocab[entry['id']] = entry['content']

    num_entries = max(vocab.keys()) + 1 if vocab else 0
    print(f"Vocab size: {len(vocab)} tokens, max_id: {num_entries - 1}")

    # Decode BPE visual encoding (Ġ→space, Ċ→newline, etc.)
    byte_decoder = build_byte_decoder()

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', num_entries))
        f.write(struct.pack('<I', num_entries))

        for i in range(num_entries):
            s = vocab.get(i, '')
            if s:
                s = decode_bpe_token(s, byte_decoder)
            b = s.encode('utf-8') if s else b''
            f.write(struct.pack('<H', len(b)))
            if b:
                f.write(b)

    print(f"Wrote {out_path} ({num_entries} entries)")


if __name__ == '__main__':
    main()

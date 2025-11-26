#!/usr/bin/env python3
"""
make_pairs.py

Create a JSON file of {instruction, output} pairs.

Usage:
  python make_pairs.py \
    --instructions educational_instruct_filtered.json \
    --snippets ..\..\..\pascal_snippets.txt \
    --output pascal_instruction_pairs.json

Default behavior pairs entries up to the smaller of the two lists.
Use --cycle to repeat snippets when there are fewer snippets than instructions.
"""
import json
import argparse
import os
import re
from typing import List, Any


def load_instructions(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    instrs: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                instrs.append(item.strip())
            elif isinstance(item, dict):
                # common keys (note: some files may have a typo 'instuction')
                for key in ('instruction', 'instuction', 'prompt', 'input', 'text'):
                    if key in item and isinstance(item[key], str):
                        instrs.append(item[key].strip())
                        break
                else:
                    # fallback: pick the first string-valued field
                    picked = None
                    for v in item.values():
                        if isinstance(v, str):
                            picked = v.strip(); break
                    if picked is not None:
                        instrs.append(picked)
                    else:
                        # as last resort, serialize the item
                        instrs.append(json.dumps(item, ensure_ascii=False))
            else:
                instrs.append(str(item))
    elif isinstance(data, dict):
        # if top-level object, try to find a list inside
        for v in data.values():
            if isinstance(v, list):
                # recurse by writing to a temp file-like flow
                for item in v:
                    if isinstance(item, str):
                        instrs.append(item.strip())
                    elif isinstance(item, dict):
                        for key in ('instruction', 'instuction', 'prompt', 'input', 'text'):
                            if key in item and isinstance(item[key], str):
                                instrs.append(item[key].strip()); break
                        else:
                            picked = None
                            for vv in item.values():
                                if isinstance(vv, str):
                                    picked = vv.strip(); break
                            instrs.append(picked if picked is not None else json.dumps(item, ensure_ascii=False))
                break
        if not instrs:
            instrs.append(json.dumps(data, ensure_ascii=False))
    else:
        instrs.append(str(data))

    return instrs


def load_snippets(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    # split on lines that contain only '---' (with optional surrounding whitespace)
    parts = re.split(r'(?m)^\s*---\s*$', text)
    snippets = [p.strip() for p in parts if p.strip()]
    return snippets


def make_pairs(instructions: List[str], snippets: List[str], mode: str = 'min', cycle: bool = False) -> List[dict]:
    if mode == 'min':
        n = min(len(instructions), len(snippets))
    elif mode == 'max':
        n = max(len(instructions), len(snippets))
    else:
        n = len(instructions)

    pairs: List[dict] = []
    if not snippets:
        snippets = ['']

    for i in range(n):
        instr = instructions[i] if i < len(instructions) else ''
        if i < len(snippets):
            out = snippets[i]
        else:
            if cycle:
                out = snippets[i % len(snippets)]
            else:
                out = ''
        pairs.append({'instruction': instr, 'output': out})
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Pair instructions with Pascal snippets.')
    parser.add_argument('--instructions', '-i', default='educational_instruct_filtered.json')
    parser.add_argument('--snippets', '-s', default='..\\..\\pascal_snippets.txt')
    parser.add_argument('--output', '-o', default='pascal_instruction_pairs.json')
    parser.add_argument('--mode', choices=('min', 'max', 'all'), default='min',
                        help='How many pairs to create (min: up to smaller list, max: up to larger list, all: all instructions)')
    parser.add_argument('--cycle', action='store_true', help='Cycle snippets when fewer than instructions')
    args = parser.parse_args()

    instr_path = os.path.abspath(args.instructions)
    snippets_path = os.path.abspath(args.snippets)
    out_path = os.path.abspath(args.output)

    if not os.path.exists(instr_path):
        print(f'Instructions file not found: {instr_path}')
        return
    if not os.path.exists(snippets_path):
        print(f'Snippets file not found: {snippets_path}')
        return

    instructions = load_instructions(instr_path)
    snippets = load_snippets(snippets_path)

    if args.mode == 'all':
        mode = 'all'
        n = len(instructions)
    else:
        mode = args.mode

    pairs = make_pairs(instructions, snippets, mode=mode, cycle=args.cycle)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f'Wrote {len(pairs)} pairs to {out_path}')


if __name__ == '__main__':
    main()

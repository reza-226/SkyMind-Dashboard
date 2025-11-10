#!/usr/bin/env python3
import pickle
from pathlib import Path

cache_file = Path('analysis/realtime/realtime_cache.pkl')
with open(cache_file, 'rb') as f:
    cache = pickle.load(f)

print('Cache type:', type(cache))
if isinstance(cache, dict):
    print('Keys:', list(cache.keys()))
    for key in cache.keys():
        val = cache[key]
        print(f'\n{key}:')
        print(f'  Type: {type(val)}')
        if isinstance(val, list):
            print(f'  Length: {len(val)}')
            if val:
                print(f'  First item: {val[0]}')
        elif isinstance(val, dict):
            print(f'  Keys: {list(val.keys())[:10]}')
elif isinstance(cache, list):
    print('List length:', len(cache))
    if cache:
        print('First item:', cache[0])

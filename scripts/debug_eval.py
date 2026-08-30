import requests, json

# Load first problem
with open('/root/etm/data/aime_2025.jsonl') as f:
    prob = json.loads(f.readline())

prompt = ('Problem: {problem}\n\n'
          'Please reason step by step, and put your final answer within \\boxed{{}}.\n'
          'Solution: Let').format(problem=prob['problem'])

print('=== PROMPT (last 200 chars) ===')
print(repr(prompt[-200:]))
print()

# Test 1: exact eval params (temp=0.8, stop=['Problem:', '\n\n\n'])
print('=== TEST 1: eval params (temp=0.8) ===')
r = requests.post('http://localhost:8000/v1/completions', json={
    'model': 'trained', 'prompt': prompt,
    'max_tokens': 2048, 'temperature': 0.8,
    'top_p': 0.95, 'stop': ['Problem:', '\n\n\n'],
}, timeout=300)
print('status:', r.status_code)
d = r.json()
if 'choices' in d:
    out = d['choices'][0]['text']
    print('output len:', len(out))
    print('finish_reason:', d['choices'][0].get('finish_reason'))
    print('output (first 500):', repr(out[:500]))
else:
    print('ERROR:', json.dumps(d, indent=2)[:500])

# Test 2: greedy (temp=0), no stop tokens
print()
print('=== TEST 2: greedy, no stop ===')
r2 = requests.post('http://localhost:8000/v1/completions', json={
    'model': 'trained', 'prompt': prompt,
    'max_tokens': 300, 'temperature': 0,
}, timeout=300)
d2 = r2.json()
if 'choices' in d2:
    out2 = d2['choices'][0]['text']
    print('output len:', len(out2))
    print('finish_reason:', d2['choices'][0].get('finish_reason'))
    print('output (first 500):', repr(out2[:500]))
else:
    print('ERROR:', json.dumps(d2, indent=2)[:500])

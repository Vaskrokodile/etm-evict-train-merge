import requests, json
prompt = 'Problem: What is 7 + 5?\n\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution: Let'
r = requests.post('http://localhost:8000/v1/completions', json={
    'model': 'trained', 'prompt': prompt,
    'max_tokens': 200, 'temperature': 0,
    'stop': ['Problem:', '\n\n\n'],
}, timeout=120)
print('status', r.status_code)
d = r.json()
print(repr(d['choices'][0]['text']))

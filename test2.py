"""Single-call smoke test for the gateway routing path. ~$0.0001."""
from models import call_model

out = call_model('llama-3.1-8b', 'Reply with exactly: HI', max_tokens=10)
print(repr(out))
assert 'HI' in out.upper(), f'unexpected output: {out!r}'
print('PASS')
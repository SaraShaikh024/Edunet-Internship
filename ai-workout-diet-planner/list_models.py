import os
import google.generativeai as genai

api_key = os.environ.get('GOOGLE_API_KEY')
if not api_key:
    import toml
    data = toml.load('.streamlit/secrets.toml')
    api_key = data.get('GOOGLE_API_KEY')

print('using key prefix', api_key[:6] if api_key else None)

if not api_key:
    print('no key configured')
    exit(1)

genai.configure(api_key=api_key)
models = genai.list_models()
print('models:')
for m in models:
    try:
        print(' -', m.name, 'features=', getattr(m, 'features', None))
    except Exception as e:
        print(' - model object repr', m)

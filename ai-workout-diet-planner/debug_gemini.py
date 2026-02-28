import os
import google.generativeai as genai

api_key = None
try:
    # simulate streamlit secrets by reading secrets file if exists
    import toml
    secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
    if os.path.exists(secrets_path):
        data = toml.load(secrets_path)
        api_key = data.get('GOOGLE_API_KEY')
except Exception as e:
    pass
if not api_key:
    api_key = os.environ.get('GOOGLE_API_KEY')
print('api_key used:', api_key)

if not api_key:
    print('no key found');
    exit(1)

try:
    genai.configure(api_key=api_key)
    print('configured genai')
except Exception as e:
    print('configuration error', e)

models_to_try = ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro', 'models/gemini-pro']
for m in models_to_try:
    try:
        print('trying model', m)
        model = genai.GenerativeModel(m)
        print('model created', model)
        resp = model.generate_content('test')
        print('response text:', resp.text[:200])
        break
    except Exception as e:
        print('model', m, 'failed with', e)

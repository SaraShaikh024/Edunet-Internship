import google.generativeai as genai
print('module version', genai.__version__ if hasattr(genai, '__version__') else '(no version)')
print('attributes:', [name for name in dir(genai) if not name.startswith('_')])
print('has GenerativeModel', hasattr(genai, 'GenerativeModel'))
print('has generate_content', hasattr(genai, 'generate_content'))
print('has GenerativeModel?', hasattr(genai, 'GenerativeModel'))
print('dir GenerativeModel', dir(genai.GenerativeModel) if hasattr(genai, 'GenerativeModel') else [])

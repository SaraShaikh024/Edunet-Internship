import os
import google.generativeai as g

print('env key:', os.getenv('GOOGLE_API_KEY'))
try:
    g.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    print('configured')
    # try a simple request
    resp = g.generate_content('Hello')
    print('response snippet', resp.text[:100])
except Exception as e:
    print('error configuring or calling:', e)

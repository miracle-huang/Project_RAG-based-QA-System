import urllib.request
import os
import ssl

os.makedirs('images', exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'https://www.araya.org/wp-content/uploads/2024/10/rag_overview.png'
urllib.request.urlretrieve(url, 'images/rag_overview.png')
print('Downloaded successfully!')
print('File size:', os.path.getsize('images/rag_overview.png'), 'bytes')

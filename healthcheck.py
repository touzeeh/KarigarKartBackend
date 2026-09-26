import os
import urllib.request

if __name__ == '__main__':
    print('KarigarKart backend build configuration OK')
    print('PORT:', os.getenv('PORT', '8000'))

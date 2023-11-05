import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'requirements/vgg16.pth')

IPADDR = 'http://127.0.0.1:5000'

IMAGE_DIR = os.path.join(BASE_DIR, 'static/flower102')

IMAGE_BASE_URL = IPADDR + '/static/flower102'

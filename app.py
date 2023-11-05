from flask import Flask
from flask_cors import CORS
from blueprints import bp
from utils import global_model

app = Flask(__name__)
app.config.from_pyfile('config.py')
app.register_blueprint(bp)
CORS(app=app, supports_credentials=True)

global_model.global_init(app.config)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

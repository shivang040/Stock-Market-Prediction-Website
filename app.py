from flask import Flask, render_template
from flask_restful import Api
from flask_cors import CORS

from resources.predict import Predict

app = Flask(__name__)
CORS(app)
api = Api(app)


@app.route('/')
def home():
    return render_template('index.html')

api.add_resource(Predict, '/predict/<string:name>')

if __name__ == '__main__':
    app.run(port=5000, debug=True)

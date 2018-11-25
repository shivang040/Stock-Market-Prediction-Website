from flask_restful import Resource
from models.lstm import LSTM_Model


class Predict(Resource):
    def get(self, name):
        predict, original = LSTM_Model.LSTM_Pred(name)
        return {'predicted': '{}'.format(predict), 'original': '{}'.format(original)}, 200

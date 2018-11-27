from flask_restful import Resource
from models.lstm import LSTM_Model


class Predict(Resource):
    def get(self, name):
        predict, original, tomm_pred, mse = LSTM_Model.LSTM_Pred(name)
        return {'predicted': '{}'.format(predict),
                'original': '{}'.format(original),
                'tommrw_prdctn': '{}'.format(tomm_pred),
                'mn_sqre_err': '{}'.format(mse)}, 200

import pandas as pd
import pickle
import os
import lightgbm

class ScoringService(object):
    @classmethod
    def get_model(cls, model_path, inference_df, inference_log):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            inference_df: Past data not subject to prediction.
            inference_log: Past log data that is not subject to prediction.

        Returns:
            bool: The return value. True for success.
        """
        
        # model_path = '../model/model.pkl'
        model_path = os.path.join(model_path, 'model.pkl')

        print(model_path)
        with open(model_path, 'rb') as f:
            cls.model = pickle.load(f)
        
        # cls.model = None
        cls.data = inference_df
        cls.log_pathes = inference_log

        return True


    @classmethod
    def predict(cls, input, input_log):
        """Predict method

        Args:
            input: meta data of the sample you want to make inference from (DataFrame)
            input_log: meta data of the sample you want to make inference from (DataFrame)

        Returns:
            prediction: Inference for the given input. Return columns must be ['datetime', 'start_code', 'end_code', 'KP'](DataFrame).
        """

        prediction = input.copy()
        prediction = prediction[['datetime', 'start_code', 'end_code', 'KP']]
        df_input = input.copy()
        
        df_input['weekday'] = pd.to_datetime(df_input['datetime'], format='%Y-%m-%d').apply(lambda x: x.weekday())
        columns_input = ['start_code', 'end_code', 'KP', 'OCC', 'allCars', 'speed', 'weekday']
        
        # model_path = '../model/model.pkl'

        model_input = df_input[columns_input]

        # with open(model_path, 'rb') as f:
        #     model = pickle.load(f)


        y_pred = cls.model.predict(model_input)
        # print(y_pred)
        prediction['prediction_speed'] = y_pred
        # prediction['prediction'] = prediction.eval('prediction_speed < 40')
        prediction['prediction'] = prediction['prediction_speed'].apply(lambda x: 1 if x < 40.0 else 0)
        print(prediction['prediction'].sum())
        # prediction.drop(index=['prediction_speed'], inplace=True)
        #prediction['prediction'] = prediction['weekday'].apply(lambda x: 1 if x==4 or x==5 else 0)
        prediction = prediction[['datetime', 'start_code', 'end_code', 'KP', 'prediction']]

        return prediction

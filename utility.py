# -*- coding: utf-8 -*-
"""
@author: Syed Hasib Akhter Faruqui
@email : syed-hasb-akhter.faruqui@my.utsa.edu
Website: www.shafnehal.com
"""
# Import libraries
from tensorflow.keras.models import model_from_json
from sklearn.externals import joblib

# Define necessary functions
# Load Weight
def load_model(StudyGroup):
    # Load Model
    json_file = open(StudyGroup + "_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    print('Model Loaded!)
    # load weight
    model.load_weights(StudyGroup + "_model_weight.h5")
    print('Weights Loaded!\nStudy Group:' + StudyGroup)
    return model

# Load Scale
def Load_Scale(scalerfile):
    # Load the data Scale
    scale = joblib.load(scalerfile + "_model.sav")
    print("Scale file Loading Complete!!")
    return scale
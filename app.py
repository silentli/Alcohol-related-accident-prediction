#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:40:08 2022

@author: guolee
"""

#%% Import packages

from flask import Flask,jsonify,request
import pickle


#%% Import packages
app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))


@app.route('/')
def index():
    return 'hello!!'

@app.route('/predict', methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    insertValues = request.get_json()
    x1=str(insertValues['year'])
    x2=str(insertValues['month'])
    input = x1 + '-' + x2
    
    result = model.predict(start=input,end=input).round()
    #return jsonify('prediction')
    return jsonify({'prediction': int(result)})

if __name__ == '__main__':
    app.run(debug=True)
    



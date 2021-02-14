# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:47:38 2021

@author: Suraj
"""
from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
import distance
from nltk.stem import PorterStemmer
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import scipy
import xgboost as xgb
from fuzzywuzzy import fuzz
import pickle
import joblib
import gensim
import smart_open
from gensim.models import Word2Vec
from string import punctuation
stop_words = stopwords.words('english')

app = Flask(__name__)

with open('XGBoostC2.pkl', 'rb') as file: #load model as pickle file
    pickle_model = pickle.load(file) 

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

def sent2vec(s):
    print('inside the sent2vec')
    words = str(s).lower() 
    words = word_tokenize(words) 
    words = [w for w in words if not w in stop_words] 
    words = [w for w in words if w.isalpha()] 
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M) 
    v = M.sum(axis=0) 
    return v / np.sqrt((v ** 2).sum())

def feature_preparation(question1,question2):
    print('inside feature preparation')
    a = {'question1': question1, 'question2': question2}
    testdf = pd.DataFrame(a,columns = ['question1','question2'],index=[0])
    print('created dataframe')
    
    question1_vectors = np.zeros((testdf.shape[0], 300))
    for i, q in enumerate(testdf.question1.values):
        question1_vectors[i, :] = sent2vec(q)
    
    question2_vectors  = np.zeros((testdf.shape[0], 300)) 
    for i, q in enumerate(testdf.question2.values):
        question2_vectors[i, :] = sent2vec(q)
    
    print('question vectors done')
    
    testdf['len_q1'] = testdf.question1.apply(lambda x: len(str(x)))
    testdf['len_q2'] = testdf.question2.apply(lambda x: len(str(x)))
    testdf['common_words'] = testdf.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
    testdf['fuzz_ratio'] = testdf.apply(lambda x: fuzz.ratio(str(x['question1']), str(x['question2'])), axis=1)
    testdf['fuzz_partial_ratio'] = testdf.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
    testdf['fuzz_partial_token_set_ratio'] = testdf.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    testdf['fuzz_partial_token_sort_ratio'] = testdf.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    testdf['fuzz_token_set_ratio'] = testdf.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    testdf['fuzz_token_sort_ratio'] = testdf.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    testdf['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    testdf['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    testdf['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    testdf['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    testdf['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    testdf['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    testdf['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    
    testdf.drop(['question1','question2'],axis = 1, inplace = True)

    return testdf
    
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    print('inside predict')
    if(request.method == 'POST'):
        question1 = request.form['message1']
        question2 = request.form['message2']
        predictdf = feature_preparation(question1,question2)
        result = pickle_model.predict(predictdf)
        return render_template('result.html',prediction = result)
        
        
if __name__ == '__main__':
	app.run(debug=True)
        
import pandas as pd
import numpy as np
df= pd.read_csv('spam_ham_dataset.csv')
df.sort_values(by=['label_num'], inplace=True, ascending=False)
df = df[:2*len(df[df["label_num"]==1])]
df =df.reset_index(drop=True)
df['label_num'].mean()
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
def filter_words(text):
    words = text.split()
    stop_words = set(stopwords.words("english"))
    non_alphabet_pattern = re.compile(r'[^a-zA-Z]')
    filtered_words = [word for word in words if word not in stop_words]
    filtered_words = [re.sub(non_alphabet_pattern, '', word) for word in filtered_words]
    filtered_words = list(filter(None, filtered_words))
    return filtered_words
words_dict=[]
words_ =[]
for i in range(len(df)):
    f =filter_words(df['text'][i])
    words_dict.extend(f)
    words_.append(f)
df['words'] = words_
words_dict = list(set(words_dict))
words_ =None
print(words_dict)
words_dict['id']=words_dict.index
dict_ = words_dict.to_dict('list')
words_dict =None
dict_
np.random.seed(0)
df = df.sample(frac=1).reset_index(drop=True)
train_data = df[:int(len(df)*0.8)].reset_index(drop=True)
test_data = df[int(len(df)*0.8):].reset_index(drop=True)
train_data_0 = train_data[train_data['label_num']==0].reset_index(drop=True)
train_data_1 = train_data[train_data['label_num']==1].reset_index(drop=True)

train_features_0 = {}
for i in dict_['id']:
    train_features_0[i] =[]
for i in range(len(train_data_0)):
    words_ =np.array(train_data_0['words'][i])
    for word in set(words_):
        if word in dict_[0]:
            freq = (words_==word).sum()
            id_ = dict_['id'][dict_[0].index(word)]
            train_features_0[id_].append(freq)


train_features_1 = {}
for i in dict_['id']:
    train_features_1[i] =[]
for i in range(len(train_data_1)):
    words_ =np.array(train_data_1['words'][i])
    for word in set(words_):
        if word in dict_[0]:
            freq = (words_==word).sum()
            id_ = dict_['id'][dict_[0].index(word)]
            train_features_1[id_].append(freq)
mean_var_0 = {}
for i in dict_['id']:
    if train_features_0[i]==[]:
        mean_var_0[i] =[0.0001,0.0001]
        continue
    mean = np.mean(train_features_0[i])
    var = np.var(train_features_0[i])
    mean_var_0[i] =[mean+0.0001,var+0.0001]

mean_var_1 = {}
for i in dict_['id']:
    if train_features_1[i]==[]:
        mean_var_1[i] =[0.0001,0.0001]
        continue
    mean = np.mean(train_features_1[i])
    var = np.var(train_features_1[i])
    mean_var_1[i] =[mean+0.0001,var+0.0001]
mean_var_0 = {}
for i in dict_['id']:
    if train_features_0[i]==[]:
        mean_var_0[i] =[0.0001,0.0001]
        continue
    mean = np.mean(train_features_0[i])
    var = np.var(train_features_0[i])
    mean_var_0[i] =[mean+0.0001,var+0.0001]

mean_var_1 = {}
for i in dict_['id']:
    if train_features_1[i]==[]:
        mean_var_1[i] =[0.0001,0.0001]
        continue
    mean = np.mean(train_features_1[i])
    var = np.var(train_features_1[i])
    mean_var_1[i] =[mean+0.0001,var+0.0001]
mean_var_0 = {}
for i in dict_['id']:
    if train_features_0[i]==[]:
        mean_var_0[i] =[0.0001,0.0001]
        continue
    mean = np.mean(train_features_0[i])
    var = np.var(train_features_0[i])
    mean_var_0[i] =[mean+0.0001,var+0.0001]

mean_var_1 = {}
for i in dict_['id']:
    if train_features_1[i]==[]:
        mean_var_1[i] =[0.0001,0.0001]
        continue
    mean = np.mean(train_features_1[i])
    var = np.var(train_features_1[i])
    mean_var_1[i] =[mean+0.0001,var+0.0001]
def gaussian_prob_0(x, id_):
    mean = mean_var_0[id_][0]
    var = mean_var_0[id_][1]
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * var)))
    return coeff * exponent + 0.00001

def gaussian_prob_1(x, id_):
    mean = mean_var_1[id_][0]
    var = mean_var_1[id_][1]
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * var)))
    return coeff * exponent + 0.00001
def pois_prob_0(x, id_):
    mean = mean_var_0[id_][0]
    return np.exp(-mean)*np.power(mean,x)/np.math.factorial(x)

def pois_prob_1(x, id_):
    mean = mean_var_1[id_][0]
    return np.exp(-mean)*np.power(mean,x)/np.math.factorial(x)
def bernoulli_prob_0(x, id_):
    mean = mean_var_0[id_][0]
    return np.power(mean,x)*np.power(1-mean,1-x)

def bernoulli_prob_1(x, id_):
    mean = mean_var_1[id_][0]
    return np.power(mean,x)*np.power(1-mean,1-x)
def multino_prob_0(x, id_):
    mean = mean_var_0[id_][0]
    return np.power(mean,x)

def multino_prob_1(x, id_):
    mean = mean_var_1[id_][0]
    return np.power(mean,x)
p_0 = len(train_data_0)/len(train_data)
p_1 = len(train_data_1)/len(train_data)
print(p_0,p_1)

pred =[]
for i in range(len(test_data)):
    prob_0 =1
    prob_1 =1
    for word in test_data['words'][i]:
        id_ = dict_[0].index(word)
        freq = (np.array(test_data['words'][i])==word).sum()
        prob_0 *= gaussian_prob_0(freq, id_)
        prob_1 *= gaussian_prob_1(freq, id_)
    prob_0 = (p_0*prob_0)/((p_0*prob_0)+(p_1*prob_1))
    prob_1 = (p_1*prob_1)/((p_0*prob_0)+(p_1*prob_1))

    if prob_0>prob_1:
        pred.append(0)
    else :
        pred.append(1)
print("Accuracy: ", (np.array(pred)==np.array(test_data['label_num'])).sum()/len(test_data))
import scipy.special

# ...

def pois_prob_0(x, id_):
    mean = mean_var_0[id_][0]
    return np.exp(-mean) * np.power(mean, x) / scipy.special.factorial(x)

def pois_prob_1(x, id_):
    mean = mean_var_1[id_][0]
    return np.exp(-mean) * np.power(mean, x) / scipy.special.factorial(x)

# ...
print("Accuracy: ", (np.array(pred)==np.array(test_data['label_num'])).sum()/len(test_data))
pred =[]
for i in range(len(test_data)):
    prob_0 =1
    prob_1 =1
    for word in test_data['words'][i]:
        id_ = dict_[0].index(word)
        freq = (np.array(test_data['words'][i])==word).sum()
        prob_0 *= bernoulli_prob_0(freq, id_)
        prob_1 *= bernoulli_prob_1(freq, id_)
    prob_0 = (p_0*prob_0)/((p_0*prob_0)+(p_1*prob_1))
    prob_1 = (p_1*prob_1)/((p_0*prob_0)+(p_1*prob_1))

    if prob_0>prob_1:
        pred.append(0)
    else :
        pred.append(1)
print("Accuracy: ", (np.array(pred)==np.array(test_data['label_num'])).sum()/len(test_data))
pred =[]
for i in range(len(test_data)):
    prob_0 =1
    prob_1 =1
    for word in test_data['words'][i]:
        id_ = dict_[0].index(word)
        freq = (np.array(test_data['words'][i])==word).sum()
        prob_0 *= multino_prob_0(freq, id_)
        prob_1 *= multino_prob_1(freq, id_)
    prob_0 = (p_0*prob_0)/((p_0*prob_0)+(p_1*prob_1))
    prob_1 = (p_1*prob_1)/((p_0*prob_0)+(p_1*prob_1))

    if prob_0>prob_1:
        pred.append(0)
    else :
        pred.append(1)
print("Accuracy: ", (np.array(pred)==np.array(test_data['label_num'])).sum()/len(test_data))

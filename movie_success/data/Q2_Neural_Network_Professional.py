import numpy as np
import pandas as pd
import re
import sys
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from math import log10
from nltk.tokenize import RegexpTokenizer
from pandas.core import datetools
import seaborn
seaborn.set()

def corr_points(x):
    oscar_win = re.search(r'Won (\d+) Oscar', x)
    oscar_nominated = re.search(r'Nominated for (\d+) Oscar', x)
    GoldenG_win = re.search(r'Won (\d+) Golden', x)
    GoldenG_nominated = re.search(r'Nominated for (\d+) Golden', x)
    minor_wins = re.search(r'(\d+) win', x)
    minor_nominations = re.search(r'(\d+) nomination', x)
    score = 0
    if oscar_win:
        score = score + 10 * int(oscar_win.group(1))
    if oscar_nominated:
        score = score + 5 * int(oscar_nominated.group(1))
    if GoldenG_win:
        score = score + 7 * int(GoldenG_win.group(1))
    if GoldenG_nominated:
        score = score + 3 * int(GoldenG_nominated.group(1))
    if minor_wins:
        score = score + int(minor_wins.group(1))
    if minor_nominations:
        score = score + 0.25 * int(minor_nominations.group(1))
    return score

def transfer_to_number(x): 
    # transfer a dataframe column to number matrix
    types =[]
    type_size = 0
    for i in range(len(x)):
        for j in x[i]:
            if j not in types:
                types.append(j)
                type_size = type_size + 1
                
    two_D_array = [x[:] for x in [[0] * len(types)] * len(x)] 
    
    for i in range(len(x)):
        for j in x[i]:
            index = 0
            while j != types[index]:
                index = index + 1
            two_D_array[i][index] = 1        
    
    print('There are ', len(two_D_array[0]), ' different types in total.')
    print('There are ', len(two_D_array), ' records collected.')
    
    return two_D_array

def remove_rare_data(x,y,z):
    # remove the elements in attribute 'y', which occurs fewer than 'z' times ,from dataframe 'x'
    x_size = len(x[y])
    types = []
    times = []
    x1 = []
    y1 =[]
    mark = [[] for i in range(len(x[y]))]
    for i in range(x_size):
        for j in range(len(x[y][i])):
                if x[y][i][j] not in types:
                    types.append(x[y][i][j])
                    times.append(1)
                    x1.append([i])
                    y1.append([j])
                else:
                    index = types.index(x[y][i][j])
                    times[index] = times[index] + 1 
                    x1[index].append(i)
                    y1[index].append(j)
    for i in range(len(times)):
        if times[i] <= z:
            for j in range(times[i]):
                mark[x1[i][j]].append(y1[i][j])
    for i in range(x_size):
        if len(mark[i]) != 0:
            for j in sorted(mark[i], reverse=True):
                del x[y][i][j] 
    for i in range(x_size):
        if len(x[y][i]) == 0:
            x = x.drop(i)
    x = x.reset_index(drop = True)
    print('Remove rare elements in ', y,  ' successfully')
    return x

def remove_rare_data_array(x, y):
    measure = y
    
    # basically the same process as transfer_to_number() function
    types =[]
    type_size = 0
    for i in range(len(x)):
        for j in x[i]:
            if j not in types:
                types.append(j)
                type_size = type_size + 1
                
    two_D_array = [x[:] for x in [[0] * len(types)] * len(x)] 
    
    for i in range(len(x)):
        for j in x[i]:
            index = 0
            while j != types[index]:
                index = index + 1
            two_D_array[i][index] = 1    

    sum = [0]* len(two_D_array[0])
    # print(sum)
    for i in range(len(two_D_array)):
        for j in range(len(two_D_array[0])):   
            if two_D_array[i][j] == 1:
                # if there is a score (record) on this genre, mark it
                sum[j] = sum[j] + 1
    
    x_more_than_y_times = len(types)
    
    # if that element (e.g. director) appears only once, we set the whole column to be -1 
    for i in range(len(sum)):
        if sum[i] <= measure:
            x_more_than_y_times = x_more_than_y_times - 1
            for j in range(len(two_D_array)):
                two_D_array[j][i] = -1
    
    # delete the whole -1 column
    for i in range(len(two_D_array)):
        two_D_array[i] = [e for e in two_D_array[i] if e not in (-2, -1)]
    
    # remove movies with all 0's in a row. (No director)
    two_D_array = [v for v in two_D_array if sum(v) != 0]
    return two_D_array

def extract_first_2(x):
    if len(x) <= 2:
        return x
    else:
        return x[:2]
    
def extract_first_3(x):
    if len(x) <= 3:
        return x
    else:
        return x[:3]

def extract_first_4(x):
    if len(x) <= 4:
        return x
    else:
        return x[:4]

def to_list(x):
    return [x]

def combine_list(x,y):
    #combine two 2d lists together
    for i in range(len(x)):
        x[i].extend(y[i])
    return x

def model_test(model,X_test,y_test,a):
    # Using model "model" and input data "X_test", "y_test"
    accept = 0
    real = np.asarray(y_test, dtype="float64")
    predict = np.asarray(model.predict(X_test), dtype="float64")
    difference = np.absolute(real - predict)
    for i in range(real.size):
        if difference[i] <= a:
            accept = accept + 1
    score = accept/real.size
    print ('The corrected score for testing data is: ', score)
    return 0


def main(in_directory1,in_directory2,in_directory3):
	wiki = pd.read_json(in_directory1, orient='record', lines=True)
	rt = pd.read_json(in_directory2, orient='record', lines=True)
	omdb = pd.read_json(in_directory3, orient='record', lines=True)

	data_temp = wiki.merge(rt, on = 'rotten_tomatoes_id')
	data = data_temp.merge(omdb, left_on = 'imdb_id_x', right_on = 'imdb_id')
	# data = data.dropna(subset = ['nbox', 'ncost', 'publication_date', 'cast_member','director' ,'production_company', 'audience_average', 'audience_percent','critic_average', 'critic_percent'])
	# data.info()
	# Have a look on each attribute in this dataframe


	# Begin the filtering:

	# weighted points is associated with a movie's PROFESSIONAL SUCCESS
	# this point is based on weights among oscar, golden globe and other wins / nominations
	data['weighted_points'] = data['omdb_awards'].map(corr_points)

	# Extract the information that we really need, and drop N/A values
	professional = data[['director','cast_member','genre','production_company','weighted_points']]
	professional = professional.dropna(subset = ['director','cast_member','genre','production_company','weighted_points']).reset_index(drop = True)



	# We only focus on main values, such as main directors, protagonists and major genres 
	professional['director'] = professional['director'].apply(extract_first_2)
	professional['cast_member'] = professional['cast_member'].apply(extract_first_4)
	professional['genre'] = professional['genre'].apply(extract_first_3)
	# Filtering down.Could have a look:
	# print(professional)

	# Remove elements that appear only once
	professional = remove_rare_data(professional, 'cast_member', 1)
	professional = remove_rare_data(professional, 'director', 1)


	# transfer to list so that we can further transfer to number list
	professional['production_company'] = professional['production_company'].apply(to_list)


	# 
	company = transfer_to_number(professional['production_company'])
	director = transfer_to_number(professional['director'])
	cast_member= transfer_to_number(professional['cast_member'])
	genre = transfer_to_number(professional['genre'])


	X = combine_list(director,cast_member)
	X = combine_list(X, genre)
	X = combine_list(X, company)
	y = np.asarray(professional['weighted_points'], dtype="|S6")


	# User Neural Network to test
	X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y)
	model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(25,15,5))
	model.fit(X_train_2, y_train_2)
	model.predict(X_test_2)
	model_test(model, X_test_2,y_test_2,5)
	training_score = model.score(X_train_2, y_train_2)
	print('The score for training data is: ', training_score)


if __name__=='__main__':
    	in_directory1 = sys.argv[1]
    	in_directory2 = sys.argv[2]
    	in_directory3 = sys.argv[3]
    	main(in_directory1,in_directory2,in_directory3)


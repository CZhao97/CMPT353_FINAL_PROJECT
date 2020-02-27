import numpy as np
import pandas as pd
from scipy import stats
import sys
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import seaborn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import MultiLabelBinarizer
porter = PorterStemmer()
lancaster=LancasterStemmer()
seaborn.set()
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

def score_corr(model, X_test, y_test, df):
    accept = 0
    total = len(X_test)
    
    datafr = pd.DataFrame(y_test)
    index_list = datafr.reset_index()['index'].tolist()
    
    predict = model.predict(X_test)
    predict_list = predict.tolist()
    
    for i in range(len(X_test)):
        accept_list = []
        index = index_list[i]
        accept_list.append(df['genres_number'][index])
        if index > 1:
            index_m1 = index - 1
            index_m2 = index - 2
            index_p1 = index + 1
            index_p2 = index + 2
            if df['plot'][index] == df['plot'][index_m1]:
                accept_list.append(df['genres_number'][index_m1])
            if df['plot'][index] == df['plot'][index_m2]:
                accept_list.append(df['genres_number'][index_m2])
            if index_p1 <= max(index_list):
                if df['plot'][index] == df['plot'][index_p1]:
                    accept_list.append(df['genres_number'][index_p1])
            if index_p2 <= max(index_list):      
                if df['plot'][index] == df['plot'][index_p2]:
                    accept_list.append(df['genres_number'][index_p2])
        elif index == 1:
            index_m1 = index - 1
            index_p1 = index + 1
            index_p2 = index + 2
            if df['plot'][index] == df['plot'][index_m1]:
                accept_list.append(df['genres_number'][index_m1])
            if df['plot'][index] == df['plot'][index_p1]:
                accept_list.append(df['genres_number'][index_p1])
            if df['plot'][index] == df['plot'][index_p2]:
                accept_list.append(df['genres_number'][index_p2])
        else:
            index_p1 = index + 1
            index_p2 = index + 2
            if df['plot'][index] == df['plot'][index_p1]:
                accept_list.append(df['genres_number'][index_p1])
            if df['plot'][index] == df['plot'][index_p2]:
                accept_list.append(df['genres_number'][index_p2])
                
        if predict_list[i] in accept_list:
            accept = accept + 1
        
    score = accept / total
    # print ('The corrected score for testing data is: ', score)
    return score

def get_partition(number_movies):
    total = 0
    for x in number_movies:
        total += number_movies[x]
    for x in number_movies:
        number_movies[x] = number_movies[x]/total
    return number_movies

def get_genres(omdb):
    genres = []
    for i in omdb['omdb_genres']:
        for j in i:
            if j not in genres:
                genres.append(j)
    dic_genres = {}
    for i in genres:
        dic_genres.setdefault(i, [])

    for index,j in enumerate(omdb['omdb_genres']):
        for i in j:
            if i in dic_genres:
                dic_genres[i].append(omdb['omdb_plot'][index])
    return dic_genres

def each_number_genres(genres_comment):
    number_movies = {}
    for i in genres_comment:
        length = len(genres_comment[i])
        number_movies.setdefault(i,length)
    return number_movies

def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text

def stemSentence_porter(sentence):
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def token_plot_porter(omdb):
    omdb_temp = omdb.copy()
    for index,plot in enumerate(omdb_temp['omdb_plot']):
        omdb_temp['omdb_plot'][index] = stemSentence_porter(plot)
    return omdb_temp

def stemSentence_lancaster(sentence):
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(lancaster.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def token_plot_lancaster(omdb):
    omdb_temp = omdb.copy()
    for index,plot in enumerate(omdb_temp['omdb_plot']):
        omdb_temp['omdb_plot'][index] = stemSentence_lancaster(plot)
    return omdb_temp




def main ():
    omdb = pd.read_json(sys.argv[1], orient='record', lines=True)

    genres = []
    for i in omdb['omdb_genres']:
        for j in i:
            if j not in genres:
                genres.append(j)

    genres_comment = get_genres(omdb)

    number_movies = each_number_genres(genres_comment)

    filtered_movies = {}
    for i in number_movies:
        if number_movies[i] >= 800:
            filtered_movies.update({i: number_movies[i]})

    data = get_partition(filtered_movies)

    df = pd.DataFrame.from_dict(data,orient='index',columns = ['partition'])
    df.plot.pie(y='partition', figsize=(10, 15))

    filtered_genres = [x for x in filtered_movies]

    for index, genres in enumerate(omdb['omdb_genres']):
        temp = []
        for i in genres:
            if i in filtered_genres:
                temp.append(i)
        if temp != []:
            omdb['omdb_genres'][index] = temp
        else:
            omdb['omdb_genres'][index] = []

    omdb = omdb[omdb.astype(str)['omdb_genres'] != '[]'].reset_index(drop=True)

    omdb['omdb_plot'] = omdb['omdb_plot'].apply(lambda x: clean_text(x))

    omdb_porter = token_plot_porter(omdb)
    omdb_lancaster = token_plot_lancaster(omdb)

    merged_list = ''
    for i in omdb_lancaster['omdb_plot']:
        merged_list = merged_list + i + ' '
    tokenized_word = word_tokenize(merged_list)
    fdist = FreqDist(tokenized_word)


    fdist.plot(32,cumulative=False)
    # plt.show()

    stop_words=set(stopwords.words("english"))

    for index, plot in enumerate(omdb_porter['omdb_plot']):
        temp = plot.split()
        temp_list = ''
        for i in temp:
            if i not in stop_words:
                temp_list = temp_list + ' ' + i
        omdb_porter['omdb_plot'][index] = temp_list

    for index, plot in enumerate(omdb_lancaster['omdb_plot']):
        temp = plot.split()
        temp_list = ''
        for i in temp:
            if i not in stop_words:
                temp_list = temp_list + ' ' + i
        omdb_lancaster['omdb_plot'][index] = temp_list

    filtered_sent_lancaster=[]
    for w in tokenized_word:
        if w not in stop_words:
            filtered_sent_lancaster.append(w)


    merged_list = ''
    for i in omdb_porter['omdb_plot']:
        merged_list = merged_list + i + ' '
    tokenized_word = word_tokenize(merged_list)
    fdist = FreqDist(tokenized_word)

    fdist.plot(32,cumulative=False)
    # plt.show()

    filtered_sent_porter=[]
    for w in tokenized_word:
        if w not in stop_words:
            filtered_sent_porter.append(w)

    dic_saving_genres = {}
    for index, genres in enumerate(filtered_genres):
        dic_saving_genres.update({genres: index})

    genres_comment = get_genres(omdb)

    predict_df = pd.DataFrame(columns=['genres_number', 'plot'])

    for index,genres in enumerate(omdb['omdb_genres']):
        for i in genres:
            row = [dic_saving_genres[i],omdb['omdb_plot'][index]]
            predict_df.loc[len(predict_df)] = row


    selector = SelectPercentile(f_classif, percentile=20)
    vectorizer = TfidfVectorizer(max_df = 0.4,stop_words='english')
    y = predict_df['genres_number'].astype('int')
    X = predict_df['plot']
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    X_train = vectorizer.fit_transform(X_train)
        
    selector.fit(X_train,y_train)
    X_train = selector.transform(X_train).toarray()
        
    model = MultinomialNB(alpha=1 ,fit_prior = False)
    model.fit(X_train,y_train)
    X_test = vectorizer.transform(X_test)
    X_test = selector.transform(X_test).toarray()



    tfid_for_each_genre_of_plot_without_stem = model.score(X_test,y_test)

    tfid_for_each_genre_of_plot_scored_by_ourselves_without_stem = score_corr(model, X_test, y_test, predict_df)

    predict_df_porter = pd.DataFrame(columns=['genres_number', 'plot'])
    for index,genres in enumerate(omdb_porter['omdb_genres']):
        for i in genres:
            row = [dic_saving_genres[i],omdb_porter['omdb_plot'][index]]
            predict_df_porter.loc[len(predict_df_porter)] = row

    selector = SelectPercentile(f_classif, percentile=20)
    vectorizer = TfidfVectorizer(max_df = 0.4,stop_words='english')
    y = predict_df_porter['genres_number'].astype('int')
    X = predict_df_porter['plot']
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    X_train = vectorizer.fit_transform(X_train)
        
    selector.fit(X_train,y_train)
    X_train = selector.transform(X_train).toarray()
        
    model = MultinomialNB(alpha=1 ,fit_prior = False)
    model.fit(X_train,y_train)
    X_test = vectorizer.transform(X_test)
    X_test = selector.transform(X_test).toarray()
    tfid_for_each_genre_of_plot_with_stem = model.score(X_test,y_test)

    tfid_for_each_genre_of_plot_scored_by_ourselves_with_stem = score_corr(model, X_test, y_test, predict_df_porter)


    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(omdb_porter['omdb_genres'])
    y = multilabel_binarizer.transform(omdb_porter['omdb_genres'])
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000, stop_words = 'english')

    X_train, X_test, y_train, y_test = train_test_split(omdb_porter['omdb_plot'], y, test_size=0.2, random_state=9)

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    lr = LogisticRegression()
    clf = OneVsRestClassifier(lr)

    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)

    MultiLabelBinarizer_tfid_with_stem = f1_score(y_test, y_pred, average="micro")


    y_pred_prob = clf.predict_proba(X_test_tfidf)

    t = 0.3 
    y_pred_new = (y_pred_prob >= t).astype(int)

    MultiLabelBinarizer_tfid_with_stem_and_probability = f1_score(y_test, y_pred_new, average="micro")


    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(omdb_porter['omdb_genres'])
    y = multilabel_binarizer.transform(omdb_porter['omdb_genres'])


    cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1))

    X_train, X_test, y_train, y_test = train_test_split(omdb_porter['omdb_plot'], y, test_size=0.2, random_state=9)

    X_train_tfidf = cv.fit_transform(X_train)
    X_test_tfidf = cv.transform(X_test)

    lr = LogisticRegression()
    clf = OneVsRestClassifier(lr)

    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)

    MultiLabelBinarizer_count_with_stem = f1_score(y_test, y_pred, average="micro")


    y_pred_prob = clf.predict_proba(X_test_tfidf)

    t = 0.3
    y_pred_new = (y_pred_prob >= t).astype(int)

    MultiLabelBinarizer_count_with_stem_and_probability = f1_score(y_test, y_pred_new, average="micro")

    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(omdb['omdb_genres'])
    y = multilabel_binarizer.transform(omdb['omdb_genres'])
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000, stop_words = 'english')

    X_train, X_test, y_train, y_test = train_test_split(omdb['omdb_plot'], y, test_size=0.2, random_state=9)

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    lr = LogisticRegression()
    clf = OneVsRestClassifier(lr)

    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)

    MultiLabelBinarizer_tfid_without_stem = f1_score(y_test, y_pred, average="micro")

    y_pred_prob = clf.predict_proba(X_test_tfidf)

    t = 0.3 # threshold value
    y_pred_new = (y_pred_prob >= t).astype(int)

    MultiLabelBinarizer_tfid_without_stem_and_probability = f1_score(y_test, y_pred_new, average="micro")


    # https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/
    
    print('tfid_for_each_genre_of_plot_without_stem=',tfid_for_each_genre_of_plot_without_stem)
    print('tfid_for_each_genre_of_plot_scored_by_ourselves_without_stem=',tfid_for_each_genre_of_plot_scored_by_ourselves_without_stem)
    print('tfid_for_each_genre_of_plot_with_stem=',tfid_for_each_genre_of_plot_with_stem)
    print('tfid_for_each_genre_of_plot_scored_by_ourselves_with_stem=',tfid_for_each_genre_of_plot_scored_by_ourselves_with_stem)
    print('MultiLabelBinarizer(list of genres)_tfid_with_stem=',MultiLabelBinarizer_tfid_with_stem)
    print('MultiLabelBinarizer(list of genres)_tfid_with_stem_and_probability_estimate=',MultiLabelBinarizer_tfid_with_stem_and_probability)
    print('MultiLabelBinarizer(list of genres)_count_with_stem=',MultiLabelBinarizer_count_with_stem)
    print('MultiLabelBinarizer(list of genres)_count_with_stem_and_probability_estimate=',MultiLabelBinarizer_count_with_stem_and_probability)
    print('MultiLabelBinarizer(list of genres)_tfid_without_stem=',MultiLabelBinarizer_tfid_without_stem)
    print('MultiLabelBinarizer(list of genres)_tfid_without_stem_and_probability_estimate=',MultiLabelBinarizer_tfid_without_stem_and_probability)

    print('For result with plot and graph, please run NLP.ipynb')
if __name__ == '__main__':
    main()

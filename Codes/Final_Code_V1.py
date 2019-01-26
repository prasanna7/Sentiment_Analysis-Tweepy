# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 21:31:49 2018

@author: Rohit Ladsaria
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import string
from wordcloud import WordCloud, STOPWORDS 
from nltk.tokenize import word_tokenize
from PIL import Image
import random
import csv
import tweepy
import collections
import nltk
from nltk.classify import NaiveBayesClassifier
import matplotlib.dates as mdates
from PIL import Image



####input your credentials here
consumer_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_token_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

##### Best buy
# Open/Create a file to append data
csvFile = open('bb.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q=["bestbuy","airpods"],count=100,
                           lang="en",
                           since="2018-07-07").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
csvFile.close()

##########################Scraping Ends##############

##########Get the data into Python##########
temp = 'bb.csv'
file = temp
filename = temp
#file = pd.read_csv(filename)

###################End################

############Wordcloud###############

#file = 'Bestbuy and apple.csv'
src_terms = 'bestbuy best buy apple fatkiddeals'

def clean_tweets(file, src_terms):
        
    # making some exclusion catagories, search terms, general, numbers
    
    exclude= ["s", 'https', "'re", "'s", "n't", "'", 'co', 'head', 'check',
              'latest', 'available', 'time', 'shop', 'promo', 'code']
    num = [str(num) for num in np.array(range(10000))]
    
    stop_words = set(stopwords.words('english'))
        
    search_terms = src_terms.split()
    
    #import file
    df = pd.read_table(file, delimiter=',', usecols=[1], 
                       header=None, names=['tweet']) 
    
    s = pd.Series(df['tweet']).dropna()
    
    # tokenize into individual words
    tokenized = s.apply(lambda x: word_tokenize(x))
    
    # exclude unwanted items
    lst = [word.lower() for x in tokenized for word in x]
    
    lst = [word for word in lst if word not in stop_words]
    
    rm_lst = [word for word in lst 
               if any(char in set(string.punctuation) for char in word)]
    
    lst = [word for word in lst if word not in rm_lst]
    
    lst = [word for word in lst if word not in exclude]
    
    rm_lst = [word for word in lst
               if any(char in num for char in word)]
    
    lst = [word for word in lst if word not in rm_lst]
    lst = [word for word in lst if word not in search_terms]
    
    return lst


def word_cloud(lst):
    
    string = " ".join(lst)
        
    def grey_color_func(word, font_size, position, orientation, random_state=None,
                        **kwargs):
        return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)
    
    mask = np.array(Image.open("twitter_mask.png"))
    
    stopwords = set(STOPWORDS)
    stopwords.add("int")
    stopwords.add("ext")
    
    wc = WordCloud(max_words=75, mask=mask, stopwords=stopwords, margin=10,
                   random_state=1).generate(string)
    
    plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
               interpolation="bilinear")
    plt.axis("off")
    plt.figure()
    return plt.show()
#############End Word Cloud#########################

############# #tag association #####################

def find_strip_pos(word):
    pos = word.find('\\')
    if pos == -1:
        return ''.join(e for e in word if e.isalnum())
    else:
        return ''.join(e for e in word[:pos] if e.isalnum())


def hashtag_association(filename, list_search_hashtags=[]):
    file = pd.read_csv(filename, header=None)
    file.columns = ['Date', 'Tweets']
    list_search_hashtags = [x.lower() for x in list_search_hashtags]
    file = file[file['Tweets'].notnull()]
    hashtag_list = []
    for index, rows in file.iterrows():
        a_tweet = file['Tweets'][index]
        temp_list = a_tweet.split()
        for j in range(len(temp_list)):
            if (temp_list[j].startswith('#')) & (temp_list[j].lower() not in
                                                 list_search_hashtags):
                hashtag_list.append(find_strip_pos(temp_list[j].lower()))
    for word in hashtag_list:
        if word in [x.replace('#', '') for x in list_search_hashtags]:
            hashtag_list.remove(word)
    counter = collections.Counter(hashtag_list)
    freq_hashtags = pd.DataFrame.from_dict(counter,
                                           orient='index').reset_index()
    freq_hashtags.columns = ['hashtag', 'frequency']
    freq_hashtags = freq_hashtags.sort_values(by=['frequency'],
                                              ascending=False)
    d = {}
    for a, x in freq_hashtags.values:
        d[a] = x

    wordcloud = WordCloud(colormap="Greens")
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return plt.show()


######################End #tag visualization #############

###################Timeline  Chart#############
def timeline(filename):
    
    
    def format_sentence(sent):
        return({word: True for word in nltk.word_tokenize(sent)})

    pos = []
    with open("pos_tweets.txt") as f:
        for i in f: 
            pos.append([format_sentence(i), 'pos'])
    
    neg = []
    with open("neg_tweets.txt",  encoding="utf-8") as f:
        for i in f: 
            neg.append([format_sentence(i), 'neg'])
    
    
    # next, split labeled data into the training and test data
    training = pos + neg    

    classifier = NaiveBayesClassifier.train(training)
    func = lambda x: classifier.classify(format_sentence(x))
    df = pd.read_csv(filename, header = None, names = ["Time", "Tweet"])

    df["catg"] = df["Tweet"].map(func)
    df["Date"] = pd.to_datetime(df["Time"]).dt.date

    df_new = pd.DataFrame(df.groupby(["Date", "catg"]).count()["Tweet"]).reset_index()

    df_pivot = df_new.pivot(index="Date", columns="catg", values="Tweet")
    df_pivot = df_pivot.fillna(value = 0)

    category = ["pos_%", "neg_%"]
    
    df_pivot["pos_%"] = round(df_pivot["pos"] * 100 / (df_pivot["pos"] + df_pivot["neg"]), 0)
    df_pivot["neg_%"] = round(df_pivot["neg"] * 100 / (df_pivot["pos"] + df_pivot["neg"]), 0)

    df_final = df_pivot[["pos_%", "neg_%"]]
#
#    plt.plot(df_final)
#    plt.legend(category)
#    plt.xlabel("Dates")
#    plt.ylabel("%")
#    plt.title("Sentiment distribution across days")

    fig, ax = plt.subplots(figsize=(10,7), facecolor = (0.207, 0.254, 0.254))
    df_final.plot(ax=ax)
    plt.xlabel("Dates", fontsize=15)
    plt.ylabel("%", fontsize=15)
    plt.title("Sentiment distribution across days", fontsize=20)
    ax.set_facecolor('#354141')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    
    #set ticks every week
    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
     
    
    return None
    
####################End timeline chart##################
    
#####################Bigram Generator###############

#### To be called with bigram from the below function. Outputs positive and negative set of words
def sankey_diagram(new_bigram):
    neg_words=pd.read_table('neg_tweets.txt',header=None)
    neg_word_list=neg_words[0].tolist()
    pos_words=pd.read_table('pos_tweets.txt',header=None)
    pos_word_list=pos_words[0].tolist()
    pos_word_list = [x.lower() for x in pos_word_list]
    neg_word_list = [x.lower() for x in neg_word_list]
    pos_list=[]
    neg_list=[]
    for word in new_bigram:
        list1=word.split()
        if (list1[0].strip().lower() in pos_word_list) or (list1[1].strip().lower() in pos_word_list):
            pos_list.append(word)
        if (list1[0] in neg_word_list) or (list1[1] in neg_word_list):
            neg_list.append(word)
    
    counter=collections.Counter(pos_list)
    pos_freq_hashtags = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    pos_fre_desc = pos_freq_hashtags.sort_values(by=0,ascending=False)
    
    counter=collections.Counter(neg_list)
    neg_freq_hashtags = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    neg_fre_desc = neg_freq_hashtags.sort_values(by=0,ascending=False)
    
    neg_fre_desc['sankey']=0
    for index,rows in neg_fre_desc.iterrows():
        neg_fre_desc['sankey'][index] ='Negative ['+str(neg_fre_desc[0][index])+"] "+str(neg_fre_desc['index'][index])
                
    pos_fre_desc['sankey']=0
    for index,rows in pos_fre_desc.iterrows():
        pos_fre_desc['sankey'][index] ='Positive ['+str(pos_fre_desc[0][index])+"] "+str(pos_fre_desc['index'][index])
                
    
    
    
    return neg_fre_desc,pos_fre_desc

   
   

####The Twitter Data Frame has to be passed to bigram generator. The output of this fun 
####is the paramter for sankey function 
def bigram_generator(filename):
    df = pd.read_csv(filename, header=None)
    df.columns = ['Date', 'Tweets']
    df = df[df['Tweets'].notnull()]
    exclude= ["s", 'https', "'re", "'s", "n't", "'", 'co', 'head', 'check',
              'latest', 'available', 'time', 'shop', 'promo', 'code']
    
    tokenized = df['Tweets'].apply(lambda x: word_tokenize(x))
    lst = [word.lower() for x in tokenized for word in x]
  
    
    bigram_list = []
    for index, rows in df.iterrows():
        a_tweet = df['Tweets'][index]
        temp_list = a_tweet.split()
        bi_temp_list = [x+' '+y for x,y in zip(temp_list, temp_list[1:])]
        bigram_list.append(bi_temp_list)

    
    new_bigram=[]
    
    for tweet in bigram_list:
        for word in tweet:
            if (word.find('@')==-1) and (word.find('http')==-1) and (word.find('\\')==-1):
                new_bigram.append(word)
    return new_bigram
#####################End Bigram ####################
    
word_cloud(clean_tweets(file, src_terms))
hashtag_association(filename)
timeline(filename)
new_bigram = bigram_generator(filename)
#sankey_diagram(new_bigram)

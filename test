
# Python program to generate word vectors using Word2Vec 

  
# importing all necessary modules 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize 
import pandas as pd

import warnings 

  

warnings.filterwarnings(action = 'ignore') 

  

import gensim 

from gensim.models import Word2Vec 

# csv einlesen
testdata=pd.read_csv("./Test.csv", sep=";", header=None)



    
#  Reads ‘alice.txt’ file 

sample = open("G:/Dropbox/_Uni/1. Dissertation/Scripts/word2vec/input.txt", "r", encoding="ISO-8859-1") 

s = sample.read() 

  
# Replaces escape character with space 

f = s.replace("\n", " ") 

  

data = [] 

  
# iterate through each sentence in the file 

for i in sent_tokenize(f): 

    temp = [] 

      

    # tokenize the sentence into words 

    for j in word_tokenize(i): 

        temp.append(j.lower()) 

  

    data.append(temp) 

  
# Create CBOW model 

model1 = gensim.models.Word2Vec(data, min_count = 1,  

                              size = 100, window = 5) 

datalist=[]

for i, row in testdata.iterrows():
  rowlist= []

  fotocounter="Foto "+ str(i+1)
  for x in row: 
     try: 
         x=str(x)
         
         #i+1 = Fotonummer
         rowlist.append([model1[x]])
  

     except KeyError:
        rowlist.append(["False"])
  print(rowlist)
  print(len(rowlist))
  header= ["1","2","3"]
  print(header)
  datalist.append([fotocounter, pd.DataFrame(rowlist, columns=header)])
print(datalist)     
    

df=pd.DataFrame(datalist)
#transponiert zeilen und spalten
#df_transposed = df.T
df.to_csv("vector-output.csv", index=False)
# Print results 

#print("Cosine similarity between 'hotel' " + 
#
#               "and 'sea' - CBOW : ", 
#
#    model1.similarity('hotel', 'sea')) 
#
#      
#
#print("Cosine similarity between 'alice' " +
#
#                 "and 'machines' - CBOW : ", 
#
#      model1.similarity('alice', 'wonderland'))
#print(model1['airport'])

  
# Create Skip Gram model 

#model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
#
#                                             window = 5, sg = 1) 

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#topics to fetch
categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
traning_data=fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=True)

print("\n".join(traning_data.data[0].split("\n")[:10]))
#we just count the words occurances
#countVectorizer will do the tokenizing end up with the document term matrix
count_vectors=CountVectorizer()
x_train_counts=count_vectors.fit_transform(traning_data.data)
#print(count_vectors.vocabulary_)
#we transform the word occurances into tf-idf
#tfdfVictorizer = CountVectorizer +TfidfTransformer
tfidf_transform=TfidfTransformer()
x_train_tfidf=tfidf_transform.fit_transform(x_train_counts)
#print(x_train_tfidf)

model=MultinomialNB().fit(x_train_tfidf,traning_data.target)
new=["this is have nothing to do with church and religion","Software engineering is getting hotter and hotter nowadays"]
x_new_counts=count_vectors.transform(new)
#the x_new_tfidf is the numerical presentation of the sentences
x_new_tfidf=tfidf_transform.transform(x_new_counts)
predicted=model.predict(x_new_tfidf)

for doc,category in zip(new,predicted):
    print("%r-------->%s"%(doc,traning_data.target_names[category]))
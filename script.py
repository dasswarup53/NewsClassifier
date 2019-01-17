import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import  accuracy_score

df=pd.read_csv("fake_or_real_news.csv")
df=df.set_index("Unnamed: 0")

y=df.label

df.drop("label",axis=1)

X_train,X_test,y_train,y_test=train_test_split(df["text"],y,test_size=0.33, random_state=53)

count_vectorizer = CountVectorizer(stop_words="english")
count_train=count_vectorizer.fit_transform(X_train)
count_test=count_vectorizer.transform(X_test)

tfidf_vectorizer=TfidfVectorizer(stop_words="english",max_df=0.7)
tfidf_train= tfidf_vectorizer.fit_transform(X_train)
tfidf_test=tfidf_vectorizer.transform(X_test)

# count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
# print(count_df.head())
# tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
# print(tfidf_df.head())

#trying MNB classfier on tfidf dataset
clf=MultinomialNB()
clf.fit(tfidf_train,y_train)
pred=clf.predict(tfidf_test)
score=accuracy_score(y_test,pred)
print("accuracy:   %0.3f" % score)

#trying MB on countvectorizer dataset
clf=MultinomialNB()
clf.fit(count_train,y_train)
pred=clf.predict(count_test)
score=accuracy_score(y_test,pred)
print("accuracy:   %0.3f" % score)

#as the results are not very impressive lets try out linear models
linear_clf=PassiveAggressiveClassifier(n_iter=50)
linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
score = accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
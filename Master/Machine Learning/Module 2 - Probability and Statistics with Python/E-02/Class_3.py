import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('score_de_jamonosidad.csv', usecols={'v1','v2','v3','score'})
y = data.pop('score')
clf = linear_model.SGDClassifier(loss='hinge', penalty='l2', shuffle=False, warm_start=False)
clf.fit(data,y)
objetivo1 = pd.read_csv('jamones_por_calificar.csv', usecols={'v1','v2','v3'})
objetivo1['score'] = clf.predict(objetivo1)


reg = LogisticRegression()
reg.fit(data,y)
reg.score(data,y)
objetivo2 = pd.read_csv('jamones_por_calificar.csv', usecols={'v1','v2','v3'})
objetivo2.head()
objetivo2['score'] = reg.predict(objetivo2)
print(objetivo1)
print(objetivo2)
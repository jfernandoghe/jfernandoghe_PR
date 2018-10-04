import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve

# Load dataset
data = pd.read_csv("creditcard.csv")
count_Class = pd.value_counts(data["Class"], sort=True)
count_Class.plot(kind='bar', figsize=(8, 10))

# Uneven dataset, downsample dataset to obtain 50/50 after the 492 fraud cases
No_of_frauds = len(data[data["Class"]==1])
fraud_index = np.array(data[data["Class"]==1].index)
normal_index = data[data["Class"]==0].index
random_normal_indices = np.random.choice(normal_index, No_of_frauds, replace=False)
random_normal_indices = np.array(random_normal_indices)
undersampled_indices = np.concatenate([fraud_index, random_normal_indices])
undersampled_data = data.iloc[undersampled_indices, :]

#
count_Class=pd.value_counts(undersampled_data["Class"], sort= True)
count_Class.plot(kind= 'bar', figsize=(8, 8))

# Normalization of amount
sc = StandardScaler()
undersampled_data["scaled_Amount"] = sc.fit_transform(undersampled_data.iloc[:,29].values.reshape(-1,1))

# Removal of time and amount
undersampled_data = undersampled_data.drop(["Time", "Amount"], axis= 1)
print(undersampled_data.head())

X = undersampled_data.iloc[:, undersampled_data.columns != "Class"].values
y = undersampled_data.iloc[:, undersampled_data.columns == "Class"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

classifier = SVC(C=1, kernel='rbf', random_state=0)
classifier.fit(X_train, y_train.ravel())
metrics.mean_absolute_error(y_test.ravel(), classifier.predict(X_test))

res = cross_validation.cross_val_score(classifier, X_train, y_train.ravel(), cv=5, scoring='accuracy')
print(res)
lr = LogisticRegression(C=0.01, penalty="l1")
lr.fit(X_train, y_train.ravel())

metrics.mean_absolute_error(y_test.ravel(), lr.predict(X_test))
res = cross_validation.cross_val_score(lr, X_train, y_train.ravel(), cv=5, scoring='accuracy')
print(res)
y_pred_score = lr.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_pred_score)
roc_auc = auc(fpr,tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

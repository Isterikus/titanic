from sklearn.preprocessing import LabelEncoder
from pandas import read_csv, DataFrame, Series
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import pylab as pl
import matplotlib.pyplot as plt
from dataPreprocess import get

data = read_csv("data/train.csv")
# data = get(data)
data.Age = data.Age.median()
MaxPassEmbarked = data.groupby('Embarked').count()['PassengerId']
data.Embarked[data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
data.PassengerId[data.Fare.isnull()]
data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
label = LabelEncoder()
dicts = {}

label.fit(data.Sex.drop_duplicates()) #задаем список значений для кодирования
dicts['Sex'] = list(label.classes_)
data.Sex = label.transform(data.Sex) #заменяем значения из списка кодами закодированных элементов 

label.fit(data.Embarked.drop_duplicates())
dicts['Embarked'] = list(label.classes_)
data.Embarked = label.transform(data.Embarked)

test = read_csv('data/test.csv')
test.Age[test.Age.isnull()] = test.Age.mean()
test.Fare[test.Fare.isnull()] = test.Fare.median() #заполняем пустые значения средней ценой билета
MaxPassEmbarked = test.groupby('Embarked').count()['PassengerId']
test.Embarked[test.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
result = DataFrame(test.PassengerId)
test = test.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)

label.fit(dicts['Sex'])
test.Sex = label.transform(test.Sex)

label.fit(dicts['Embarked'])
test.Embarked = label.transform(test.Embarked)

target = data.Survived
train = data.drop(['Survived'], axis=1) #из исходных данных убираем Id пассажира и флаг спасся он или нет
kfold = 5 #количество подвыборок для валидации
itog_val = {} #список для записи результатов кросс валидации разных алгоритмов

def parse(data):
	label = LabelEncoder()
	dicts = {}

	label.fit(data.Sex.drop_duplicates())
	dicts['Sex'] = list(label.classes_)

	label.fit(data.Embarked.drop_duplicates())
	dicts['Embarked'] = list(label.classes_)

	data.Age[data.Age.isnull()] = data.Age.mean()
	data.Fare[data.Fare.isnull()] = data.Fare.median() #заполняем пустые значения средней ценой билета
	MaxPassEmbarked = data.groupby('Embarked').count()['PassengerId']
	data.Embarked[data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
	result = DataFrame(data.PassengerId)
	data = data.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)

	label.fit(dicts['Sex'])
	data.Sex = label.transform(data.Sex)

	label.fit(dicts['Embarked'])
	data.Embarked = label.transform(data.Embarked)
	return data

def crossValidation(model_rfc, model_knc, model_svc, model_lr):
	scores = cross_validation.cross_val_score(model_rfc, train, target, cv = kfold)
	itog_val['RandomForestClassifier'] = scores.mean()
	scores = cross_validation.cross_val_score(model_knc, train, target, cv = kfold)
	itog_val['KNeighborsClassifier'] = scores.mean()
	scores = cross_validation.cross_val_score(model_lr, train, target, cv = kfold)
	itog_val['LogisticRegression'] = scores.mean()
	scores = cross_validation.cross_val_score(model_svc, train, target, cv = kfold)
	itog_val['SVC'] = scores.mean()

	DataFrame.from_dict(data = itog_val, orient='index').plot(kind='bar', legend=False, grid=True)

	plt.show()

def rocCurve(model_rfc, model_knc, model_svc, model_lr):
	ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.25)
	pl.clf()
	plt.figure(figsize=(8,6))
	#SVC
	model_svc.probability = True
	probas = model_svc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
	fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
	roc_auc  = auc(fpr, tpr)
	pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVC', roc_auc))
	#RandomForestClassifier
	probas = model_rfc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
	fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
	roc_auc  = auc(fpr, tpr)
	pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('RandonForest',roc_auc))
	#KNeighborsClassifier
	probas = model_knc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
	fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
	roc_auc  = auc(fpr, tpr)
	pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('KNeighborsClassifier',roc_auc))
	#LogisticRegression
	probas = model_lr.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
	fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
	roc_auc  = auc(fpr, tpr)
	pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LogisticRegression',roc_auc))
	pl.plot([0, 1], [0, 1], 'k--')
	pl.xlim([0.0, 1.0])
	pl.ylim([0.0, 1.0])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.legend(loc=0, fontsize='small')
	pl.show()

def find_best():
	model_rfc = RandomForestClassifier(n_estimators = 100) #в параметре передаем кол-во деревьев
	model_knc = KNeighborsClassifier(n_neighbors = 18) #в параметре передаем кол-во соседей
	model_lr = LogisticRegression(penalty='l1', tol=0.01)
	model_svc = svm.SVC() #по умолчанию kernek='rbf'
	crossValidation(model_rfc, model_knc, model_svc, model_lr)
	rocCurve(model_rfc, model_knc, model_svc, model_lr)

def solveToFile():
	model_rfc = RandomForestClassifier(n_estimators = 100)
	model_rfc.fit(train, target)
	result.insert(1,'Survived', model_rfc.predict(test))
	result.to_csv('data/solution.csv', index=False)

if __name__ == '__main__':
	pass
	# find_best()
	# solveToFile()

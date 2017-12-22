from pandas import read_csv, DataFrame, Series
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def getData(data):
	data.Age = data.Age.median()
	MaxPassEmbarked = data.groupby('Embarked').count()['PassengerId']
	data.Embarked[data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
	data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
	return data

if __name__ == '__main__':
	data = read_csv('data/train.csv')

	# Analyzing fields importance
	data.pivot_table('PassengerId', 'Pclass', 'Survived', 'count').plot(kind='bar', stacked=True)
	fig, axes = plt.subplots(ncols=2)
	data.pivot_table('PassengerId', ['SibSp'], 'Survived', 'count').plot(ax=axes[0], title='SibSp')
	data.pivot_table('PassengerId', ['Parch'], 'Survived', 'count').plot(ax=axes[1], title='Parch')

	plt.show()

	# Analyzing fields fullness
	cab_not_null = data.PassengerId[data.Cabin.notnull()].count()
	age_not_null = data.PassengerId[data.Age.notnull()].count()
	print("ALL = ", data.PassengerId.count() - 1)
	print("Cabin not null = ", cab_not_null)
	# Too much null in Cabin - ignore field
	print("Age not null = ", age_not_null)
	# Age almost full - fill it median
	data.Age = data.Age.median()

	# 2 Embarked null - fill it with most popular
	print(data[data.Embarked.isnull()])
	MaxPassEmbarked = data.groupby('Embarked').count()['PassengerId']
	data.Embarked[data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]

	# 0 Fare null - good :)
	print(data.PassengerId[data.Fare.notnull()])

	# Dropping useless fields
	data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

	print(data.head())
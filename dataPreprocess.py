from sklearn.preprocessing import LabelEncoder

from dataAnalize import getData

def get(data):
	data = getData(data)
	label = LabelEncoder()
	dicts = {}

	label.fit(data.Sex.drop_duplicates())
	dicts['Sex'] = list(label.classes_)
	data.Sex = label.transform(data.Sex)

	label.fit(data.Embarked.drop_duplicates())
	dicts['Embarked'] = list(label.classes_)
	data.Embarked = label.transform(data.Embarked)
	return data

if __name__ == '__main__':
	data = get()

	print(data.head())
	data.to_csv("data/mine_train.csv")

import pandas as pd
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


music_data = pd.read_csv('music.csv')

##print(music_data)

x = music_data.drop(columns = ["genre"])
y = music_data['genre']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)

model = DecisionTreeClassifier()

model.fit(x.values, y)

prediction = model.predict([ [21,1], [22,0]])

print(prediction)
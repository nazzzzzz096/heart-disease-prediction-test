from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess_scale

data = load_breast_cancer()
print(data.data)
print(data.target)
print(data.data.shape)
print(data.target.shape)

x=preprocess_scale(data.data)
print(x)
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=50,random_state=42)
model.fit(x_train,y_train)
print("model trained")
with open('model.pkl','wb') as f:
    pickle.dump(model,f)

print("model saved")
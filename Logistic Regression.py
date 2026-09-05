import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegreession 
from sklearn.metrics import confusion_matrix, Precision_score, Recall_score, f1_score
from sklearn.model_selection import train_test_split

X= pd.DataFrame({
    'Study Hours': [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,8,9],
    'Percentage': [32,38,42,45,48,51,55,58,62,65,68,71,75,78,85]
})
y = [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]

print ("Input Values:",X) 
print(y)
X_train, X_test, y_train, y_test=train_test_split (X,y,test_size=0.2, random_state=42, stratify=y)
model= LogisticRegreession()
model.fit(X_train, y_train) 
y_pred= model.predict(X_test)
print("Confusion Matrix:", confusion_matrix(y, y_pred))
print("Precision Score:", Precision_score(y, y_pred))
print("Recall Score:", Recall_score(y, y_pred))
print("F1 Score:", f1_score(y, y_pred))

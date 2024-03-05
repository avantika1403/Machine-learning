from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV

iris=datasets.load_iris()
X=iris.data
Y=iris.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42,stratify=Y )

sc= StandardScaler()
sc.fit(X_train)
X_train_std= sc.transform(X_train)
X_test_std= sc.transform(X_test)

knn=KNeighborsClassifier(n_neighbors=5,p=2,weights='uniform',algorithm=' auto')
knn.fit(X_train_std,Y_train)
print("Name: Rohit \t Roll no: 09")
print("Traing Accuracy score: %.3f " %knn.score(X_train_std,Y_train))
print("Test Accuracy score: %.3f" %knn.score(X_test_std,Y_test))
iris=datasets.load_iris()
X=iris.data
Y=iris.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42,stratify=Y )

pipeline=make_pipeline(StandardScaler(),KNeighborsClassifier())

param_grid= [{
    'kneighborsclassifier__n_neighbors':[2,3,4,5,6,7,8,9,10],
    'kneighborsclassifier__p':[1,2],
    'kneighborsclassifier__weights':['uniform','distance'],
'kneighborsclassifier__algorithm':['auto','ball_tree','kd_tree','brute']}]

gs=GridSearchCV(pipeline, param_grid=param_grid,
                scoring="accuracy",
                refit=True,
                cv=10,
                verbose=1,
                n_jobs=2)

gs.fit(X_train,Y_train)

print("Best Score: %.3f " %gs.best_score_ ," \nBest Parameters:",gs.best_params_)
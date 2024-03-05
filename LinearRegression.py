print("Practical no.1A : Linear Regression \nName: Rohit \t Roll no.:09")
print("---------------------------------------------------")
import random
from sklearn.linear_model import LinearRegression
print("Read the train Data")
print("---------------------------------------------------")
feature_set = []
target_set = []
no_of_rows = 200
limit = 2000
for i in range(0, no_of_rows):
    x = random.randint(0, limit)
    y = random.randint(0, limit)
    z = random.randint(0, limit)
    g = 10 * x + 2 * y + 3 * z
print("x=", x, "\ty=", y, "\tz=", z, "\tg=", g), feature_set.append([x, y, z]), target_set.append(g)
print("Here the training of model begins. ")
model = LinearRegression()
model.fit(feature_set, target_set)
print("Training of model ends here!")
print("Testing started here")
test_data = [[1, 1, 0]]
print("Test data:", test_data)
prediction = model.predict(test_data)
print("prediction:" + str(prediction) + '\t' + "Coefficient:" + str(model.coef_))
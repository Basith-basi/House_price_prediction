import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=pd.read_csv("house_data.csv") #reading the dataset
print("Dataset:") #printing the dataset
print(data)
x=data[['area']] #feature
y=data['price'] #target
model=LinearRegression() #linear regression model
model.fit(x,y) #training the model
area=float(input("Enter house area (sq.ft): ")) #taking user input for area
predicted_price=model.predict([[area]])
print("\n Predicted Price:",predicted_price[0]) #printing the predicted price
# Plot graph
plt.scatter(data['area'], data['price'])
plt.plot(data['area'], model.predict(x))
plt.xlabel("area(sq.ft)")
plt.ylabel("price")
plt.title(" house price Prediction")
plt.show()
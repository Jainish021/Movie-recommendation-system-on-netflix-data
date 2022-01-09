import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# importing the data from the file and arranging to make a matrix of User_id*Movie_id.
data = pd.read_csv("netflix/TrainingRatings.txt",
                   names=["Movie_id", "User_id", "Ratings"])

cols = data["Movie_id"]
rows = data["User_id"]
ratings = data["Ratings"]
df = data.pivot(index="User_id", columns="Movie_id", values="Ratings")
df = df.sort_index(axis=0)
df.astype('float16')

# calculating the mean values of the ratings.
mean_values = df.mean(axis=1).to_numpy()


# Centering the matrix of User_id*Movie_id to zero and replacing the nan alue to zero.
matrix = df.to_numpy()
matrix = np.subtract(matrix.T, mean_values).T
matrix = np.nan_to_num(matrix)
matrix.astype(np.float16)
normal = np.linalg.norm(matrix, axis=1)
del df


# Calculating the weights for all the users
weights = []
print("Calculating the weights. It will take 10-15 mins.")
for i, j, k in zip(range(28978), matrix, normal):
    numerator = np.dot(j, np.transpose(matrix))
    denominator = np.multiply(k, normal)
    denominator = np.where(denominator == 0, 1, denominator)
    weights.append(np.divide(numerator, denominator))
weights = np.asarray(weights)
np.fill_diagonal(weights, 0)

# Importing the trining data
df1 = pd.read_csv("netflix/TrainingRatings.txt",
                  names=["Movie_id", "User_id", "Ratings"])
cols = pd.unique(df1.loc[:, "Movie_id"])
rows = pd.unique(df1.loc[:, "User_id"])
data = df1.pivot(index="User_id", columns="Movie_id", values="Ratings")
data = data.sort_index(axis=0)


# Importing the test data
test_data = pd.read_csv("netflix/TestingRatings.txt",
                        names=["Movie_id", "User_id", "Ratings"])
test_data = test_data.to_numpy()

# Filtering the weights to use only the weights of the neighbors.
weights = np.clip(weights, 0, 1000)

# Calculating the kappa.
k = np.divide(1, (np.sum(np.absolute(weights), axis=1)))
np.nan_to_num(k, 0)

# Making the predictions over all the users
pred = np.dot(weights, matrix)
pred = np.multiply(k, pred.T).T
pred = np.add(mean_values, pred.T).T
pred = np.round(pred)

# Getting the predicted values for the target users only
predicted_y = []
for data in test_data:
    user = np.where(rows == data[1])
    movie = np.where(cols == data[0])
    predicted = pred[user[0], movie[0]]
    predicted = predicted[0]
    # print(predicted)
    predicted_y.append(predicted)


# Calculating the RMSE and MAE values.
true_y = test_data[:, 2]
rmse = mean_squared_error(true_y, predicted_y)
mae = mean_absolute_error(true_y, predicted_y)
print("RMSE is :", rmse)
print("MAE is :", mae)

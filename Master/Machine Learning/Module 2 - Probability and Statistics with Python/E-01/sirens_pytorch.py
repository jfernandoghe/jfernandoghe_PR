# coding: utf-8
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Reading CSV
dataset = pd.read_csv(
    '/home/jfernandoghe/Documents/x_Datasets/Dataset/Sirens/sirenas_endemicas_y_sirenas_migrantes_historico.csv')
dataset.head(100)
# Building one dataframe
df = pd.DataFrame(data=dataset)
mapping = {'sirena_endemica': 1, 'sirena_migrante': 2}
df = df.replace({'especie': mapping})
# Building dataframes for X and Y
y = dataset.pop('especie')
y_df = pd.DataFrame(data=y)
mapping = {'sirena_endemica': 1, 'sirena_migrante': 2}
df_y = y_df.replace({'especie': mapping})


# Convert pandas.dataframe to pytorch.tensor
x_temp = df.iloc[:, :-1].values
y_temp = df_y.iloc[:, :].values
X_train = torch.FloatTensor(x_temp)
Y_train = torch.FloatTensor(y_temp)

print(df)

# x1_var = Variable(X_train, requires_grad=True)
# linear_layer1 = nn.Linear(4, 1)
# # create a linear layer (i.e. a linear equation: w1x1 + w2x2 + w3x3 + w4x4 + b, with 4 inputs and 1 output)
# # w and b stand for weight and bias, respectively
# predicted_y = linear_layer1(x1_var)
# # run the x1 variable through the linear equation and put the output in predicted_y
# print("----------------------------------------")
# print(predicted_y)
# print("----------------------------------------")
# # prints the predicted y value (the weights and bias are initialized randomly; my output was 1.3712)



# model = tree.DecisionTreeClassifier()
# model.fit(data,y)
# result = pd.read_csv('/home/jfernandoghe/Documents/x_Datasets/Dataset/Sirens/sirenas_endemicas_y_sirenas_migrantes.csv')
# result
# result.pop("especie")
# result_predict = model.predict(result)
# result_predict
# pd_result = pd.DataFrame(data=result_predict, columns=['sirena_endemica','sirena_migrante'])
# pd_result.head()
# result["especie"] = pd_result.idxmax(axis=1)
# result
# result.to_csv('/home/jfernandoghe/Documents/Master/07_Codes/Local/ML-Applied//resultado_sirenas.csv')

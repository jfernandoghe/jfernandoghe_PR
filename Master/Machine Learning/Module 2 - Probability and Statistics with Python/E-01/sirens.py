# coding: utf-8

import numpy as np
from sklearn import tree
import pandas as pd
data = pd.read_csv('/home/jfernandoghe/Documents/x_Datasets/Dataset/Sirens/sirenas_endemicas_y_sirenas_migrantes_historico.csv')
data.head(100)
y = pd.get_dummies(data.pop("especie"))
y.shape
data.shape
y.head()
model = tree.DecisionTreeClassifier()
model.fit(data, y)
result = pd.read_csv('/home/jfernandoghe/Documents/x_Datasets/Dataset/Sirens/sirenas_endemicas_y_sirenas_migrantes.csv')
result
result.pop("especie")
result_predict = model.predict(result)
result_predict
pd_result = pd.DataFrame(data=result_predict, columns=['sirena_endemica','sirena_migrante'])
pd_result.head()
result["especie"] = pd_result.idxmax(axis=1)
result
result.to_csv('/home/jfernandoghe/Documents/Master/07_Codes/Local/ML-Applied//resultado_sirenas.csv')






















# class CustomDatasetFromImages(Dataset):
#     def __init__(self, csv_path):
#         """
#         Args:
#             csv_path (string): path to csv file
#             img_path (string): path to the folder where images are
#             transform: pytorch transforms for transforms and tensor conversion
#         """
#         # Transforms
#         self.to_tensor = transforms.ToTensor()
#         # Read the csv file
#         self.data_info = pd.read_csv(csv_path, header=None)
#         # First column contains the image paths
#         self.image_arr = np.asarray(self.data_info.iloc[:, 0])
#         # Second column is the labels
#         self.label_arr = np.asarray(self.data_info.iloc[:, 1])
#         # Third column is for an operation indicator
#         self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
#         # Calculate len
#         self.data_len = len(self.data_info.index)
#
#     def __getitem__(self, index):
#         # Get image name from the pandas df
#         single_image_name = self.image_arr[index]
#         # Open image
#         img_as_img = Image.open(single_image_name)
#
#         # Check if there is an operation
#         some_operation = self.operation_arr[index]
#         # If there is an operation
#         if some_operation:
#             # Do some operation on image
#             # ...
#             # ...
#             pass
#         # Transform image to tensor
#         img_as_tensor = self.to_tensor(img_as_img)
#
#         # Get label(class) of the image based on the cropped pandas column
#         single_image_label = self.label_arr[index]
#
#         return (img_as_tensor, single_image_label)
#
#     def __len__(self):
#         return self.data_len
#
# if __name__ == "__main__":
#     # Call dataset
#     custom_mnist_from_images =  \
#         CustomDatasetFromImages('/home/jfernandoghe/Documents/x_Datasets/Dataset/Sirens/sirenas_endemicas_y_sirenas_migrantes_historico.csv')
#     print (custom_mnist_from_images[1])
#
#
#
#
#
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # sirena migrante 0
# # sirena endemica 1
# # sirens = pd.read_csv(datasetroot+'sirenas_endemicas_y_sirenas_migrantes_historico.csv')
# #
# # X=sirens
# # y=sirens.pop(5)
# #
# # print(X)
# # # X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
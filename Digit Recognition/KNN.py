import numpy as np
import pandas as pd

#Import KNN Model
from sklearn.neighbors import KNeighborsClassifier as KNN

#Used for training set expansion
from scipy.ndimage.interpolation import shift

#For image plotting

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#For plotting multiple digits

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


#Read our data as a pandas dataframe, initially
mnist = pd.read_csv("train.csv")
mnist.head()

#Convert the dataframe to a numpy array (matrix)
mnist = np.array(mnist)

#Split data into predictor and target variables
X, y = mnist[:,1:], mnist[:,0]

#initialize the model with pre-selected hyper-parameters
knn_clf = KNN(n_neighbors=4, weights="distance", n_jobs=-1)

#Fit the Model on the expanded training set
knn_clf.fit(X,y)

#Read test data
test = pd.read_csv("test.csv")
#convert it to numpy array format
test = np.array(test)

#predictions
predictions = knn_clf.predict(test)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submission.csv", index=False, header=True)
print("finished")
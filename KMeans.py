import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set Data Column Names
colnames=['AREA', 'PERIMETER', 'LENGTHKERNAL', 'WIDTHKERNAL','LENGTHGROOVE','COMPACTNESS'] 

# Read A CSV file containing data and set the column names for each line.211
Input_Data = pd.read_csv('seeds_dataset.csv', names=colnames, header=None)

print (Input_Data)

# No Missing Values so no need to Input_Data.dropna(inplace=True)

# Initlizaing Standard Scaler and saving it to a variable to be used later on

Scalar = StandardScaler()

# Add New Columns to the Data for the scalered version of data and then tranform the existing data using the Standard Scaler and save to new columns. Standardise data
Input_Data[['AREA_Scaler', 'PERIMETER_Scaler', 'LENGTHKERNAL_Scaler', 'WIDTHKERNAL_Scaler','LENGTHGROOVE_Scaler','COMPACTNESS_Scaler']] = Scalar.fit_transform(Input_Data[['AREA', 'PERIMETER', 'LENGTHKERNAL', 'WIDTHKERNAL','LENGTHGROOVE','COMPACTNESS']])

print(Input_Data)

# Create a subplot
figure, axis = plt.subplots(1, 3, figsize=(18, 5))

# Optimum number of clusters for a given set of data. - K means

def Optimum_clusters(data, max_k, plt_figure):
    Means = []
    Inertias = []
    
    for k in range(1, max_k):
        # Creates a new New Kmeans algorithm which has a number of clusters equal to k
        kmeans = KMeans(n_clusters=k)
        
        # Adds the inputted data into the algorithm, this will also run the clustering algorithm and make each point in the data be assigned to a cluster.
        kmeans.fit(data)
        
        # Adds the number of Clusters (k) to the array
        Means.append(k)
        
        # Sum of squared distances of samples to their closest cluster center. added the array
        Inertias.append(kmeans.inertia_)
        
    #Generate the Output Data into a graph for optimial user readability1
    string = "Figure " + str(plt_figure+1)

    # Plots the Kmeans number on the X axis and the inertias on the Y axis.
    axis[plt_figure].plot(Means, Inertias)
    
    # Adds A Title to each graph.
    axis[plt_figure].set_title(string)
    
# Give Pairs of data to Created Variable Above.

Optimum_clusters(Input_Data[['AREA_Scaler', 'PERIMETER_Scaler']], 10, 0)
Optimum_clusters(Input_Data[['LENGTHKERNAL_Scaler', 'WIDTHKERNAL_Scaler']], 10, 1)
Optimum_clusters(Input_Data[['LENGTHGROOVE_Scaler','COMPACTNESS_Scaler']], 10, 2)

# Create Kmeans with the number of clusters equaling to a chosen number
kmeans1 = KMeans(n_clusters=3)
kmeans2 = KMeans(n_clusters=3)
kmeans3 = KMeans(n_clusters=4)

# Insert the data into the Kmeans algorithm.
kmeans1.fit(Input_Data[['AREA_Scaler', 'PERIMETER_Scaler']])
kmeans2.fit(Input_Data[['LENGTHKERNAL_Scaler', 'WIDTHKERNAL_Scaler']])
kmeans3.fit(Input_Data[['LENGTHGROOVE_Scaler','COMPACTNESS_Scaler']])

# Create new columns in Input_data table and set them to the Kmeans labels.
Input_Data['kmeans_1'] = kmeans1.labels_
Input_Data['kmeans_2'] = kmeans2.labels_
Input_Data['kmeans_3'] = kmeans3.labels_

print(Input_Data)


# Create a subplot
figure2, axis2 = plt.subplots(1, 3, figsize=(18, 5))

axis2[0].scatter(x=Input_Data['AREA'], y=Input_Data['PERIMETER'], c = Input_Data['kmeans_1'])
axis2[0].set_title("Figure 1 - Area and Perimeter")
axis2[0].set_ylabel('Perimeter')
axis2[0].set_xlabel('Area')

axis2[1].scatter(x=Input_Data['LENGTHKERNAL'], y=Input_Data['WIDTHKERNAL'], c = Input_Data['kmeans_2'])
axis2[1].set_title("Figure 2 - Kernal Length and Width")
axis2[1].set_ylabel('Kernal Width')
axis2[1].set_xlabel('Kernal Length')

axis2[2].scatter(x=Input_Data['LENGTHGROOVE'], y=Input_Data['COMPACTNESS'], c = Input_Data['kmeans_3'])
axis2[2].set_title("Figure 3 - Groove length and compactness")
axis2[2].set_ylabel('Compactness')
axis2[2].set_xlabel('Groove Length')

plt.show()
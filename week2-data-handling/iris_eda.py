import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv') # Read data from URL
INDEX = 1

def plot(num, x, y, species):
    plt.figure(num)
    x_ = raw_data[raw_data['species'] == species][x]
    y_ = raw_data[raw_data['species'] == species][y]
    plt.scatter(x_,y_)
    plt.xlabel(species.upper() + " " + x.replace('_',' ').upper())
    plt.ylabel(species.upper() + " " + y.replace('_',' ').upper())
    plt.yticks(rotation=90)
    m, c = np.polyfit(x_, y_, 1) #gets value for y = mx+c
    plt.plot(x_, m*x_+c)
    plt.title('Figure '+ str(num))
    plt.show()

for i in raw_data['species'].unique():
   plot(INDEX,'sepal_length', 'sepal_width',i)
   INDEX+=1
   plot(INDEX,'petal_length','petal_width',i)
   INDEX+=1


#print(raw_data['species'].value_counts()['setosa'])



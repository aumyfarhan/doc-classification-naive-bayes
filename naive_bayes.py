#written by farhan
#2018
# import necesssary python mofdules

import pandas as pd
import numpy as np
import math
import csv



# parsing input argument for training file, testing file name
#training_file=sys.argv[1]
#testing_file=sys.argv[2]

#training_file='training.csv'
#testing_file='testing.csv'

# list of different beta values to be used
beta_values=[0.00001,0.0001,0.001,0.01,0.1,1]

# no of class in the data
class_no=20

# total no of words(considered as feature attributes) in the data
word_no=61188

# defining numpy array to store each word count for each class
word_count_matrix=np.zeros((20,word_no))

# defining numpy array to store each class label count
label_count=np.zeros((20,1))

# defining chunksize (no of rows) to be read from the training and testing
# csv files at a time
chunk_size=1000


# reading "chunk_size=1000" amount of training example data
# at a time and appending to a list and then concatenating to
# pandas dataframe

chunks=[]
for chunk in pd.read_csv('training.csv', chunksize=chunk_size, header=None,dtype=int):
    chunks.append(chunk)
    
df=pd.concat(chunks,axis=0)

training_all=df.values
X_all=training_all[:,1:-1]
label_all=training_all[:,-1]

del chunks
del chunk
del df

# counting and storing each word occurance to word_count_matrix;
# also counting class occurance and storing to label_count


for i in range(0,class_no):
    ind=[p for p,q in enumerate(label_all) if q==(i+1)]
    
    # class occurance counting
    label_count[i]=label_count[i]+len(ind)
    
    # word occurance counting for each individual class
    word_count_matrix[i,:]=word_count_matrix[i,:]+np.sum(X_all[ind,:],axis=0)


# reading testing file and doing class prediction
chunks=[]
for chunk in pd.read_csv('testing.csv', chunksize=chunk_size, header=None,dtype=int):
    chunks.append(chunk)
    
df=pd.concat(chunks,axis=0)

testing_all=df.values
del chunks
del chunk
del df

X_test=testing_all[:,1:]

# adding one to the first column
TT=np.ones((X_test.shape[0],1))
X_test=np.column_stack((TT,X_test))

label_id=testing_all[:,0]




# itrating through beta_values to calculate 
# conditional probability matrix for each chosen beta values
        
for i,beta in enumerate(beta_values):
    
    
    #alpha values from beta values
    alpha=1+beta
    
    # defining conditional probability matrix 
    pp=np.zeros((word_no+1,20))
    
    # caculating conditional probability matrix
    for i in range(0,class_no):
        total_sum=sum(word_count_matrix[i,:])+(alpha-1)*word_no
        
        # appending the class log scale probabilty to the left of conditional
        # probabilty matrix for convenient matrix multiplication
        # while calculating posterior probability of
        # each testing example
        pp[0,i]=math.log2(label_count[i]/sum(label_count[:]))
        
        # calculating log scale conditional probability
        # using chosen beta value to do smoothing for 
        # missing word (features) in a particular class 
        pp[1:,i]=[math.log2((word_count_matrix[i,j]+(alpha-1))/total_sum) for j in range(0,word_no)]
        
        
    print('TRAINING DONE\n')
    print('USED BETA: {:6f}\n'.format(beta))
    
    # list to store testing example id and predicted class label
    test_label=[]

    
    #  predicting testing example class label
    
    # multiplication of input testing example data matrix and condtional
    # probability matrix
    temp_matrix=np.matmul(X_test,pp)
    
    # argmax to find index of class with maximum posterior probability
    # for the given test example data and adding one for 
    # zero indexing of python        
    temp_label=temp_matrix.argmax(axis=1)+1
    test_label=test_label+temp_label.tolist()
    
    
    # output filename creation
    
    output_file='final_naive_bayes_'+str(beta)+'_result.csv'
    
    # appending the test data sample id and predicted data sample class label in a list
    table=[]
    pd=str('id')
    cc=str('class')
    table.append([pd,cc]) 
    for i,j in enumerate(label_id):
        y=test_label[i]
        table.append([j,y])
    
    
    # creating a csv file of data sample id and their corresponding  class label 
    with open(output_file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in table:
            writer.writerow(val)
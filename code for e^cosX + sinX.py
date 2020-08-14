# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:30:53 2020

@author: Dell
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import math


"""Domain = (0,1). Generate 100 data points randomly 
   using below function by adding Gaussian noise with 0 mean.
   Divide the data set as 80:20 (train:test). """ 
   
########################## input values in domin (0,1) ########################   
#input_x = np.random.uniform(low  = 0 , high = 1, size= [100,1])
#print(input_x)
#np.reshape(input_x ,( 100, 1))
#print(input_x.shape)
#
#
#input_x0 = np.linspace(0, 1, num=100, endpoint=False, dtype = float )
#print(input_x0)
#np.reshape(input_x0 ,(100, 1))
#input_x1 = np.array(input_x0)
#print("shape of input X1 ",input_x1)
#
#print("shape of input x1\n",input_x1.shape)


input_x0 = np.linspace(0, 1, num=100, endpoint=False)
input_x = [ ]
for i in range(len(input_x0)):
    input_x.append(float(0 + input_x0[i]))
input_x = np.reshape(input_x , ( 100, 1))
#print("input value x is \n", input_x)

#np.reshape(input_x1 ,( 100, 1))
#print(input_x1.shape)


#print(x)
#################  output data using function for input_x  ####################
out_array_y = []   
for i in range(len(input_x)): 
    out_array_y.append(float(math.exp(math.cos(input_x[i]) + math.sin(input_x[i]))))
out_array_y = np.reshape(out_array_y , ( 100, 1))
#print("output value y is \n", out_array_y)

#print(out_array_y)    
#out_array_y = np.array(out_array_y)
#print("shape of output y \n ",out_array_y)
#np.reshape(out_array_y ,(1, -1))
#print("shape of the array ",out_array_y.shape)

#
#out_array_y1 = []   
#for i in range(len(input_x1)): 
#    out_array_y1.append(float(math.exp(math.cos(input_x1[i]) + math.sin(input_x1[i]))))    
#print("\n  y value for input x1 \n", out_array_y1)
#    
################ to Add noise in the data set ##################################
z = np.random.normal(0 , 0.2 , size= [100,1])
final_out_array_y = []   
for i in range(len(z)): 
    final_out_array_y.append( float(out_array_y[i] + z[i]) ) 
final_out_array_y = np.reshape(final_out_array_y , ( 100, 1))
#print("\t\n output value final y is \n", final_out_array_y)
    


x = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
input_x1= []
final_out_array_y1 = []
for i in range(len(x)):
    e = x[i]
#    print(e)
    input_x1.append(input_x[e]) 
    final_out_array_y1.append(final_out_array_y[e])
final_out_array_y1 = np.reshape(final_out_array_y1 , ( 10, 1))
#print("input value x is \n", final_out_array_y1)

##print("output data with noise \n",final_out_array_y)
#
#z = np.random.normal(0 , 0.2 , size= [100,1])
#final_out_array_y1 = []   
#for i in range(len(z)): 
#    final_out_array_y1.append( float(out_array_y1[i] + z[i]) )    


###################### total data in array ####################################
####################### data in a single coulmn ########################################
#
#print("\n #################################################################### \n")
#data =  np.column_stack(( input_x , final_out_array_y))
#print("final data set x input and y output \n",data)
#print("\n #################################################################### \n")
#
#      
################### training and testing data split  ############################
#
#X_train, X_test, y_train, y_test = train_test_split(input_x, final_out_array_y, test_size=0.20, random_state=42)
########################## train and test to understand ####################### 
##print(X_train)
##print("size of the X training input",X_train.shape)
##print("\n -------------------------------- ")
##print(X_test)
##print("size of the X testing input",X_test.shape)
##print("\n -------------------------------- ")
##print("training output y train is  \n",y_train)
###print("size of the y training input",y_train.shape)
##print("\n -------------------------------- ")
##print("testing output y test is  \n",y_test)
###print("size of the y training input",y_test.size)
##print("\n -------------------------------- ")
#################################################################################
################################################################################
#
#"""
#   Choose 10 points from the training data-set and 
#   perform regression for degrees: {1,3,6,9}. 
#
#"""
#
################################################################################\
################################################################################
#
#print("\n #################################################################### \n")
###################################### linear regression #######################      
#################### Fitting Linear Regression to the dataset ##################
#
lin_reg = LinearRegression()
lin_reg.fit( input_x , final_out_array_y)
output_y_value10 = lin_reg.predict(input_x)
plt.scatter(input_x , final_out_array_y, color = 'k')
plt.plot(input_x, output_y_value10 , color = 'y')
plt.title('Linear Regression degree 1')
plt.xlabel('Input X value')
plt.ylabel('Output Y')
plt.show()


lin_reg = LinearRegression()
lin_reg.fit( input_x1 , final_out_array_y1)
output_y_value1 =  lin_reg.predict(input_x1)
plt.scatter(input_x1 , final_out_array_y1, color = 'k')
plt.plot(input_x1, output_y_value1, color = 'y')
plt.title('Linear Regression degree 1')
plt.xlabel('Input X value')
plt.ylabel('Output Y')
plt.show()

###############################################################################
############### RMS ERROR FOR lInear case #####################################


Y_diff = 0 
for i in range(len(input_x1)):
    Y_diff = Y_diff + (output_y_value1[i] - final_out_array_y1[i])**2
#
RMS_ERROR_1 = ( (Y_diff) / len(input_x1) )**0.5
#
print("RMS Error for polynomial degree 3 case = ", RMS_ERROR_1)


Y_diff = 0 
for i in range(len(input_x1)):
    Y_diff = Y_diff + (output_y_value10[i] - final_out_array_y1[i])**2
#
RMS_ERROR10_1 = ( (Y_diff) / len(input_x1) )**0.5
#
print("RMS Error more than 10 input polynomial degree 1 case = ", RMS_ERROR10_1)


#lin_reg = LinearRegression()
#lin_reg.fit( input_x1 , final_out_array_y1)
#plt.scatter(input_x1 , final_out_array_y1, color = 'k')
#plt.plot(input_x1, lin_reg.predict(input_x1), color = 'y')
#plt.title('Linear Regression degree 1')
#plt.xlabel('Input X value')
#plt.ylabel('Output Y')
#plt.show()
      
      
      
#      
#print("\n #################################################################### \n")
#print("\n #################################################################### \n")
#
############################# polynomial regression ############################        
############# Fitting degree 3rd Polynomial, Regression to the dataset ##########
poly_reg3 = PolynomialFeatures(degree = 3) 
X_poly3 = poly_reg3.fit_transform(input_x)
#print(X_poly3)
#print(X_poly3.shape)

poly_reg3.fit(X_poly3, final_out_array_y)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly3, final_out_array_y)
output_y_value13 = lin_reg_3.predict(poly_reg3.fit_transform(input_x))
plt.scatter(input_x, final_out_array_y)
plt.plot(input_x, output_y_value13 , color = 'm')
plt.title('Polynomial Regression Degree 3')
plt.xlabel('input data X')
plt.ylabel('output data Y')
plt.show()


poly_reg3 = PolynomialFeatures(degree = 3) 
X_poly3 = poly_reg3.fit_transform(input_x1)
#print(X_poly3)
#print(X_poly3.shape)

poly_reg3.fit(X_poly3, final_out_array_y1)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly3, final_out_array_y1)
output_y_value3 =  lin_reg_3.predict(poly_reg3.fit_transform(input_x1))
plt.scatter(input_x1, final_out_array_y1)
plt.plot(input_x1, output_y_value3 , color = 'm')
plt.title('Polynomial Regression Degree 3')
plt.xlabel('input data X')
plt.ylabel('output data Y')
plt.show()


###############################################################################
############### RMS ERROR FOR degree 3 case #####################################

Y_diff = 0 
for i in range(len(input_x1)):
    Y_diff = Y_diff + (output_y_value3[i] - final_out_array_y1[i])**2
#
RMS_ERROR_3 = ( (Y_diff) / len(input_x1) )**0.5
#
print("RMS Error for polynomial degree 3 case = ", RMS_ERROR_3)



Y_diff = 0 
for i in range(len(input_x1)):
    Y_diff = Y_diff + (output_y_value13[i] - final_out_array_y1[i])**2
#
RMS_ERROR10_3 = ( (Y_diff) / len(input_x1) )**0.5
#
print("RMS Error more than 10 input polynomial degree 1 case = ", RMS_ERROR10_3)

#
#    


#print(final_out_array_y1)



#
############# Fitting degree 6th Polynomial, Regression to the dataset ##########
poly_reg6 = PolynomialFeatures(degree = 6)
X_poly6 = poly_reg6.fit_transform(input_x)
poly_reg6.fit(X_poly6, final_out_array_y)
lin_reg_6 = LinearRegression()
lin_reg_6.fit(X_poly6, final_out_array_y)
plt.scatter(input_x, final_out_array_y, color = 'red')
plt.plot(input_x, lin_reg_6.predict(poly_reg6.fit_transform(input_x)), color = 'blue')
plt.title('Polynomial Regression of Degree 6')
plt.xlabel('input x') 
plt.ylabel('output Y')
plt.show()

poly_reg6 = PolynomialFeatures(degree = 6)
X_poly6 = poly_reg6.fit_transform(input_x1)
poly_reg6.fit(X_poly6, final_out_array_y1)
lin_reg_6 = LinearRegression()
lin_reg_6.fit(X_poly6, final_out_array_y1)
output_y_value6 = lin_reg_6.predict(poly_reg6.fit_transform(input_x1))
plt.scatter(input_x1, final_out_array_y1, color = 'red')
plt.plot(input_x1, output_y_value6, color = 'blue')
plt.title('Polynomial Regression of Degree 6')
plt.xlabel('input x') 
plt.ylabel('output Y')
plt.show()

###############################################################################
############### RMS ERROR FOR degree 6 case #####################################


Y_diff = 0 
for i in range(len(input_x1)):
    Y_diff = Y_diff + (output_y_value6[i] - final_out_array_y1[i])**2
#
RMS_ERROR_6 = ( (Y_diff) / len(input_x1) )**0.5
#
print("RMS Error for polynomial degree 6 case = ", RMS_ERROR_6)

#
#
#
#
############# Fitting degree 9th Polynomial, Regression to the dataset ##########
poly_reg9 = PolynomialFeatures(degree = 9)
X_poly9 = poly_reg9.fit_transform(input_x)
poly_reg9.fit(X_poly9, final_out_array_y)
lin_reg_9 = LinearRegression()
lin_reg_9.fit(X_poly9, final_out_array_y)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,normalize=False)
## Visualising the Polynomial Regression results
plt.scatter(input_x, final_out_array_y, color = 'g')
plt.plot(input_x, lin_reg_9.predict(poly_reg9.fit_transform(input_x)), color = 'k')
plt.title('Polynomial Regression of Degree 9')
plt.xlabel('input x') 
plt.ylabel('output Y')
plt.show()
#
#
poly_reg9 = PolynomialFeatures(degree = 9)
X_poly9 = poly_reg9.fit_transform(input_x1)
#print(X_poly9)
poly_reg9.fit(X_poly9, final_out_array_y1)
lin_reg_9 = LinearRegression()
lin_reg_9.fit(X_poly9, final_out_array_y1)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,normalize=False)
output_y_value9 =  lin_reg_9.predict(poly_reg9.fit_transform(input_x1))
#print("output y = \n", output_y_value9)
## Visualising the Polynomial Regression results
plt.scatter(input_x1, final_out_array_y1, color = 'g')
plt.plot(input_x1,output_y_value9 , color = 'k')
plt.title('Polynomial Regression of Degree 9')
plt.xlabel('input x') 
plt.ylabel('output Y')
plt.show()



###############################################################################
############### RMS ERROR FOR degree 9 case #####################################


Y_diff = 0 
for i in range(len(input_x1)):
    Y_diff = Y_diff + (output_y_value9[i] - final_out_array_y1[i])**2
#
RMS_ERROR_9 =( (Y_diff) / len(input_x1) )**0.5
#
print("RMS Error for polynomial degree 9 case = ", RMS_ERROR_9)
#


degree_of_polynomial = [ 1,3,6,9]

degree_of_polynomial = np.reshape(degree_of_polynomial, (4,1))
print(degree_of_polynomial.shape)

RMS_value_of_10_data_point = []
RMS_value_of_more_than_10_data_point = []

#for i in range(len(degree_of_polynomial)):
#    RMS_value_of_10_data_point.append(0 + RMS_ERROR_9 )

RMS_value_of_10_data_point.append( RMS_ERROR_1 )
RMS_value_of_10_data_point.append( RMS_ERROR_3 )
RMS_value_of_10_data_point.append( RMS_ERROR_6 )
RMS_value_of_10_data_point.append( RMS_ERROR_9 )

RMS_value_of_10_data_point = np.reshape(RMS_value_of_10_data_point, (4,1))




      
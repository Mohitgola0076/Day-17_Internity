                # Data Analysis with SciPy : 
                

SciPy is a python library that is useful in solving many mathematical equations and algorithms. It is designed on the top of Numpy 
library that gives more extension of finding scientific mathematical formulae like Matrix Rank, Inverse, polynomial equations, LU 
Decomposition, etc. Using its high level functions will significantly reduce the complexity of the code and helps in better 
analyzing the data. SciPy is an interactive Python session used as a data-processing library that is made to compete with its 
rivalries such as MATLAB, Octave, R-Lab,etc. It has many user-friendly, efficient and easy-to-use functions that helps to solve 
problems like numerical integration, interpolation, optimization, linear algebra and statistics.

The benefit of using SciPy library in Python while making ML models is that it also makes a strong programming language available 
for use in developing less complex programs and applications.

              Example : 
            
# import numpy library
import numpy as np
A = np.array([[1,2,3],[4,5,6],[7,8,8]])


                               # Linear Algebra :
    # 1.) Determinant of a Matrix :

          # Example : 
# importing linalg function from scipy
from scipy import linalg

# Compute the determinant of a matrix
linalg.det(A)

          # Output :
2.999999999999997

                        # 2.) Compute pivoted LU decomposition of a matrix : 
                        
LU decomposition is a method that reduce matrix into constituent parts that helps in easier calculation of complex matrix 
operations. The decomposition methods are also called matrix factorization methods, are base of linear algebra in computers, even 
for basic operations such as solving systems of linear equations, calculating the inverse, and calculating the determinant of a 
matrix.
        # The decomposition is:
A = P L U
where P is a permutation matrix, L lower triangular with unit diagonal elements, and U upper triangular.

              # Example :
P, L, U = linalg.lu(A)
print(P)
print(L)
print(U)
# print LU decomposition
print(np.dot(L,U))

              # Output :
array([[ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 1.,  0.,  0.]])

array([[ 1.        ,  0.        ,  0.        ],
       [ 0.14285714,  1.        ,  0.        ],
       [ 0.57142857,  0.5       ,  1.        ]])

array([[ 7.        ,  8.        ,  8.        ],
       [ 0.        ,  0.85714286,  1.85714286],
       [ 0.        ,  0.        ,  0.5       ]])

array([[ 7.,  8.,  8.],
       [ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])
       
                         # 4.) Eigen values and eigen vectors of this matrix :
            
            # Example : 
eigen_values, eigen_vectors = linalg.eig(A)
print(eigen_values)
print(eigen_vectors)

                # Output :
array([ 15.55528261+0.j,  -1.41940876+0.j,  -0.13587385+0.j])

array([[-0.24043423, -0.67468642,  0.51853459],
       [-0.54694322, -0.23391616, -0.78895962],
       [-0.80190056,  0.70005819,  0.32964312]])
       
                            # Solving systems of linear equations can also be done : 
                            
                # Example : 
                
v = np.array([[2],[3],[5]])
print(v)
s = linalg.solve(A,v)
print(s)

         # Output :
array([[2],
       [3],
       [5]])

array([[-2.33333333],
       [ 3.66666667],
       [-1.        ]])
       
                                    #  Hypothesis testing: comparing two groups : 
                    
                    # Student’s t-test: the simplest statistical test :
                 
          # 1-sample t-test: testing the value of a population mean
            ../../_images/two_sided.png
scipy.stats.ttest_1samp() tests if the population mean of data is likely to be equal to a given value (technically if 
observations are drawn from a Gaussian distributions of given population mean). It returns the T statistic, and the p-value (see 
the function’s help):
                Example : 
                
>>> stats.ttest_1samp(data['VIQ'], 0) 
With a p-value of 10^-28 we can claim that the population mean for the IQ (VIQ measure) is not 0.

         # 2-sample t-test: testing for difference across populations
We have seen above that the mean VIQ in the male and female populations were different. To test if this is significant, we do a 2-sample t-test with scipy.stats.ttest_ind():


                                    # Paired tests: repeated measurements on the same individuals : 
                     
PIQ, VIQ, and FSIQ give 3 measures of IQ. Let us test if FISQ and PIQ are significantly different. We can use a 2 sample test:

               # Example :
>>> stats.ttest_ind(data['FSIQ'], data['PIQ'])                  

The problem with this approach is that it forgets that there are links between observations: FSIQ and PIQ are measured on the 
same individuals. Thus the variance due to inter-subject variability is confounding, and can be removed, using a “paired test”, 
or “repeated measures test”:

            # Example :
>>> stats.ttest_rel(data['FSIQ'], data['PIQ']) 


This is equivalent to a 1-sample test on the difference:

            # Example :
>>> stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0)


T-tests assume Gaussian errors. We can use a Wilcoxon signed-rank test, that relaxes this assumption:

                    # Example :
>>> stats.wilcoxon(data['FSIQ'], data['PIQ'])   

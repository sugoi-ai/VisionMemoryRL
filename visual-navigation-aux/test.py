import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
from eva2 import EVA
from op2 import OPT
import seaborn as sns


x1 = EVA['oven']
x2 = EVA['chair1']
x3 = EVA['tree3']
x4 = EVA['tree1']
x5 = EVA['fruit']
x6 = EVA['tree2']
x7 = EVA['knife']
x = x1+x2+x3+x4+x5+x6+x7


# library and data
import seaborn as sns
import matplotlib.pyplot as plt
 



y1 = OPT['oven']
y2 = OPT['chair1']
y3 = OPT['tree3']
y4 = OPT['tree1']
y5 = OPT['fruit']
y6 = OPT['tree2']
y7 = OPT['knife']
y = y1+y2+y3+y4+y5+y6+y7

z1 = [xi-yi for xi,yi in zip(x1,y1)]
z2 = [xi-yi for xi,yi in zip(x2,y2)]
z3 = [xi-yi for xi,yi in zip(x3,y3)]
z4 = [xi-yi for xi,yi in zip(x4,y4)]
z5 = [xi-yi for xi,yi in zip(x5,y5)]
z6 = [xi-yi for xi,yi in zip(x6,y6)]
z7 = [xi-yi for xi,yi in zip(x7,y7)]
z = [xi-yi for xi,yi in zip(x,y)]

# the histogram of the data
colors = ['r','b','y','back','pink','o','g']
bins = np.linspace(0, 500, 10)
plt.hist([x1, x2, x3, x4, x5, x6, x7], bins, label=['Oven', 'Chair','tree3','tree1','fruit','tree2','knife'])
plt.xlabel("Number of steps", color='black')
plt.ylabel("Number of tests", color='black')
plt.legend(loc='upper right')
plt.show()
	
'''
# the histogram of the data
colors = ['r','b','y','back','pink','o','g']
bins = np.linspace(0, 500, 100)
plt.hist(z, bins, label='overall')
plt.xlabel("Number of step", color='black')
plt.legend(loc='upper right')
plt.show()
'''

sub_1 = []
sub_2 = []
sub_3 = []

for index, value in enumerate(y):
	if value <= 15:
		sub_1.append(x[index]-value)
	elif value <=30:	
		sub_2.append(x[index]-value)
	else:	
		sub_3.append(x[index]-value)

colors = ['r','b','y']
bins = np.linspace(0, 500, 20)
plt.hist([sub_1, sub_2, sub_3], bins, label=['Group 1', 'Group 2','Group 3'])
plt.xlabel("Number of steps", color='black')
plt.ylabel("Number of tests", color='black')
plt.legend(loc='upper right')
plt.show()

'''
# plot
sns.distplot( sub_1 , color="r",hist=False)
sns.distplot( sub_2 , color="g",hist=False)
sns.distplot( sub_3 , color="b",hist=False)
plt.show()
'''

thres = [-1,10,20,30,40,50,100]
count_10 = 0
count_20 = 0
count_30 = 0
count_40 = 0
count_50 = 0

for index, value in enumerate(sub_1):
	if value <= 10:
		count_10 +=1
	elif value <=20:
		count_20 +=1
	elif value <=30:
		count_30 +=1
	elif value <=40:	
		count_40 +=1
	else:	
		count_50 +=1
print "sub_1"
print count_10, count_20, count_30, count_40, count_50


count_10 = 0
count_20 = 0
count_30 = 0
count_40 = 0
count_50 = 0

for index, value in enumerate(sub_3):
	if value <= 10:
		count_10 +=1
	elif value <=20:
		count_20 +=1
	elif value <=30:
		count_30 +=1
	elif value <=40:	
		count_40 +=1
	else:	
		count_50 +=1
print "sub_2"
print count_10, count_20, count_30, count_40, count_50

count_10 = 0
count_20 = 0
count_30 = 0
count_40 = 0
count_50 = 0

for index, value in enumerate(sub_3):
	if value <= 10:
		count_10 +=1
	elif value <=20:
		count_20 +=1
	elif value <=30:
		count_30 +=1
	elif value <=40:	
		count_40 +=1
	else:	
		count_50 +=1
print "sub_3"
print count_10, count_20, count_30, count_40, count_50



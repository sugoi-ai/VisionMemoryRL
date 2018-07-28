import numpy as np

b =np.array([5,5,5,5,5,5,5])
a = b
while a.shape[0]>1 and a[-1]==a[-2]:
  a = a[:-1]
print(a)  

print(b)
if a[-1] in a[:-1]:
  print(1)
else:
  print(0)   

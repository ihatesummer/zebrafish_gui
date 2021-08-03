from numpy import linspace

a = linspace(1,5,5)
b = a.copy()
a[4] = 10
b[0] = 0
for i in range(1, len(a)):
     b[i] = a[i] - a[i-1]
print(b)
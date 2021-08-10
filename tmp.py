import numpy as np
from os.path import exists
from os import remove
from datetime import datetime
import matplotlib.pyplot as plt



def get_first_available(y_part, bDetected_part):
    if bDetected_part[0] == True:
        return y_part[0]
    else:
        # Recursion until available
        return get_first_available(
            y_part[1:], bDetected_part[1:]
        )

def get_last_available(y_part, bDetected_part):
    if bDetected_part[-1] == True:
        return y_part[-1]
    else:
        # Recursion until available
        return get_last_available(
            y_part[:-1], bDetected_part[:-1]
        )

a = np.array([0.0, 2, 4, 0, 0])
b = a!=0
print(f"a: {a}")
print(f"b: {b}")
if b[0] == False:
    a[0] = get_first_available(a[1:], b[1:])
if b[4] == False:
    a[4] = get_last_available(a[:-1], b[:-1])
print(f"first and last fixed a: {a}")
for i in range (1, len(a)-1):
    if b[i] == False:
        if len(a[:i]) == 1:
            prev = a[:i]
        else: 
            prev = get_last_available(
                a[:i], b[:i])
        if len(a[i+1:]) == 1:
            next = a[i+1:]
        else:
            next = get_first_available(
                    a[i+1:], b[i+1:])
        a[i] = (prev + next) / 2
print(f"interpolated a: {a}")
a2 = a
print(np.where(a!=a2)[0])
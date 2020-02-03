from random import randint
from array import *
import numpy

rows, cols = (5, 10);

A = numpy.zeros(shape=(rows,cols))
B = numpy.zeros(shape=(rows,cols))

f = open("adj.txt","w+");
for i in range(matSize):
    for j in range(matSize):
        num = randint(0,10);
        if(j==matSize):
            f.write(str(num));
        else:
            f.write(str(num)+" ");
    f.write("\r")
f.close();

intCast = lambda l : [int(item) for item in l];

k=l=0;
fh=open("adj.txt");

for line in fh:
    if(k<=rows-1):
        x=line.split(" ");
        A[k]=intCast(x[:cols]);
        k+=1;
    else:
        x=line.split(" ");
        B[l]=intCast(x[:cols]);
        l+=1;
print(A)
print(B)

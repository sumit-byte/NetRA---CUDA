from random import randint
from array import *
rows, cols = (5, 10);

matSize = 10;
f = open("/home/netra/Desktop/Swaraj/adj.txt","w+");
for i in range(matSize):
    for j in range(matSize):
        num = randint(0,10);
        if(j==matSize):
            f.write(str(num));
        else:
            f.write(str(num)+" ");
    f.write("\r\n")
f.close();

intCast = lambda l : [int(item) for item in l];

rows, cols = (5, 10);
A = [[]]
B = [[]]
k=l=0;
fh=open("/home/netra/Desktop/Swaraj/adj.txt");

for line in fh:
    if(k<=rows-1):
        x=line.split(" ");
        A.insert(k,intCast(x[:cols]));
        k+=1;
    else:
        x=line.split(" ");
        B.insert(l,intCast(x[:cols]));
        l+=1;
del A[rows]
del B[rows]
print(A)
print(B)

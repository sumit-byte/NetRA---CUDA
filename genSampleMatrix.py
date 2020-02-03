from random import randint
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

import os
import openslide
import matplotlib
import numpy
import matplotlib.pyplot as plt


slide = openslide.OpenSlide("J:\\TDHEtif\\2018-02-04 13_41_38.kfb.tif")
dim=0

level_count = slide.level_count
print ('level_count = ', level_count)
[m,n] = slide.level_dimensions[dim] #得出高倍下的（宽，高）(97792,219648)
print (m,n)


tile = numpy.array(slide.read_region((10000,10000),dim, (15000,15000)))
plt.figure()
plt.imshow(tile)
# pylab.show()


slide.close()






# os.system("ls")
images_list = os.listdir('J:/TDHE/')
os.chdir("D:\\sharedhome\\KFB转Tif或SVS工具2.0\\x86\\")
# os.system("cd D:\\sharedhome\\KFB转Tif或SVS工具2.0\\x86\\")
# os.system("dir")
i = 0
for image_name in images_list:
    temp = "KFbioConverter.exe "+"\"J:\\TDHE\\"+image_name+"\""+" "+"\"J:\\TDHEtif\\"+image_name+".tif"+"\""+" 5"
    print(temp)
    os.system(temp)
    i=i+1
    print(i)
    if i==1:
        break


from PIL import Image
import os

# im = Image.open("lenna.jpg")
# im.show()
#
# # 指定逆时针旋转的角度
# im_rotate = im.rotate(45)
# im_rotate.show()

# data_dir = 'D:/sharedhome/models/data/GCtest/Cancer/'

# im = Image.open(data_dir+images_list[0])
# imrot1 = im.rotate(90)
# imrot2 = im.rotate(180)
# imrot3 = im.rotate(270)
# imrot1.save(data_dir+'r90_'+images_list[0])
# imrot2.save(data_dir+'r180_'+images_list[0])
# imrot3.save(data_dir+'r270_'+images_list[0])

def rot_dataset(data_dir):
    listimg = os.listdir(data_dir)
    for class_name in listimg:
        im = Image.open(data_dir + class_name)
        imrot1 = im.rotate(90)
        imrot2 = im.rotate(180)
        imrot3 = im.rotate(270)
        imrot1.save(data_dir + 'r90_' + class_name)
        imrot2.save(data_dir + 'r180_' + class_name)
        imrot3.save(data_dir + 'r270_' + class_name)

rot_dataset('D:/sharedhome/models/data/GCtest/Normal/')
rot_dataset('D:/sharedhome/models/data/GCtest/Cancer/')


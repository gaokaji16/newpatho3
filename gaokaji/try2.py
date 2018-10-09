import os

# class_names_to_ids = {'cat': 0, 'dog': 1}
# data_dir = 'D:/sharedhome/models/data/catdog/'
# output_path = 'D:/sharedhome/models/data/catdog/list.txt'

class_names_to_ids = {'Cancer': 0, 'Normal': 1}
data_dir = 'D:/sharedhome/models/data/GCtest2/'
output_path = 'D:/sharedhome/models/data/GCtest2/list.txt'




fd = open(output_path, 'w')
for class_name in class_names_to_ids.keys():
    images_list = os.listdir(data_dir + class_name)
    for image_name in images_list:
        fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))

fd.close()


import random

_NUM_VALIDATION = 200
_RANDOM_SEED = 0
# list_path = 'D:/sharedhome/models/data/catdog/list.txt'
# train_list_path = 'D:/sharedhome/models/data/catdog/list_train.txt'
# val_list_path = 'D:/sharedhome/models/data/catdog/list_val.txt'

list_path = 'D:/sharedhome/models/data/GCtest2/list.txt'
train_list_path = 'D:/sharedhome/models/data/GCtest2/list_train.txt'
val_list_path = 'D:/sharedhome/models/data/GCtest2/list_val.txt'

fd = open(list_path)
lines = fd.readlines()
fd.close()
random.seed(_RANDOM_SEED)
random.shuffle(lines)

fd = open(train_list_path, 'w')
for line in lines[_NUM_VALIDATION:]:
    fd.write(line)

fd.close()
fd = open(val_list_path, 'w')
for line in lines[:_NUM_VALIDATION]:
    fd.write(line)

fd.close()


import sys

# 注意这里的路径：
from research.slim.datasets import dataset_utils

import math
import os
import tensorflow as tf

def convert_dataset(list_path, data_dir, output_dir, _NUM_SHARDS=5):
    fd = open(list_path)
    lines = [line.split() for line in fd]
    fd.close()
    num_per_shard = int(math.ceil(len(lines) / float(_NUM_SHARDS)))
    with tf.Graph().as_default():
        decode_jpeg_data = tf.placeholder(dtype=tf.string)
        decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_path = os.path.join(output_dir,
                    'data_{:05}-of-{:05}.tfrecord'.format(shard_id, _NUM_SHARDS))
                tfrecord_writer = tf.python_io.TFRecordWriter(output_path)
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(lines))
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Converting image {}/{} shard {}'.format(
                        i + 1, len(lines), shard_id))
                    sys.stdout.flush()
                    image_data = tf.gfile.FastGFile(os.path.join(data_dir, lines[i][0]), 'rb').read()
                    image = sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
                    height, width = image.shape[0], image.shape[1]
                    example = dataset_utils.image_to_tfexample(
                        # image_data, b'jpg', height, width, int(lines[i][1]))
                        image_data, b'jpg', 400, 400, int(lines[i][1]))
                    tfrecord_writer.write(example.SerializeToString())
                tfrecord_writer.close()
    sys.stdout.write('\n')
    sys.stdout.flush()

# # os.system('mkdir -p train')其实就是建立一个目录，不建立会报错，后来手动建立的
# convert_dataset('D:/sharedhome/models/data/catdog/list_train.txt', 'D:/sharedhome/models/data/catdog/', 'D:/sharedhome/models/data/catdog/train/')
# # os.system('mkdir -p val')
# convert_dataset('D:/sharedhome/models/data/catdog/list_val.txt', 'D:/sharedhome/models/data/catdog/', 'D:/sharedhome/models/data/catdog/val/')

# os.system('mkdir -p train')其实就是建立一个目录，不建立会报错，后来手动建立的
convert_dataset('D:/sharedhome/models/data/GCtest2/list_train.txt', 'D:/sharedhome/models/data/GCtest2/', 'D:/sharedhome/models/data/GCtest2/train/')
# os.system('mkdir -p val')
convert_dataset('D:/sharedhome/models/data/GCtest2/list_val.txt', 'D:/sharedhome/models/data/GCtest2/', 'D:/sharedhome/models/data/GCtest2/val/')













from PIL import Image
# import os
# import tensorflow as tf
# filename='D:/sharedhome/models/data/catdog/train/'
# filename_queue = tf.train.string_input_producer([filename])
#
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)
#
# features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw' : tf.FixedLenFeature([], tf.string),
#                                        })
# image = tf.image.decode_jpeg(features['img_raw'])
#
#
# print(serialized_example)




import matplotlib
import matplotlib.pyplot as plt
import os
import tensorflow as tf
filename='D:/sharedhome/models/data/catdog/train/data_00000-of-00005.tfrecord'
filename_queue = tf.train.string_input_producer([filename]) #读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件

features = tf.parse_single_example(serialized_example,
                                   features={
                                                # 'image/encoded': bytes_feature(image_data),
                                                # 'image/format': bytes_feature(image_format),
                                                # 'image/class/label': int64_feature(class_id),
                                                # 'image/height': int64_feature(height),
                                                # 'image/width': int64_feature(width),
                                            'image/encoded': tf.FixedLenFeature([], tf.string),
                                            'image/format': tf.FixedLenFeature([], tf.string),
                                            'image/height': tf.FixedLenFeature([], tf.int64),
                                            'image/width': tf.FixedLenFeature([], tf.int64),
                                   })  #取出包含image和label的feature对象

# features = tf.parse_single_example(serialized_example)

image = tf.image.decode_jpeg(features['image/encoded'])

# image = tf.decode_raw(features['image/encoded'], tf.uint8)
height = tf.cast(features['image/height'], tf.int32)
width = tf.cast(features['image/width'], tf.int32)
# tf.train.shuffle_batch必须确定shape


# image = tf.decode_raw(features['img_raw'], tf.uint8)
# image = tf.reshape(image, [128, 128, 3])
# label = tf.cast(features['label'], tf.int32)
with tf.Session() as sess: #开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(2):
        example = sess.run([image,height,width])#在会话中取出image和label
        # a[i]=example
        # img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        # img.save('D:/PycharmProjects/models/data/train/'++str(i)+'.jpg')#存下图片
        # print(example)

    coord.request_stop()
    coord.join(threads)

# print(example[0][0])
# a=example[0][0]
# b=example[0][1]
# img=Image.fromarray(example, 'RGB')
# plot_images(example)
# plt.imshow(img)
#
# # newimage = tf.reshape(example[0], [example[1], example[2]])
# # newimage = tf.cast(newimage, tf.float32) / 255.0
#
#
# plt.imshow(a)
# plt.imshow(b)
# from PIL import Image
# im = Image.open(image)
# im.show()
plt.imshow(example[0])
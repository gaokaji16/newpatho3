import multiresolutionimageinterface as mir
reader = mir.MultiResolutionImageReader()
mr_image = reader.open('D:/sharedhome/TDHE2/2018-02-0413_41_38.tif')
output_path = 'D:/sharedhome/TDHE2/new/'
annotation_list = mir.AnnotationList()
xml_repository = mir.XmlRepository(annotation_list)
xml_repository.setSource('D:/sharedhome/TDHE2/2018-02-0413_41_38.xml')
xml_repository.load()
annotation_mask = mir.AnnotationToMask()
newtype = True
label_map = {'metastases': 1, 'normal': 2} if newtype else {'_0': 1, '_1': 1, '_2': 0}
conversion_order = ['metastases', 'normal'] if newtype else  ['_0', '_1', '_2']
annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)

import pylab
import matplotlib
import numpy
import matplotlib.pyplot as plt
import openslide
# slide = openslide.OpenSlide('D:\\sharedhome\\TDHE2\\2018-02-0413_41_38.tif')
slide = openslide.OpenSlide('J:\\TDHE\\tif\\2018-02-04 13_48_24.tif')

level_count = slide.level_count
print('level_count = ', level_count)
[m, n] = slide.dimensions #得出高倍下的（宽，高）(97792,219648)
print(m, n)

[m1,n1] = slide.level_dimensions[2] #级别k，且k必须是整数，下采样因子和k有关
print(m1,n1)      # m1 = m/下采样因子 此时k为1




tile = numpy.array(slide.read_region((500,500),2, (2000,2000)))
plt.figure()
plt.imshow(tile)
pylab.show()

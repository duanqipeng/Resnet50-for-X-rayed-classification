from PIL import Image, ImageFilter,ImageEnhance
import glob as gb
import os
import time
import random
img_path = gb.glob(r"C:\Users\uestc\Desktop\项目\DatasetUtil\dataset/*.jpg")
for path1 in img_path:
    # jiaodu=random.randint(0,360)
    # transint=random.randint(0,1)
    (filepath, tempfilename) = os.path.split(path1)
    (filename, extension) = os.path.splitext(tempfilename)
    img=Image.open(path1)
    # if transint==1:
    #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # else:
    #     out = img.transpose(Image.FLIP_TOP_BOTTOM)
    # img=img.filter(ImageFilter.BLUR)
    img=img.resize((895, 600), Image.ANTIALIAS)  # resize image with high-quality
    # brightness = ImageEnhance.Brightness(img)
    # im_brightness = brightness.enhance(0.7)
    #
    # # img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # out = im_brightness.save(filepath+'/'+filename+str(time.time())+extension)
    out = img.save(path1)
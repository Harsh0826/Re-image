import sys
from PIL import Image

image_fullpath = sys.argv[1]
image_name = sys.argv[2]
image_format = sys.argv[3]
img = Image.open(str(image_fullpath)).convert("RGB")
image_save_path = image_fullpath.replace(image_name,"temp."+image_format)
img.save(image_save_path,image_format)
print("/media/temp."+image_format)
from PIL import Image
import os, sys

def resize(c):
    path = "Iranis_Dataset_Files"+"/"+c+"/"
    dirs = os.listdir( path )
    if not os.path.isdir("Iranis_Dataset_Files_resized"):
        os.mkdir("Iranis_Dataset_Files_resized")
    if not os.path.isdir("Iranis_Dataset_Files_resized"+"/"+c+"/"):
        os.mkdir("Iranis_Dataset_Files_resized"+"/"+c+"/")
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((30,30), Image.ANTIALIAS)
            imResize.save(f.replace("Iranis_Dataset_Files","Iranis_Dataset_Files_resized") + '.jpg', 'JPEG', quality=100)


# for c in ["0","1","2","3","4","5","6","7","8","9","A","B","D",
#           "Gh","H","J","L","M","N","P","PuV","PwD","Sad","Sin",
#           "T","Taxi","V","Y"]:
#     resize(c)

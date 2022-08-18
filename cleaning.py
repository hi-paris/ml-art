import os
from PIL import Image
import cv2

# folder_path = r"/home/infres/ext-6343/venv_ml_art/ml-art/data"
# extensions = []
# corupt_img_paths=[]

# for fldr in os.listdir(folder_path):
#     sub_folder_path = os.path.join(folder_path, fldr)
#     for filee in os.listdir(sub_folder_path):
#         file_path = os.path.join(sub_folder_path, filee)
#         # print('** Path: {}  **'.format(file_path), end="\r", flush=True)
#         try:
#             im = Image.open(file_path)
#         except:
#             print(file_path)
#             corupt_img_paths.append(file_path)
#             continue
#         else:
#             rgb_im = im.convert('RGB')
#             if filee.split('.')[1] not in extensions:
#                 extensions.append(filee.split('.')[1])

# print("*********LEN ***************",len(corupt_img_paths))


# rename
def rename_images(folder_path):
    """Change images name
Args:
    folder_path (str): the path of the folder containing images.
    """
    for fldr in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, fldr)
        i=0
        for filee in os.listdir(sub_folder_path):
            os.rename(sub_folder_path+"/"+filee,sub_folder_path+"/"+str(i)+".png")
            i=i+1


import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

#显示一个系列图
def show_img(data):
    for i in range(data.shape[0]):
        cv2.imwrite('test.png', data[i,:,:])
        print(i)
 
#单张显示
def show_img_single(ori_img):
    cv2.imwrite('test.png', ori_img[100])

if __name__ == "__main__":
    #window下的文件夹路径 
    img_path = '/home/xiaoguai0992/covid-19-20/COVID-19-20_v2/Train/volume-covid19-A-0011_ct.nii.gz'
    mask_path = '/home/xiaoguai0992/covid-19-20/COVID-19-20_v2/Train/volume-covid19-A-0011_seg.nii.gz'
    img = read_img(img_path)
    mask = read_img(mask_path)
    # show_img_single(data)
    for i in tqdm(range(len(img))):
        cv2.imwrite('/home/xiaoguai0992/covid-19-20/Temp/img/'+str(i)+'.png', img[i])
        cv2.imwrite('/home/xiaoguai0992/covid-19-20/Temp/mix/'+str(i)+'_mix.png', np.hstack([img[i],mask[i]*255]))
        cv2.imwrite('/home/xiaoguai0992/covid-19-20/Temp/mask/'+str(i)+'_mask.png', np.array(mask[i]*255).astype('uint8'))

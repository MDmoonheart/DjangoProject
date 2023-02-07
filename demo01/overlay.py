import radiomics
from matplotlib import pyplot as plt
from radiomics import featureextractor
import pandas as pd
import openpyxl
#coding=utf-8
import SimpleITK as sitk

dataDir = 'E:/SZH Data'
#folderList = ['001','002','003','004','005']
settings = {'label': 2}
extractor = featureextractor.RadiomicsFeatureExtractor(additionalInfo=True, **settings,geometryTolerance = 1e-5)
df = pd.DataFrame()


imageName = 'C:/Users/VM01/Desktop/test.nii.gz'
maskName = dataDir + '/SZH_Contour/SZH_0014_P2.nii.gz'
featureVector = extractor.execute(imageName, maskName)
print(type(featureVector))
print(featureVector)
# df_add = pd.DataFrame.from_dict(featureVector.values()).T
# df_add.columns = featureVector.keys()
# df = pd.concat([df,df_add])
# df.to_excel(dataDir + 'results.xlsx')

itk_img=sitk.ReadImage(maskName)
img = sitk.GetArrayFromImage(itk_img)
img.shape   # 查看形状 单通道灰度图片
plt.imshow(img[0])  # 展示其中一张有mask的图片
plt.imsave('C:/Users/VM01/Desktop/14.png',img[0],cmap='gray')



# def dcm2nii(dcms_path, nii_path):
# 	# 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
#     reader.SetFileNames(dicom_names)
#     image2 = reader.Execute()
# 	# 2.将整合后的数据转为array，并获取dicom文件基本信息
#     image_array = sitk.GetArrayFromImage(image2)  # z, y, x
#     origin = image2.GetOrigin()  # x, y, z
#     spacing = image2.GetSpacing()  # x, y, z
#     direction = image2.GetDirection()  # x, y, z
# 	# 3.将array转为img，并保存为.nii.gz
#     image3 = sitk.GetImageFromArray(image_array)
#     image3.SetSpacing(spacing)
#     image3.SetDirection(direction)
#     image3.SetOrigin(origin)
#     sitk.WriteImage(image3, nii_path)
#
#
# if __name__ == '__main__':
#     dcms_path = r'E:\SZH Data\Images + Radiology reports\SZH0014\P2'  # dicom序列文件所在路径
#     nii_path = r'C:\Users\VM01\Desktop\test.nii.gz'  # 所需.nii.gz文件保存路径
#     dcm2nii(dcms_path, nii_path)




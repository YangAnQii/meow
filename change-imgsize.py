import cv2
import os

# 读取图片
image = cv2.imread("E:\MatlabProject\LIDC-IDRI jianqie\\5\\5\\0423-00109.jpg")
str0 = 'E:\MatlabProject\LIDC-IDRI jianqie\\5\\5\\'
filenames=os.listdir('E:\MatlabProject\LIDC-IDRI jianqie\\5\\5')
path = 'E:\pythonProject\LIDC-IDRI-train\\5\\'
for i in filenames:
    #print(str0+str(i))
    image = cv2.imread(str0+str(i))
    image1 = cv2.resize(image, (64, 64))
    cv2.imwrite(path+str(i), image1)


# 图片缩放
# image1 = cv2.resize(image, (56, 56))
# # 图片显示
# cv2.imshow("resize", image1)
# cv2.imshow("image", image)
#
# # 等待窗口
# cv2.waitKey(0),
# cv2.destroyAllWindows()



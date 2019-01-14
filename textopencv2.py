# -*- coding: utf-8 -*-
"""
Created on Wed May 02 14:51:25 2018

@author: asuspc
"""

import cv2
import numpy as np


def Sobel(pi):#索贝尔评价函数
  cc=0.0
  m,n=pi.shape
  for i in range (1,m-1):
      for j in range (1,n-1):
        gx=(pi[i+1,j-1]+2*pi[i+1,j]+pi[i+1,j+1])-(pi[i-1,j-1]+2*pi[i-1,j]+pi[i-1,j+1])
        gy=(pi[i-1,j+1]+2*pi[i,j+1]+pi[i+1,j+1])-(pi[i-1,j-1]+2*pi[i,j-1]+pi[i+1,j-1])
        cc=cc+gx*gx+gy*gy
  return cc        


def laplacian(pi):#拉普拉斯评价函数
   cc=0.0
   m,n=pi.shape
   for i in range (1,m-1):
       for j in range (1,n-1):
         cc=cc+(20*pi[i,j]-4*pi[i,j-1]-4*pi[i,j+1]-4*pi[i-1,j]-4*pi[i+1,j]-pi[i+1,j+1]-pi[i+1,j-1]-pi[i-1,j-1]-pi[i-1,j+1]);
   return cc


def mean(img): # 均值
    cc=0.0
    m,n=img.shape
    for i in range (0,m):
        for j in range (0,m):
            cc=cc+img[i,j]
    return cc
    

def ahash(img1,img2): #哈希算法比较两幅图像的相似性
   tmp=cv2.resize(img1,(8,8),interpolation=cv2.INTER_CUBIC)
   tmp2=cv2.resize(img2,(8,8),interpolation=cv2.INTER_CUBIC)
   tmp=np.float32(tmp/4.0)
   tmp2=np.float32(tmp2/4.0)
   a,b=tmp.shape
   re=np.zeros((a,b))
   m=mean(tmp)
   n=mean(tmp2)
   me=m/64
   me2=n/64
   #print m,n,me,me2
   #print tmp
   for i in range (0,8):
      for j in range (0,8):
        if tmp[i,j]>me:tmp[i,j]=1 
        else:tmp[i,j]=0       
         
        if tmp2[i,j]>me2:tmp2[i,j]=1         
        else:tmp2[i,j]=0        

        re[i,j]=tmp[i,j]-tmp2[i,j]
   
   num=0
   #print re
   for i in range (0,8):
      for j in range (0,8):
          if re[i,j]!=0:
            num=num+1
   #print num
   return num


lx=9.0
ly=9.0
pic = cv2.imread('1.png')
#cv2.imshow('pic',pic)
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
m,n=pic.shape
m=int(m)
n=int(n)
depth=np.ones((m,n))


#pinhole = cv2.imread('2.jpg')
#pinhole = cv2.cvtColor(pinhole, cv2.COLOR_BGR2GRAY)
allfocus=pic


for num in range (2,4):
    print num
    name = str(num)+'.png'
    img = cv2.imread(name)
#    cv2.imshow('name',img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range (int(lx/2)+1,m-int(lx/2)-1,2):
        for j in range (int(ly/2)+1,n-int(ly/2)-1,2):
            aa=allfocus[i-int(lx/2):i+int(lx/2),j-int(ly/2):j+int(ly/2)]
            bb=img[i-int(lx/2):i+int(lx/2),j-int(ly/2):j+int(ly/2)]
#            pin=pinhole[i-int(lx/2):i+int(lx/2),j-int(ly/2):j+int(ly/2)]
            pd_forward=Sobel(aa)
            pd_next=Sobel(bb)
            pd_f=laplacian(aa)
            pd_n=laplacian(bb)
            #pd_fh=ahash(aa,pin)
            #pd_nh=ahash(bb,pin)
            if pd_forward>pd_next or pd_f>pd_n:
                depth[i-int(lx/2):i+int(lx/2),j-int(ly/2):j+int(ly/2)]=depth[i-int(lx/2):i+int(lx/2),j-int(ly/2):j+int(ly/2)]
                allfocus[i-int(lx/2):i+int(lx/2),j-int(ly/2):j+int(ly/2)]=allfocus[i-int(lx/2):i+int(lx/2),j-int(ly/2):j+int(ly/2)]
            else:
                depth[i-int(lx/2):i+int(lx/2),j-int(ly/2):j+int(ly/2)]=num
                allfocus[i-int(lx/2):i+int(lx/2),j-int(ly/2):j+int(ly/2)]=img[i-int(lx/2):i+int(lx/2),j-int(ly/2):j+int(ly/2)]


depth=depth.astype(np.uint8)
cv2.imshow('allfocus',allfocus)
cv2.imshow('depth',depth*50)
cv2.imwrite('allfocus-temp.png',allfocus)
cv2.imwrite('depth-temp.png',depth*50)
cv2.waitKey(0)
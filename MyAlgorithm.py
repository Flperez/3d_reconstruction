# -*- coding: utf-8 -*-

import sys
from docutils.nodes import image
from sensors import sensor
import numpy as np
import threading
from pyProgeo.progeo import Progeo
import cv2
import os
import time
from matplotlib import pyplot as plt


class MyAlgorithm():

    def __init__(self, sensor):
        self.sensor = sensor
        self.imageRight=np.zeros((320,240,3), np.uint8)
        self.imageLeft=np.zeros((320,240,3), np.uint8)
        self.lock = threading.Lock()

        print("Left Camera Configuration File:")
        self.camLeftP=Progeo(sys.argv[1], "CamACalibration")
        print("Rigth Camera Configuration File:")
        self.camRightP=Progeo(sys.argv[1], "CamBCalibration")

        self.done=False

        self.counter=0

    def setRightImageFiltered(self, image):
        self.lock.acquire()
        size=image.shape
        if len(size) == 2:
            image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        self.imageRight=image
        self.lock.release()


    def setLeftImageFiltered(self, image):
        self.lock.acquire()
        size=image.shape
        if len(size) == 2:
            image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        self.imageLeft=image
        self.lock.release()

    def writeTxt(self,path,lst_points):
        with open(path,'w') as file:
            for idx in range(lst_points.shape[0]):
                file.write("%f %f %f %d %d %d\n"%(lst_points[idx,0],lst_points[idx,1],lst_points[idx,2],
                                                  lst_points[idx, 3],lst_points[idx,4],lst_points[idx,5]))

    def correlation_coefficient(self,patch1, patch2):
        if patch1.shape[1] < patch2.shape[1]:
            patch2 = patch2[:,0:patch1.shape[1]]

        if patch2.shape[1] < patch1.shape[1]:
            patch1 = patch1[:, 0:patch2.shape[1]]

        product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
        stds = patch1.std() * patch2.std()
        if stds == 0:
            return 0
        else:
            product /= stds
            return product

    def findKeypoints(self,imageR,imageL):
        grayR = cv2.cvtColor(imageR,cv2.COLOR_RGB2GRAY)

        edgesR = cv2.Canny(grayR, 100, 200)
        keypointsR = np.asarray(np.where(edgesR == 255)).T

        keypointsR = np.concatenate((keypointsR, np.ones((keypointsR.shape[0], 1))), 1)

        grayL = cv2.cvtColor(imageL,cv2.COLOR_RGB2GRAY)

        edgesL = cv2.Canny(grayL, 100, 200)
        keypointsL =  np.asarray(np.where(edgesL == 255)).T
        keypointsL = np.concatenate((keypointsL, np.ones((keypointsL.shape[0], 1))), 1)


        self.setRightImageFiltered(cv2.cvtColor(edgesR,cv2.COLOR_GRAY2RGB))
        self.setLeftImageFiltered(cv2.cvtColor(edgesL,cv2.COLOR_GRAY2RGB))

        return keypointsR, keypointsL,edgesR,edgesL



    def getVectors(self,pointsUV1,posCam):
        pointInOpts = np.asarray([self.camLeftP.graficToOptical(np.array([pointIn[1],pointIn[0],1]))
                                  for pointIn in pointsUV1])
        point3ds = np.asarray([self.camLeftP.backproject(pointInOpt) for pointInOpt in pointInOpts])
        return np.concatenate((point3ds[:,:3]-posCam*np.ones(pointInOpts.shape), np.ones((pointInOpts.shape[0], 1))), 1)


    def getPointsfromline(self,vectors,Origin,M=120):
        lst = [[] for N in range(vectors.shape[0])]
        alfas = np.linspace(1,M,M)
        for idx,Vxyz in enumerate(vectors):
            for alfa in alfas:
                xyz = Origin + Vxyz[:3]*alfa
                lst[idx].append(np.array([xyz[0],xyz[1],xyz[2],1]))

        return lst

    def getlstProjectedPoints(self,lst3d):
        projected_lst = [[] for N in range(len(lst3d))]
        projected_lst_grafic = [[] for N in range(len(lst3d))]
        for N in range(len(lst3d)):
            if N%100==0:
                print N,"/",len(lst3d)
            for idx,xyz in enumerate(lst3d[N]):
                projected = self.camRightP.project(xyz)
                projected_grafic = np.floor(self.camRightP.opticalToGrafic(projected)).astype(np.int)
                projected_lst[N].append(projected)
                projected_lst_grafic[N].append(np.array([projected_grafic[1],projected_grafic[0],1]))

        return projected_lst,projected_lst_grafic

    def drawPoint(self,image,lstPoints,color,idx):
        if len(image.shape)==2:
            out = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        else:
            out = image.copy()

        for uv in lstPoints:
            if uv:
                out[uv[0],uv[1]]=color
        if type(lstPoints) == list:
            cv2.circle(out,(lstPoints[idx][1],lstPoints[idx][0]),3,(0,255,255))
        else:
            cv2.circle(out, (lstPoints[idx,1], lstPoints[idx,0]), 3, (0, 255, 255))
        return out


    def matchingPoints(self,keyPointsL,lst2matchingR,imageR,imageL,edgesR,edgesL,size=(5,5),save=False):
        hsvR = cv2.cvtColor(imageR,cv2.COLOR_RGB2HSV)
        hsvL = cv2.cvtColor(imageL,cv2.COLOR_RGB2HSV)
        alfa = 0.5
        beta = 0.5
        incu = int(size[0]/2)
        incv = int(size[1]/2)

        matchingR = [[] for n in range(keyPointsL.shape[0])]

        for idx,uvL in enumerate(keyPointsL):
            # print "uvL,idx", uvL,idx
            patchL = hsvL[uvL[0]-incu:uvL[0]+incu+1,uvL[1]-incv:uvL[1]+incv+1,:]
            maximum = 0
            best = None
            # eliminamos repetidos
            lst2matchingR_unique=np.array(list(set(tuple(p) for p in np.asarray(lst2matchingR[idx]))))
            for jdx,uvR in enumerate(lst2matchingR_unique):
                # print "uvR,jdx",uvR,jdx
                # Solo buscamos el matching si es un pixel de contorno
                if edgesR[uvR[0],uvR[1]]==255:
                    patchR = hsvR[uvR[0] - incu:uvR[0] + incu+1, uvR[1] - incv:uvR[1] + incv+1, :]
                    corr = alfa*MyAlgorithm.correlation_coefficient(self,patchL[:,:,0],patchR[:,:,0])\
                           +beta*MyAlgorithm.correlation_coefficient(self,patchR[:,:,1],patchR[:,:,1])
                    if corr > maximum:
                        maximum = corr
                        best = uvR


            matchingR[idx] = best.astype(np.int)
            outR = MyAlgorithm.drawPoint(self,edgesR,matchingR,(255,0,0))

            outL = MyAlgorithm.drawPoint(self,edgesL,keyPointsL[0:idx+1,:].astype(np.int),(0,255,0))
            self.setRightImageFiltered(outR)
            self.setLeftImageFiltered(outL)

        if save:
            np.save("matchingR.npy",matchingR)

        return matchingR

    def get3dColor(self,pointsL,pointsR,imageL):
        vectorR = MyAlgorithm.getVectors(self, pointsR, self.camRightP.getCameraPosition())

        return None











    def execute(self):

        # OBTENCIÓN DE IMÁGENES
        imageLeft = self.sensor.getImageLeft()
        imageRight = self.sensor.getImageRight()

        os.system("killall gzserver")
        save= True


        # KEYPOINTS
        print "Calculating canny"
        keypointsR, keypointsL,edgesR,edgesL = MyAlgorithm.findKeypoints(self,imageR=imageRight,imageL=imageLeft)
        print "Keypoints Right: {:d}\t Keypoints Left: {:d}".format(keypointsR.shape[0],keypointsL.shape[0])
        print "Done!"

        print "Calculating all vectors"
        # Lines
        vectorL = MyAlgorithm.getVectors(self,keypointsL,self.camLeftP.getCameraPosition())

        # vectorR = MyAlgorithm.getVectors(self,keypointsR,self.camRightP.getCameraPosition())

        # Get M points from N vectors: NxM
        print "Calculating list of 3D points "
        lstPoints3d_L = MyAlgorithm.getPointsfromline(self,vectorL,self.camLeftP.getCameraPosition(),M=100)
        print "Dimension",len(lstPoints3d_L),"x",len(lstPoints3d_L[0])

        # Get NxM projected points from NxM
        print "Projecting points"
        projected_R,projected_R_Grafic = MyAlgorithm.getlstProjectedPoints(self,lstPoints3d_L)

        # Matching point
        print "Matching"
        matchingR = MyAlgorithm.matchingPoints(self,keypointsL,projected_R_Grafic,imageRight,
                                               imageLeft,edgesR,edgesL,save=save)
        print "Done!"
        lst3dcolor = MyAlgorithm.get3dColor(self,keypointsL,matchingR,imageLeft)







        if self.done:
            return


        # Add your code here
        # pointIn=np.array([502,21,1])
        # pointInOpt=self.camLeftP.graficToOptical(pointIn)
        # point3d=self.camLeftP.backproject(pointInOpt)
        # projected1 = self.camRightP.project(point3d)
        # print (self.camRightP.opticalToGrafic(projected1))

        #EXAMPLE OF HOW TO SEND INFORMATION TO THE ROBOT ACTUATORS
        #self.sensor.setV(10)
        #self.sensor.setW(5)


        #SHOW THE FILTERED IMAGE ON THE GUI
        # self.setRightImageFiltered(imageRight)
        # self.setLeftImageFiltered(imageLeft)

        #PLOT 3D data on the viewer
        #point=np.array([1, 1, 1])
        #self.sensor.drawPoint(point,(255,255,255))


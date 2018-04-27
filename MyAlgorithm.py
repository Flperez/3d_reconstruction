# -*- coding: utf-8 -*-

import sys
from docutils.nodes import image
from sensors import sensor
import numpy as np
import threading
from pyProgeo.progeo import Progeo
import cv2

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

    def findKeypoints(self,imageR,imageL):
        grayR = cv2.cvtColor(imageR,cv2.COLOR_RGB2GRAY)

        edgesR = cv2.Canny(grayR, 100, 200)
        keypointsR = np.asarray(np.where(edgesR == 255)).T

        keypointsR = np.concatenate((keypointsR, np.ones((keypointsR.shape[0], 1))), 1)

        grayL = cv2.cvtColor(imageL,cv2.COLOR_RGB2GRAY)

        edgesL = cv2.Canny(grayL, 100, 200)
        keypointsL =  np.asarray(np.where(edgesL == 255)).T
        keypointsL = np.concatenate((keypointsL, np.ones((keypointsL.shape[0], 1))), 1)
        return keypointsR, keypointsL



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
            for idx,xyz in enumerate(lst3d[N]):
                projected = self.camRightP.project(xyz)
                projected_grafic = np.floor(self.camRightP.opticalToGrafic(projected)).astype(np.int)
                projected_lst[N].append(projected)
                projected_lst_grafic[N].append(np.array([projected_grafic[1],projected_grafic[0],1]))

        return projected_lst,projected_lst_grafic


    def matchingPoints(self,keyPointsR,lst2matchingL,imageR,ImageL):






    def execute(self):

        # OBTENCIÓN DE IMÁGENES
        imageLeft = self.sensor.getImageLeft()
        imageRight = self.sensor.getImageRight()

        # KEYPOINTS
        keypointsR, keypointsL = MyAlgorithm.findKeypoints(self,imageR=imageRight,imageL=imageLeft)

        # Lines
        vectorL = MyAlgorithm.getVectors(self,keypointsL,self.camLeftP.getCameraPosition())
        vectorR = MyAlgorithm.getVectors(self,keypointsR,self.camRightP.getCameraPosition())

        # Get M points from N vectors: NxM
        lstPoints3d_L = MyAlgorithm.getPointsfromline(self,vectorL,self.camLeftP.getCameraPosition(),M=100)

        # Get NxM projected points from NxM
        projected_R,projected_R_Grafic = MyAlgorithm.getlstProjectedPoints(self,lstPoints3d_L)

        # Get points to grafic




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


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
import sys
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
            for xyzRGB in lst_points:
                file.write("%f;%f;%f;%d;%d;%d\n"%(xyzRGB[0],xyzRGB[1],xyzRGB[2],
                                                  xyzRGB[3],xyzRGB[4],xyzRGB[5]))

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


    def getPointsfromline(self,vectors,Origin,M=150):
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

    def drawPoint(self,image,lstPoints,idx,color):
        out = image.copy()
        out[lstPoints[idx,0],lstPoints[idx,1]]=color
        return out

    def drawLastPoint(self,image,lstPoints,idx,color=(0,255,255)):
        out = image.copy()
        cv2.circle(out, (lstPoints[idx, 1], lstPoints[idx, 0]), 3,color, 2)
        return out

    def matchingPoints(self,keyPointsL,lst2matchingR,imageR,imageL,edgesR,edgesL,size=(5,5),save=False):
        hsvR = cv2.cvtColor(imageR,cv2.COLOR_RGB2HSV)
        hsvL = cv2.cvtColor(imageL,cv2.COLOR_RGB2HSV)
        alfa = 0.75
        beta = 0.25
        incu = int(size[0]/2)
        incv = int(size[1]/2)
        ind_not_match = []

        matchingR = np.zeros(keyPointsL.shape)

        outR = cv2.cvtColor(edgesR,cv2.COLOR_GRAY2RGB)
        outL = cv2.cvtColor(edgesL,cv2.COLOR_GRAY2RGB)

        for idx,uvL in enumerate(keyPointsL):
            if idx%100==0:
                print idx,'/',keyPointsL.shape[0]
            # print "uvL,idx", uvL,idx
            patchL = hsvL[uvL[0]-incu:uvL[0]+incu+1,uvL[1]-incv:uvL[1]+incv+1,:]
            maximum = 0
            best = np.zeros([3,])
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

            if tuple(best) == tuple(np.zeros((3,))):
                print "Point not match: ",uvL,idx
                ind_not_match.append(idx)
                # delete point


            matchingR[idx,:] = best.astype(np.int)

            # Draw matching
            outR = MyAlgorithm.drawPoint(self,outR,matchingR.astype(np.int),idx,(255,0,0))
            outL = MyAlgorithm.drawPoint(self,outL,keyPointsL.astype(np.int),idx,(0,255,0))

            self.setRightImageFiltered(MyAlgorithm.drawLastPoint(self,outR,matchingR.astype(np.int),idx))
            self.setLeftImageFiltered(MyAlgorithm.drawLastPoint(self,outL,keyPointsL.astype(np.int),idx))


        #Borramos aquellos puntos donde no se ha hecho correctamente el matching
        matchingR = np.delete(matchingR, ind_not_match, 0)
        keyPointsL = np.delete(keyPointsL,ind_not_match,0)
        if save:
            np.save("matchingR.npy",matchingR)

        return keyPointsL,matchingR,ind_not_match


    def getpointsMinimunDistance(self,vL,vR,OL,OR):

        OLR = OR - OL
        a11 = np.dot(+vR, vL)
        a12 = np.dot(-vL, vL)
        a21 = np.dot(vR, vR)
        a22 = np.dot(-vL, vR)


        b1 = np.dot(OLR, vL)
        b2 = np.dot(OLR, vR)

        a = np.array([[a11, a12], [a21, a22]])
        b = np.array([b1, b2])

        ts = np.linalg.solve(a, b)

        PL = OL + vL * ts[0]
        PR = OR + vR * ts[1]
        return PL,PR


    def get3dColor(self,matchingR,keyPointsL,vectorL,ind_not_match,imageLeft):
        vectorR = MyAlgorithm.getVectors(self, matchingR, self.camRightP.getCameraPosition())
        vectorL =  np.delete(vectorL,ind_not_match,0)
        OL = self.camLeftP.getCameraPosition()
        OR = self.camRightP.getCameraPosition()
        lstPL = []
        lstPR = []
        lstPLR_color = []
        for idx,vL in enumerate(vectorL):
            vR = vectorR[idx,:3]
            vL = vL[:3]
            PL,PR = MyAlgorithm.getpointsMinimunDistance(self,vL,vR,OL,OR)
            lstPL.append(PL)
            lstPR.append(PR)
            average = np.mean((PL,PR),0).tolist()
            color = MyAlgorithm.getColor(self,imageLeft,keyPointsL[idx,:])
            lstPLR_color.append(average+color)

        return lstPLR_color


    def getColor(self,image,point):
        return image[point[0],point[1],:].tolist()









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

        print "Calculating all vectors from left camera"
        # Lines
        vectorL = MyAlgorithm.getVectors(self,keypointsL,self.camLeftP.getCameraPosition())

        # vectorR = MyAlgorithm.getVectors(self,keypointsR,self.camRightP.getCameraPosition())

        # Get M points from N vectors: NxM
        print "Calculating list of 3D points "
        lstPoints3d_L = MyAlgorithm.getPointsfromline(self,vectorL,self.camLeftP.getCameraPosition(),M=150)
        print "Dimension",len(lstPoints3d_L),"x",len(lstPoints3d_L[0])

        # Get NxM projected points from NxM
        print "Projecting points"
        projected_R,projected_R_Grafic = MyAlgorithm.getlstProjectedPoints(self,lstPoints3d_L)

        # Matching point
        print "Matching"
        keyPointsL, matchingR, ind_not_match = MyAlgorithm.matchingPoints(self,keypointsL,
                                                                        projected_R_Grafic,imageRight,
                                                                        imageLeft,edgesR,edgesL,save=save)
        print "Done!"

        print "Triangulating points"
        lst3dcolor = MyAlgorithm.get3dColor(self,matchingR,keyPointsL,vectorL,ind_not_match,imageLeft)

        write = True
        if write:
            print "Writing data"
            MyAlgorithm.writeTxt(self,"ptsColor.txt",lst3dcolor)
            print "Done!"
            sys.exit()

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


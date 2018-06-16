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
import sys,math
from matplotlib import pyplot as plt
from numpy import linalg as LA

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

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

    def writeTxt(self,path,lst_points,mode = "lista"):
        if mode=="lista":
            with open(path,'w') as file:

                for xyzRGB in lst_points:
                    file.write("%f;%f;%f;%d;%d;%d\n"%(xyzRGB[0],xyzRGB[1],xyzRGB[2],
                                                      xyzRGB[3],xyzRGB[4],xyzRGB[5]))
        else:
            with open(path, 'a') as file:
                file.write("%f;%f;%f;%d;%d;%d\n" % (lst_points[0], lst_points[1], lst_points[2],
                                                    lst_points[3], lst_points[4], lst_points[5]))


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

        return keypointsR.astype(np.int), keypointsL.astype(np.int),edgesR,edgesL



    def getVectors(self,pointsUV1,posCam,cam="left"):
        if cam=="left":
            if len(pointsUV1.shape)==1: #un unico punto
                pointInOpts = self.camLeftP.graficToOptical(np.array([pointsUV1[1], pointsUV1[0], 1]))
                point3ds = self.camLeftP.backproject(pointInOpts)
                return np.asarray((point3ds[:3]-posCam).tolist()+[1])
            else:
                pointInOpts = np.asarray([self.camLeftP.graficToOptical(np.array([pointIn[1],pointIn[0],1]))
                                          for pointIn in pointsUV1])
                point3ds = np.asarray([self.camLeftP.backproject(pointInOpt) for pointInOpt in pointInOpts])
        else:
            if len(pointsUV1.shape)==1: #un unico punto
                pointInOpts = self.camRightP.graficToOptical(np.array([pointsUV1[1], pointsUV1[0], 1]))
                point3ds = self.camRightP.backproject(pointInOpts)
                return np.asarray((point3ds[:3]-posCam).tolist()+[1])

            else:
                pointInOpts = np.asarray([self.camRightP.graficToOptical(np.array([pointIn[1], pointIn[0], 1]))
                                          for pointIn in pointsUV1])
                point3ds = np.asarray([self.camRightP.backproject(pointInOpt) for pointInOpt in pointInOpts])
        return np.concatenate((point3ds[:,:3]-posCam*np.ones(pointInOpts.shape), np.ones((pointInOpts.shape[0], 1))), 1)


    def getPointsfromline(self,vectors,Origin):
        lst = [[] for N in range(vectors.shape[0])]
        # alfas = np.linspace(10,20,M)
        alfas = [10,100,10000]
        for idx,Vxyz in enumerate(vectors):
            for alfa in alfas:
                xyz = Origin + Vxyz[:3]*alfa
                lst[idx].append(np.array([xyz[0],xyz[1],xyz[2],1]))

        return lst

    def getlstProjectedPoints(self,lst3d):
        projected_lst_grafic = [[] for N in range(len(lst3d))]
        for N in range(len(lst3d)):
            if N%100==0:
                print N,"/",len(lst3d)
            for idx,xyz in enumerate(lst3d[N]):
                projected = self.camRightP.project(xyz)
                projected_grafic = np.floor(self.camRightP.opticalToGrafic(projected)).astype(np.int)
                projected_lst_grafic[N].append(np.array([projected_grafic[1],projected_grafic[0],1]))

        return projected_lst_grafic


    def getLineEpipolar(self,lstPoints3d,cam):
        projected_lst_grafic = []
        m=0
        n=0
        for idx, xyz in enumerate(lstPoints3d):
            if cam == "right":
                projected = self.camRightP.project(xyz)
                projected_grafic = np.floor(self.camRightP.opticalToGrafic(projected)).astype(np.int)
            if cam == "left":
                projected = self.camLeftP.project(xyz)
                projected_grafic = np.floor(self.camLeftP.opticalToGrafic(projected)).astype(np.int)
            projected_lst_grafic.append(np.array([projected_grafic[1], projected_grafic[0], 1]))

        #Get m y n from y = mx+n
        begin = projected_lst_grafic[0][1]
        end = projected_lst_grafic[2][1]
        m = (projected_lst_grafic[0][0]-projected_lst_grafic[1][0]) / (projected_lst_grafic[0][1]-projected_lst_grafic[1][1])
        n = projected_lst_grafic[0][0]-m*projected_lst_grafic[0][1]
        return m,n,begin,end

    def getPointsfromEpipolar(self,m,n,begin,end,size=5):
        horizontal = np.arange(begin, end+1).reshape(-1, 1)
        inc = int(size/2)
        n_inc = np.arange(-inc + n, n + inc + 1)
        if m == 0:
            # Par estereo canonico
            vertical = np.asarray([np.repeat(n,horizontal.shape[0]) for n in n_inc],dtype=int).reshape(-1,1)
            horizontal_inc = np.repeat(horizontal,n_inc.size,axis=1).T.reshape(-1,1)

            return np.concatenate((vertical,horizontal_inc),axis=1)
        else:
            vertical = np.asarray([np.asarray([m*x+n for x in horizontal],dtype=int) for n in n_inc],dtype=int).reshape(-1,1)
            horizontal_inc = np.repeat(horizontal,n_inc.size,axis=1).T.reshape(-1,1)
            # Limit point coordinate
            idx = np.where(vertical<self.height)[0].tolist()
            if len(idx)>0:
                vertical_limit = vertical[idx,:]
                horizontal_limit = horizontal_inc[idx,:]
            else:
                vertical_limit = vertical
                horizontal_limit = horizontal_inc

            return np.concatenate((vertical_limit,horizontal_limit),axis=1)



    def getProjectedPoints(self,lstPoints3d,cam):
        projected_lst_grafic = []
        for idx, xyz in enumerate(lstPoints3d):
            if cam == "right":
                projected = self.camRightP.project(xyz)
                projected_grafic = np.floor(self.camRightP.opticalToGrafic(projected)).astype(np.int)
            if cam == "left":
                projected = self.camLeftP.project(xyz)
                projected_grafic = np.floor(self.camLeftP.opticalToGrafic(projected)).astype(np.int)
            projected_lst_grafic.append(np.array([projected_grafic[1], projected_grafic[0], 1]))
        return projected_lst_grafic






    def drawPoint(self,image,lstPoints,idx,color):
        out = image.copy()
        if not idx:
            if len(lstPoints.shape)==1:
                out[lstPoints[0], lstPoints[1]] = color
            else:
                out[lstPoints[:,0], lstPoints[:,1],:] = color
        else:
            out[lstPoints[idx,0],lstPoints[idx,1]]= color


        return out

    def drawLastPoint(self,image,lstPoints,idx,color=(0,255,255)):
        out = image.copy()
        if not idx:
            cv2.circle(out, (lstPoints[1], lstPoints[0]), 3,color, 2)

        else:
            cv2.circle(out, (lstPoints[idx, 1], lstPoints[idx, 0]), 3,color, 2)
        return out




    def intersection(self,vectorR,vectorL,OR,OL):

        # Grueso
        inc_grueso = np.asarray([np.repeat(value,3,0) for value in np.arange(0.0, 50.0, 0.01)])
        lineR_grueso = OR+vectorR*inc_grueso
        lineL_grueso = OL+vectorL*inc_grueso

        minimun = LA.norm(OR-OL)
        for idx in range(lineR_grueso.shape[0]):
            distance = LA.norm(lineR_grueso[idx,:]-lineL_grueso[idx,:])
            if distance <= minimun:
                minimun = distance

            else:
                break

        inc_fino = np.asarray([np.repeat(value, 3, 0) for value in np.arange(inc_grueso[idx-1,0], inc_grueso[idx,0], 0.001)])
        lineR_fino = OR + vectorR * inc_fino
        lineL_fino = OL + vectorL * inc_fino

        # FINO
        for idx in range(lineR_fino.shape[0]):
            distance = LA.norm(lineR_fino[idx, :] - lineL_fino[idx, :])
            if distance <= minimun:
                minimun = distance

            else:
                break


        return lineR_fino[idx-1,:]



    def plot3dvector(self,vectorR,vectorL,OR,OL,intersec):

        N=100
        inc = np.asarray([np.repeat(value,3,0) for value in np.arange(0.0, 50, 0.1)])

        Pinter0 = np.concatenate((intersec, np.array([0,0,0]))).reshape(-1, 3)
        OLR =  np.concatenate((OL, OR)).reshape(-1, 3)
        lineR = OR+vectorR*inc
        lineL = OL+vectorL*inc
        fig = plt.figure()

        ax = fig.gca(projection='3d')

        ax.plot(lineR[:,0],lineR[:,1],lineR[:,2],'b')
        ax.plot(lineL[:,0],lineL[:,1],lineL[:,2],'r')
        ax.plot(Pinter0[:,0], Pinter0[:,1], Pinter0[:,2], 'k')
        ax.plot(OLR[:,0], OLR[:,1], OLR[:,2], 'g')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        return


    def get3dColor(self,matchingR,keyPointsL,vectorL,ind_not_match,imageLeft):
        OR = self.camRightP.getCameraPosition()
        vectorR = MyAlgorithm.getVectors(self, matchingR, OR, "left")
        vectorL =  np.delete(vectorL,ind_not_match,0)
        # disparity = matchingR[:,1]-keyPointsL[:,1]
        OL = self.camLeftP.getCameraPosition()


        lstPLR_color = []
        for idx,vL in enumerate(vectorL):
            if idx%100==0:
                print idx,'/',vectorL.shape[0]

            intersec = MyAlgorithm.intersection(self,vectorR[idx, :3], vectorL[idx, :3], OR, OL)
            color = MyAlgorithm.getColor(self,imageLeft,keyPointsL[idx,:])
            lstPLR_color.append(intersec.tolist()+color)


        return lstPLR_color

    def triangulate(self,pointR,imageRight,vectorL,idx):
        vL = vectorL[idx,:3]
        vR = MyAlgorithm.getVectors(self,pointR,self.OR,"right")[:3]
        intersec = MyAlgorithm.intersection(self, vR, vL, self.OR, self.OL)
        color = MyAlgorithm.getColor(self, imageRight, pointR)
        return intersec.tolist()+color



    def getColor(self,image,point):
        point = point.astype(np.int)
        return image[point[0],point[1],:].tolist()

    def matchingPoint(self, keyPointL, lst2matchingR, edgesR,outR, outL, size=(7, 7)):

        alfa = 0.6
        beta = 0.4
        incu = int(size[0] / 2)
        incv = int(size[1] / 2)

        patchL = self.hsvL[keyPointL[0] - incu:keyPointL[0] + incu + 1, keyPointL[1] - incv:keyPointL[1] + incv + 1, :]
        maximum = 0.5
        best = np.zeros([3,])



        # eliminamos repetidos
        for jdx, uvR in enumerate(lst2matchingR):
            # Solo buscamos el matching si es un pixel de contorno
            if edgesR[uvR[0], uvR[1]] == 255:
                patchR = self.hsvR[uvR[0] - incu:uvR[0] + incu + 1, uvR[1] - incv:uvR[1] + incv + 1, :]
                corr = alfa * MyAlgorithm.correlation_coefficient(self, patchL[:, :, 0], patchR[:, :, 0]) \
                       + beta * MyAlgorithm.correlation_coefficient(self, patchR[:, :, 1], patchR[:, :, 1])
                if corr > maximum:
                    maximum = corr
                    best = uvR

        outL = MyAlgorithm.drawPoint(self, outL, keyPointL.astype(np.int), None, (0, 255, 0))
        outL_last = MyAlgorithm.drawLastPoint(self, outL, keyPointL.astype(np.int), None)


        if tuple(best) == tuple(np.zeros((3,))):
            print "Point not match: ", keyPointL
            return np.zeros((3,)),outR,outL
            # delete point

        pointR = best.astype(np.int)

        # Draw matching
        outR = MyAlgorithm.drawPoint(self, outR, pointR.astype(np.int), None, (255, 0, 0))
        outR_last = MyAlgorithm.drawLastPoint(self, outR, pointR.astype(np.int), None)
        self.setRightImageFiltered(outR_last)
        self.setLeftImageFiltered(outL_last)
        return pointR, outR, outL

    def execute(self):

        # OBTENCIÓN DE IMÁGENES
        imageLeft = self.sensor.getImageLeft()
        imageRight = self.sensor.getImageRight()
        self.width = imageRight.shape[1]
        self.height = imageRight.shape[0]
        self.time = time.clock()
        self.outEpipol = np.zeros((self.height,self.width*2,3),dtype=np.uint8)

        self.OR = self.camRightP.getCameraPosition()
        self.OL = self.camLeftP.getCameraPosition()
        self.hsvR = cv2.cvtColor(imageRight, cv2.COLOR_RGB2HSV)
        self.hsvL = cv2.cvtColor(imageLeft, cv2.COLOR_RGB2HSV)


        os.system("killall gzserver")

        save = False
        write = True
        visor = True


        # KEYPOINTS
        print "Calculating canny"
        keypointsR, keypointsL,edgesR,edgesL = MyAlgorithm.findKeypoints(self,imageR=imageRight,imageL=imageLeft)
        outR = cv2.cvtColor(edgesR, cv2.COLOR_GRAY2RGB)
        outL = cv2.cvtColor(edgesL, cv2.COLOR_GRAY2RGB)

        print "Keypoints Right: {:d}\t Keypoints Left: {:d}".format(keypointsR.shape[0],keypointsL.shape[0])
        print "Done!"

        print "Calculating all vectors from left camera"
        # Lines
        vectorL = MyAlgorithm.getVectors(self,keypointsL,self.camLeftP.getCameraPosition())


        # Get M points from N vectors: NxM
        print "Calculating list of 3D points "
        lstPoints3d_L = MyAlgorithm.getPointsfromline(self,vectorL,self.camLeftP.getCameraPosition())
        print "Dimension",len(lstPoints3d_L),"x",len(lstPoints3d_L[0])

        # Get NxM projected points from NxM

        for idx,Points3d_L in enumerate(lstPoints3d_L):
            if idx%25==0 and idx>0:
                print idx,'/',len(lstPoints3d_L),"T: ", time.clock() - self.time
                # self.outEpipol[:,:self.width,:]=self.outL_line
                # self.outEpipol[:,self.width:,:]=self.outR_line
                # cv2.imwrite("result/imagenes_epipolar/LR/imag_epipol" + str(idx) + ".png",  self.outEpipol)

                # cv2.imwrite("result/imagenes_epipolar/R/imagR_epipol"+str(idx)+".png",self.outR_line)
                # cv2.imwrite("result/imagenes_epipolar/L/imagL_epipol"+str(idx)+".png",self.outL_line)

            # print "K", keypointsL[idx,:]

            m, n, begin, end = MyAlgorithm.getLineEpipolar(self,Points3d_L,cam="right")
            projected_R = MyAlgorithm.getPointsfromEpipolar(self,m,n,begin,end)

            # Eliminamos los repetidos
            projected_R_unique =  np.array(list(set(tuple(p) for p in np.asarray(projected_R))))

            #pintamos los puntos proyectados
            outR_line = MyAlgorithm.drawPoint(self,outR,projected_R_unique,None,(255,255,0))
            self.setRightImageFiltered(outR_line)

            pointR,outR,outL = MyAlgorithm.matchingPoint(self,keypointsL[idx,:], projected_R_unique,edgesR,outR,outL)




            if tuple(pointR) != tuple(np.zeros((3,))):
                point3DColor = MyAlgorithm.triangulate(self,pointR,imageRight,vectorL,idx)

                if write:
                    MyAlgorithm.writeTxt(self,"result/ptsColor_franja5_2.txt",point3DColor,mode="unico")

                if visor:
                    self.sensor.drawPoint(np.array([point3DColor[0], point3DColor[1], point3DColor[2]]),
                                      (point3DColor[3], point3DColor[4], point3DColor[5]))

        print "Tiempo Total: ",time.clock()-self.time
        if self.done:
            return


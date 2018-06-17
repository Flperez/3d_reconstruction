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

    def writeTxt(self,path,points,color):
        '''
        Funcion para escribir en un txt los puntos con su color
            Escribe de uno en uno
        '''
        with open(path, 'a') as file:
            file.write("%f;%f;%f;%f;%f;%f\n" % (points[0], points[1], points[2],
                                                color[0], color[1], color[2]))



    def findKeypoints(self,imageR,imageL):
        '''
        Funcion que calcula los puntos de interes donde vamos a recrear la escena
        '''
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
        '''
        Calcula los vectores de las rectas en el espacio [XYZ1]
        '''
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
        '''
        Devuelve una lista de puntos XYZ de la recta definida por su vector y su origen
        y=mx+n
        '''
        lst = [[] for N in range(vectors.shape[0])]
        alfas = [10,100,10000]
        for idx,Vxyz in enumerate(vectors):
            for alfa in alfas:
                xyz = Origin + Vxyz[:3]*alfa
                lst[idx].append(np.array([xyz[0],xyz[1],xyz[2],1]))
        return lst



    def getLineEpipolar(self,lstPoints3d,cam):
        '''
        Recibe los puntos XYZ1 -> proyecta a coordenadas pixeles
        Calcula m y n (y=mx+n) y el punto inicial y final del eje horizontal
        '''
        projected_lst_grafic = []
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
        '''
        Funcion encargada de crear todos los puntos pertenecientes a la linea epipolar
        Recibe los parametros de la recta y los puntos donde debe iterar y el grosor de la recta
        '''
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





    def drawPoint(self,image,lstPoints,idx,color):
        '''
        Cambia el nivel de intensidad de una imagen por el color elegido
        Puedes elegir un indice en concreto si deseas
        '''
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
        '''
        Dibuja un circulo en la posicion
        '''
        out = image.copy()
        if not idx:
            cv2.circle(out, (lstPoints[1], lstPoints[0]), 3,color, 2)

        else:
            cv2.circle(out, (lstPoints[idx, 1], lstPoints[idx, 0]), 3,color, 2)
        return out






    def triangulate(self,pointR,imageRight,vectorL,idx):
        '''
        Triangula los puntos con sus rectas en 3d y obtiene los 2 puntos
        si la distancia entre estos supera un cierto umbral se utiliza
        '''
        vL = vectorL[idx,:3]
        vR = MyAlgorithm.getVectors(self,pointR,self.OR,"right")[:3]
        # intersec = MyAlgorithm.intersection(self, vR, vL, self.OR, self.OL)
        pa,pb = MyAlgorithm.intersectionEcuation(self, vR, vL, self.OR, self.OL)

        if pa.size != 0:
            if LA.norm(pa-pb)<5:
                color = MyAlgorithm.getColor(self, imageRight, pointR)
                return 0.5*(pa+pb),tuple(color)
            else:
                return np.array([]),None

        else:
            return np.array([]),None




    def getColor(self,image,point):
        '''
        Obtiene el color de la imagen de ese punto
        '''
        point = point.astype(np.int)
        color = image[point[0],point[1],:].astype(np.float32).tolist()
        return (color[0]/255,color[1]/255,color[2]/255)


    def calculateSAD(self,patch1,patch2):
        if patch1.shape[1] < patch2.shape[1]:
            patch2 = patch2[:,0:patch1.shape[1]]

        if patch2.shape[1] < patch1.shape[1]:
            patch1 = patch1[:, 0:patch2.shape[1]]

        dif = abs(patch1-patch2)
        sum = np.sum(dif)
        sum /= 255*dif.size
        return sum


    def matchingPoint(self, keyPointL, lst2matchingR, edgesR,outR, outL, size=(25, 25)):
        '''
        Realiza el maching de un punto de la imagen izqda en todos aquellos puntos perteneciente
        a la linea epiplar derecha
        '''

        alfa = 0.5
        beta = 0.5
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

                corr =1-( alfa * MyAlgorithm.calculateSAD(self, patchL[:, :, 0], patchR[:, :, 0])\
                       + beta * MyAlgorithm.calculateSAD(self, patchR[:, :, 1], patchR[:, :, 1]))


                if corr > maximum:
                    maximum = corr
                    best = uvR

        outL = MyAlgorithm.drawPoint(self, outL, keyPointL.astype(np.int), None, (0, 255, 0))
        outL_last = MyAlgorithm.drawLastPoint(self, outL, keyPointL.astype(np.int), None)


        if tuple(best) == tuple(np.zeros((3,))):
            print "Point not match: ", keyPointL
            return np.zeros((3,)),outR,outL

        pointR = best.astype(np.int)

        # Draw matching
        outR = MyAlgorithm.drawPoint(self, outR, pointR.astype(np.int), None, (255, 0, 0))
        outR_last = MyAlgorithm.drawLastPoint(self, outR, pointR.astype(np.int), None)
        self.setRightImageFiltered(outR_last)
        self.setLeftImageFiltered(outL_last)
        return pointR, outR, outL


    def intersectionEcuation(self,vectorR,vectorL,OR,OL):
        '''
        Calculo de la interseccion de dos rectas en 3D y calculo del punto con menor distancia entre ellos
        '''
        ORL = OR-OL
        d1343 = ORL[0]*vectorL[0]+ORL[1]*vectorL[1]+ORL[2]*vectorL[2]
        d4321 = vectorL[0]*vectorR[0]+vectorL[1]*vectorR[1]+vectorL[2]*vectorR[2]
        d1321 = ORL[0]*vectorR[0]+ORL[1]*vectorR[1]+ORL[2]*vectorR[2]
        d4343 = vectorL[0]*vectorL[0]+vectorL[1]*vectorL[1]+vectorL[2]*vectorL[2]
        d2121 = vectorR[0]*vectorR[0]+vectorR[1]*vectorR[1]+vectorR[2]*vectorR[2]
        denom = d2121 * d4343 - d4321 * d4321
        numer = d1343 * d4321 - d1321 * d4343


        if (abs(denom) < 1e-6):
            return np.array([]),np.array([])
        else:
            mua = numer / denom
            mub = (d1343 + d4321 * (mua)) / d4343

            pa = OR+vectorR*mua
            pb = OL+vectorL*mub

            if abs(pa[0]) > 1e6 or  abs(pa[1]) > 1e6 or  abs(pa[2]) > 1e6:
                return np.array([]), np.array([])
            else:
                return pa,pb


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


        write = False
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


        # Bucle
        for idx,Points3d_L in enumerate(lstPoints3d_L):
            if idx%100==0 and idx>0:
                print idx,'/',len(lstPoints3d_L),"T: ", time.clock() - self.time

            # Linea epipolar
            m, n, begin, end = MyAlgorithm.getLineEpipolar(self,Points3d_L,cam="right")
            projected_R = MyAlgorithm.getPointsfromEpipolar(self,m,n,begin,end)

            # Eliminamos los repetidos
            projected_R_unique =  np.array(list(set(tuple(p) for p in np.asarray(projected_R))))

            #pintamos los puntos proyectados
            outR_line = MyAlgorithm.drawPoint(self,outR,projected_R_unique,None,(255,255,0))
            self.setRightImageFiltered(outR_line)


            # Matching
            pointR,outR,outL = MyAlgorithm.matchingPoint(self,keypointsL[idx,:],
                                                         projected_R_unique,edgesR,outR,outL, size=(25,25))


            if tuple(pointR) != tuple(np.zeros((3,))):

                # Triangulacion
                point3D, color = MyAlgorithm.triangulate(self,pointR,imageRight,vectorL,idx)

                # Pintamos
                if point3D.size !=0:
                    if write:
                         MyAlgorithm.writeTxt(self,"pts3DColor",point3D,color,)

                    if visor:
                        self.sensor.drawPoint(point=point3D,color=color)

        print "Tiempo Total: ",time.clock()-self.time
        if self.done:
            return


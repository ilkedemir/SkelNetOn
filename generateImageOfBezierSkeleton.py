#!/usr/bin/env python3
import os, sys, subprocess
import csv, math
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import operator as op

''' ============ Script to generate images of Bezier Skeletons ====================

dependencies : matplotlib library
pip3 install matplotlib

author @amelie (amelie.fondevilla@inria.fr)
'''
bzdg = 5; # Degree of the Bezier curves
default_nsamples = 30 # Default number of samples per bezier
dim = 3; # Dimension of the points of the Bezier curves (x,y,r)
imageDim = 255 # Size of the original image
# --- Frame representing the image
axisRange = [0,imageDim,0,imageDim]
frame_x,frame_y = [0,imageDim,imageDim,0,0],[0,0,imageDim,imageDim,0]
colorFrame = '0.75'

# Input/Output directories
input_bzskel_directory = "augmented/bzskl/"
output_bzimg_directory = "augmented/bzskl_img/"


input_bzskel_directory = "../out/"
output_bzimg_directory = "../out/img/"

# input_bzskel_directory = "CVPR_data/bzskl/"
# output_bzimg_directory = "CVPR_data/bzskl_img/"

# Bezier skeleton file : empty if run the whole directory
# input_modelname =  ["bat-6_rot148.csv"]
input_modelname = []

if not os.path.isdir(output_bzimg_directory):
    os.mkdir(output_bzimg_directory)

''' Just a plotting function '''
def plotCircle(cx,cy,r,nsamples = 30, colorCircle = ""):
    sx = np.array(nsamples*[0], float);
    sy = np.array(nsamples*[0], float);

    for i in range(nsamples):
        t = 2*math.pi*i/(nsamples-1);
        sx[i] = r*np.cos(t)+cx;
        sy[i] = r*np.sin(t)+cy;

    if len(colorCircle) == 0:
        plt.plot(sx,sy)
    else:
        plt.plot(sx,sy,color=colorCircle)

''' Bernstein basis evaluation (naive)'''
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def Bernstein(n,t):
    Bnst = [0]*(n+1);
    for i in range(n+1):
        Bnst[i] = ncr(n,i)*pow(t,i)*pow(1-t,n-i);
    return Bnst

def generateBernsteinCoefficients(n,tvec):
    bern_mem = {};
    for t in tvec:
        bern_mem[t] = Bernstein(n,t)
    return bern_mem;

# Memoization acceleration
Bernstein_mem = generateBernsteinCoefficients(bzdg,np.linspace(0,1,default_nsamples))

def getBernsteinCoef(t):
    global bzdg
    global Bernstein_mem

    try:
        coef = Bernstein_mem[t]
    except KeyError as e:
        coef = Bernstein(bzdg,t)
        Bernstein_mem[t] = coef

    return coef

def evalBernstein(polygon,t):
    global bzdg
    dim = len(polygon[0])

    Bnst = getBernsteinCoef(t)
    p = np.array(dim*[0],float)
    for i in range(bzdg+1):
        p = p + Bnst[i]*polygon[i];

    return p;

''' Medial axis data structure '''
class PointR:
    def __init__(self,pdata = [0,0,0]):
        self.x = pdata[0]
        self.y = pdata[1]
        self.r = pdata[2]

    def plotPoint(self, colorPt = "k"):
        plt.plot(self.x,(imageDim)-self.y,color=colorPt,marker=".");

    def plotCircle(self, colorCircle = "g"):
        plotCircle(self.x,(imageDim)-self.y,self.r,colorCircle=colorCircle)

    def toArray(self):
        return np.array([self.x,self.y,self.r])

    def __str__(self):
        return "[" + str(self.x) + "," + str(self.y) + "," + str(self.r) + "]"

    def __repr__(self):
        return str(self)

class Branch:
    def __init__(self,vdata,ind=-1):
        self.id = ind;
        global dim;
        self.np = int(len(vdata)/dim)
        self.cpoints = [PointR(vdata[dim*i:dim*(i+1)]) for i in range(self.np)];

    def getUniformSampling(self,nsamples=default_nsamples):
        t = np.linspace(0,1,nsamples)
        global dim
        usmplg = np.array(nsamples*[PointR()])
        for i in range(nsamples):
            usmplg[i] = PointR(evalBernstein(self.toArray(),t[i]))
        return usmplg

    def plotCircles(self, colorCircles = "g"):
        for smp in self.getUniformSampling():
            smp.plotCircle(colorCircle = colorCircles)

    def plotPoints(self, colorPoints = "k"):
        sampling = self.getUniformSampling()
        smp_x,smp_y = np.array([p.x for p in sampling]), np.array([p.y for p in sampling])
        plt.plot(smp_x,(imageDim)-smp_y,colorPoints,linestyle="-",marker=".")

    def toArray(self):
        global dim
        arr = np.array(self.np*[dim*[0]], float)
        for p in range(self.np):
            arr[p] = self.cpoints[p].toArray()
        return arr


class Shape:
    def __init__(self,vdata,shname = ""):
        global bzdg;
        global dim;
        N = (bzdg+1)*dim;
        self.name = shname;
        self.np = int(len(vdata)/dim);
        self.nb = int(self.np/(bzdg+1));
        self.branches = [Branch(vdata[N*i:N*(i+1)],i) for i in range(self.nb)];

    def plot(self,outfolder):
        plt.title("Shape {}".format(self.name));

        for bch in self.branches:
            bch.plotCircles(colorCircles = "g")

        for bch in self.branches:
            bch.plotPoints(colorPoints = "k")

        plt.axis("equal")
        plt.axis(axisRange)
        plt.plot(frame_x,frame_y,color=colorFrame)

''' Parsing methods '''
def basename_noext(filename):
	bname = os.path.basename(filename)
	bname = os.path.splitext(bname)[0]
	return bname

def parse_shape(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',');
        shape_dat = []
        for row in csv_reader:
            shape_dat += [float(r) for r in row]
        return Shape(shape_dat,basename_noext(filename));

''' Generating the images '''
def generate_image(shape, outfolder):
    plt.clf()
    shape.plot(outfolder)

    out_fname = outfolder + shape.name + ".png"
    plt.savefig(out_fname)
    return out_fname

''' ==================== MAIN ============================== '''
def generateImagesOfBezierSkeleton(bzskel_file, output_dir):
    shape = parse_shape(bzskel_file)
    return generate_image(shape,output_dir);

def getFilenames(fname):
    if os.path.isfile(fname):
        return [fname]

    if os.path.isdir(fname):
        lfiles = [];
        for f in os.listdir(fname):
            fulldir = fname + f;
            if os.path.isfile(fulldir):
                lfiles.append(fulldir)
        return lfiles;

if len(input_modelname) == 0:
    input_fnames = getFilenames(input_bzskel_directory)
else:
    input_fnames = [input_bzskel_directory + m for m in input_modelname]

input_fnames = [f for f in input_fnames if f[-4:] == ".csv"]

nfiles = len(input_fnames)
count = 0
for bzskel_file in input_fnames:
    count += 1
    print("Generating image #{} out of {} : {}".format(count,nfiles,bzskel_file))
    out_fname = generateImagesOfBezierSkeleton(bzskel_file,output_bzimg_directory)
    print("Done, image written in {}".format(out_fname))
    print("")

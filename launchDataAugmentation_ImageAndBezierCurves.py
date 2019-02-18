#!/usr/bin/env python3
import math, cv2
import sys, os, csv
import numpy as np

''' ============ Script for data Augmentation (Bezier Skeletons) ====================
Creates rotated, scaled and translated versions of a set of imagesself.
Stores all affine transformations in a CSV file.
Applies transformations to a set of bezier skeletons to create augmented bezier skeletons

dependencies : python OpenCV library
pip3 install opencv-python

author @amelie (amelie.fondevilla@inria.fr)
'''

im_dimension = 256,256 # Size of all images in the dataset
border_width = 5 # Minimal margin around the shape in the images
Nrotations = 5 # Number of rotations generated
Nscalings = 3 # Number max of scalings generated
step_translation = 50 # Step for translations in pixel
shape_dimension_limit = 50 # Minimal dimension of a shape in pixel (for scaled images)

# Input/Output directories
input_images_directory = "CVPR_data/images/"
input_bzskel_directory = "CVPR_data/bzskl/"
output_images_directory = "augmented/images/"
output_transf_directory = "augmented/transform/"
output_bzskel_directory = "augmented/bzskl/"

modelnames_file = "CVPR_data/trainset.txt"

if not os.path.isdir(output_images_directory):
    os.mkdir(output_images_directory)
if not os.path.isdir(output_transf_directory):
    os.mkdir(output_transf_directory)
if not os.path.isdir(output_bzskel_directory):
    os.mkdir(output_bzskel_directory)

def basename_noext(filename):
	bname = os.path.basename(filename)
	bname = os.path.splitext(bname)[0]
	return bname

''' ======================= Generating augmented images =================== '''

''' -- Shape extraction '''
def getShapePixelsInImage(im):
    return [(i,j) for (i,j),el in np.ndenumerate(im) if (el > 0)]

def getShapeBBox(shape_px):
    all_i,all_j = [i for (i,j) in shape_px],[j for (i,j) in shape_px]
    minP =  np.array([min(all_i), min(all_j)])
    maxP = np.array([max(all_i), max(all_j)])
    return minP,maxP;

def getShapeDimensions(shape_px):
    minP,maxP = getShapeBBox(shape_px)
    dim = maxP - minP
    height,width = dim[0],dim[1]
    return width,height

def imageFromShapePixels(shape_px,dim):
    w,h = dim
    img = np.zeros((h,w), np.uint8);
    for (i,j) in shape_px:
        if (i < 0) or (i >= w) or (j < 0) or (j >= h):
            pass
        else:
            img[i,j] = 255
    return img;

''' -- Transformations '''
''' Resizing '''
def optRatio(dim,target):
    h,w = dim
    th,tw = target
    rh = th/h
    rw = tw/w
    return min(rh,rw)

def resizeImage(im,target):
    r = optRatio(im.shape,target)
    optshape = int(r*im.shape[1]),int(r*im.shape[0])
    resized = cv2.resize(im, optshape , interpolation = cv2.INTER_AREA)

    rh,rw = resized.shape
    th,tw = target
    final = np.zeros((th,tw), np.uint8)
    final[0:rh, 0:rw] = resized

    return final,r

def scaleAndCrop(im,scaling_ratio):
    h,w = im.shape
    rw,rh = int(scaling_ratio*w),int(scaling_ratio*h)
    im_scaled = cv2.resize(im,(rw,rh), interpolation = cv2.INTER_AREA)

    im_scaled_and_cropped = np.zeros((h,w), np.uint8)
    mw,mh = min(w,rw),min(h,rh)
    im_scaled_and_cropped[0:mh, 0:mw] = im_scaled[0:mh,0:mw]

    return im_scaled_and_cropped;

def fitInLargerImage(im,nshape):
    nh,nw = nshape
    h,w = im.shape
    nh = max(nh,h)
    nw = max(nw,w)

    im_large = np.zeros((nh,nw),np.uint8)
    im_large[0:h,0:w] = im

    return im_large


''' Centering '''
def centerShapeInImage(im,offset):
    shape_px = getShapePixelsInImage(im);
    minP,maxP = getShapeBBox(shape_px)
    nsize = int(np.ceil(np.linalg.norm(maxP-minP))) # BBox diagonal

    global border_width
    im_large = fitInLargerImage(im,(nsize+border_width,nsize+border_width))

    medP = 0.5*(minP+maxP)
    nh,nw = im_large.shape
    t = np.array([int(nh/2),int(nw/2)]) - medP
    it = np.array([int(t[0]),int(t[1])])
    return translateShapeBy(im_large,it),it


def moveShapeToCornerInImage(im,offset):
    h,w = im.shape
    shape_px = getShapePixelsInImage(im);
    minP,maxP = getShapeBBox(shape_px) # min/maxP[0] is rows indices -> y

    it = np.array([int(offset-minP[0]),int(offset-minP[1])])

    return translateShapeBy(im,it),it

def translateShapeBy(im,t):
    h,w = im.shape
    shape_px = getShapePixelsInImage(im);
    translated_im = [(x + t ) for x in shape_px ]
    return imageFromShapePixels(translated_im, (w,h));

''' Rotating '''
def rotateShapeCV(im,angle_degree):
    rows,cols = im.shape;
    imcenter = np.array([cols/2,rows/2])
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle_degree,1)
    dst = cv2.warpAffine(im,M,(cols,rows))
    return dst,imcenter

''' Affine Transformation Matrices '''
def translationMatrix(t):
    T = np.identity(3)
    T[0:2,2] = t
    return T

def rotationMatrix(angle_degree,center):
    alpha = angle_degree*math.pi/180
    T_mc = translationMatrix(-center)
    T_pc = translationMatrix(center)
    cs = np.cos(alpha)
    sn = np.sin(alpha)
    R = np.array([[cs,-sn,0],[sn,cs,0],[0,0,1]])
    return T_pc @ R @ T_mc

def scalingMatrix(ratio):
    S = np.identity(3)
    S[0:2,0:2] = ratio * S[0:2,0:2]
    return S

# image space -> vector space : inversion between x and y
def imageSpaceToVectorSpace(t):
    return  np.array([t[1],t[0]],)

def getAffineTransformation(translation_vec=np.array([0,0]),rotation_angle=0,rotation_center=np.array([0,0]),scaling_ratio=1):
    T = translationMatrix(imageSpaceToVectorSpace(translation_vec))
    R = rotationMatrix(rotation_angle,imageSpaceToVectorSpace(rotation_center))
    S = scalingMatrix(scaling_ratio)
    return S @ R @ T

''' -- Data Augmentation '''

def writeCSV_transformations(fname,dict):
    with open(fname, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['File','Transformation'])
        for key, value in dict.items():
            flat_val = [x for row in value for x in row]
            all_dat = [key] + flat_val
            writer.writerow(all_dat)

def generateRotatedImages(im,angles,modelname,output_dir):
    global border_width
    global im_dimension
    dict_transformations = {}

    im_centered,origin_t = centerShapeInImage(im,border_width);
    T = getAffineTransformation(translation_vec=origin_t)

    for rotation_angle in angles:
        im_rotated,rotation_center = rotateShapeCV(im_centered,rotation_angle);
        im_rotated_resized,scaling_ratio = resizeImage(im_rotated,im_dimension)

        out_path = "{dir}{shape}_rot{angle}.png".format(dir=output_dir,shape=modelname,angle=rotation_angle)
        cv2.imwrite(out_path,im_rotated_resized);
        A = getAffineTransformation(rotation_angle=rotation_angle, rotation_center=rotation_center,scaling_ratio=scaling_ratio)

        dict_transformations[out_path]= A @ T

    return dict_transformations

def generateTranslatedImages(im,step_px,modelname,output_dir):
    global border_width
    im_origin,origin_t = moveShapeToCornerInImage(im,border_width)
    h,w = im_origin.shape
    T = getAffineTransformation(translation_vec=origin_t)
    dict_transformations = {}

    minP,maxP = getShapeBBox(getShapePixelsInImage(im_origin))
    tx,ty = 0,0
    while ((tx + maxP[0]) < h-border_width):
        ty = 0
        while ((ty + maxP[1]) < w-border_width):
            tv = np.array([tx,ty])
            im_translated = translateShapeBy(im_origin,tv)

            out_path = "{dir}{shape}_transl_{tx}_{ty}.png".format(dir=output_dir,shape=modelname,tx=tv[0],ty=tv[1])
            cv2.imwrite(out_path,im_translated);
            A = getAffineTransformation(tv)
            dict_transformations[out_path]= A @ T
            ty = ty +step_px
        tx = tx+step_px
    return dict_transformations

def generateScaledImages(im,ratios,modelname,output_dir):
    global border_width
    global im_dimension
    im_origin,origin_t = moveShapeToCornerInImage(im,border_width)
    T = getAffineTransformation(translation_vec=origin_t)
    dict_transformations = {}

    width,height = getShapeDimensions(getShapePixelsInImage(im_origin))
    scaling_ratio = 1;
    for scaling_ratio in ratios:
        outofwidth = (width*scaling_ratio >= im_dimension[1])
        outofheight = (height*scaling_ratio >= im_dimension[0])
        if ( outofwidth or outofheight ):
            continue;

        global shape_dimension_limit
        toothin = (width*scaling_ratio < shape_dimension_limit)
        toosmall = (height*scaling_ratio < shape_dimension_limit)
        if ( toothin or toosmall ):
            continue;

        im_scaled = scaleAndCrop(im_origin,scaling_ratio)

        out_path = "{dir}{shape}_scaled_{r}.png".format(dir=output_dir,shape=modelname,r=scaling_ratio)
        cv2.imwrite(out_path,im_scaled)

        A = getAffineTransformation(scaling_ratio=scaling_ratio) @ T
        dict_transformations[out_path]= A

    return dict_transformations

def generateAugmentedImages(fname, output_img_dir, output_transf_dir):
    try:
        im = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    except :
        print("Could not open file {}".format(fname))
        return
    global Nrotations

    print("Generating augmented images.")

    modelname = os.path.basename(fname)[:-4]
    dict_transformations = {}

    angles = np.random.random_integers(0,360,Nrotations)
    dict_transformations.update(generateRotatedImages(im,angles,modelname,output_img_dir))

    global step_translation
    dict_transformations.update(generateTranslatedImages(im,step_translation,modelname,output_img_dir));

    sc_ratios = 2*np.random.random_sample(Nscalings)
    sc_ratios = [np.round(r*10)/10 for r in sc_ratios]
    sc_ratios = list(set(sc_ratios))
    dict_transformations.update(generateScaledImages(im,sc_ratios,modelname,output_img_dir));

    csv_fname = "{dir}{shape}_augmented.csv".format(dir=output_transf_dir,shape=modelname)
    writeCSV_transformations(csv_fname,dict_transformations)
    return csv_fname


''' ============= Applying transformations to Bezier Skeletons ===================== '''
def homogeneousCoord(bzskel):
    return [np.array([p[0],p[1],1]) for p in bzskel]

def getCoordFromHomogeneous(hcoord):
    return (hcoord[0]/hcoord[2],hcoord[1]/hcoord[2])

def homogeneousScalar(bzskel):
    return [np.array([p[2],0,0]) for p in bzskel]

def getScalarFromHomogeneous(hscal):
    return np.sqrt(hscal[0]*hscal[0] + hscal[1]*hscal[1])

def transformedBezierSkeleton(bzskel,T):
    ht_xy = [ T @ p for p in homogeneousCoord(bzskel)];
    ht_radius = [ T @ r for r in homogeneousScalar(bzskel)]
    sign_r = [np.sign(p[2]) for p in bzskel]

    transformed_xy = [ getCoordFromHomogeneous(t) for t in ht_xy]
    transformed_radius = [ getScalarFromHomogeneous(r) for r in ht_radius]
    transformed_radius = [s*r for s,r in zip(sign_r,transformed_radius)]

    return [ (txy[0],txy[1],r) for txy,r in zip(transformed_xy,transformed_radius)]

''' -- PARSING FUNCTIONS '''
''' Affine Transformations  '''
def parseTransformations(csv_fname):
    with open(csv_fname) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',');
        line_count = 0;
        transformations = {}
        for row in csv_reader:
            if line_count == 0:
                line_count = 1
                continue
            transf_data = [float(r) for r in row[1:]];
            T = np.array([transf_data[0:3],transf_data[3:6],transf_data[6:9]])
            transformations[row[0]] = T
            line_count +=1;
        return transformations

''' Bezier Skeletons  '''
def parseBzskel(csv_fname):
    with open(csv_fname) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',');
        line_count = 0;
        bzskel = []
        for row in csv_reader:
            frow = [float(r) for r in row]
            bzskel.append((frow[0],frow[1],frow[2]))
            line_count +=1
        return bzskel


''' -- GENERATING TRANSFORMED DATA '''
def generateTransformedBzSkeleton(transformations_file,initial_skeleton_file):
    transformations = parseTransformations(transformations_file)
    bzskel = parseBzskel(initial_skeleton_file)
    transformed_data = {}
    for file,T in transformations.items():
        tbzskel = transformedBezierSkeleton(bzskel,T)
        transformed_data[file] = tbzskel

    return transformed_data

def writeCSV_bzskel(transformed_data,transformed_data_dir):
    output_fnames = []
    for fname,value in transformed_data.items():
        bz_file = transformed_data_dir + basename_noext(fname) + ".csv"
        with open(bz_file,'w') as csv_file:
            writer = csv.writer(csv_file)
            for (x,y,r) in value:
                writer.writerow([x,y,r])
        output_fnames.append(bz_file)
    return output_fnames

def applyAffineTransformations(input_transf_file,input_bzskel_file,output_bzskel_dir):
    print("Applying Affine Transformations");
    transformed_data = generateTransformedBzSkeleton(input_transf_file,input_bzskel_file)
    return writeCSV_bzskel(transformed_data, output_bzskel_dir)

''' ============= Launch data augmentation ===================== '''

def getModelnames(filename):
    with open(filename) as file:
        flines = [line.strip() for line in file]
        return [basename_noext(name) for name in flines]

modelnames = getModelnames(modelnames_file)
count = 0
for model in modelnames:
    count += 1
    print("Shape {} : {} over {}".format(model,count,len(modelnames)))
    input_image_file = input_images_directory + model + ".png"
    input_bzskel_file = input_bzskel_directory + "skelpoints_" + model + ".csv"

    if not os.path.exists(input_image_file):
        print("Could not find file {}".format(input_image_file))
        continue

    if not os.path.exists(input_bzskel_file):
        print("Could not find file {}".format(input_bzskel_file))
        continue

    transformations_fname = generateAugmentedImages(input_image_file,output_images_directory,output_transf_directory)
    transformedbz_fnames= applyAffineTransformations(transformations_fname,input_bzskel_file,output_bzskel_directory)
    print("")

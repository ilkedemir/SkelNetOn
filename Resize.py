from PIL import Image, ImageOps
import io 
import glob
from matplotlib import pyplot
import numpy as np
import cv2
import scipy
import pdb
import os
from pdb import set_trace
from math import floor, ceil

SPLIT_CHAR = '\\' #For Windows
#SPLIT_CHAR = '/' #For Linux

desired_size_rect_0 = 1024-2 #this must be always the smaller size of the rectangle (compare to the number below)
desired_size_rect_1 = 2048-2

indir = 'Input/'   #location of  images given as png files, shape as coords_name.txt and skeletons as skeledges and skelpoints
outdirCropped = 'Output/images_cropped/'  #'images_cropped

outdirResized_256 = 'Output/images_resized/' # output directory

# Three Subdirectory in resized folders
labels = '/labels/' # Contains the information for the skeleton of the images
shapes = '/shapes/' # Contains the information for the shapes of the images


# Create the folders where the files are written
if not os.path.exists(outdirCropped):
    os.makedirs(outdirCropped)
if not os.path.exists(outdirCropped + shapes):
    os.makedirs(outdirCropped + shapes)
if not os.path.exists(outdirCropped + labels):
    os.makedirs(outdirCropped + labels)
if not os.path.exists(outdirResized_256):
    os.makedirs(outdirResized_256)
if not os.path.exists(outdirResized_256 + labels):
    os.makedirs(outdirResized_256 + labels)
if not os.path.exists(outdirResized_256 + shapes):
    os.makedirs(outdirResized_256 + shapes)


    
# Crops the images, skeletons and shapes
def crop_shape():
    for filename in glob.glob(indir + shapes + '/*.png'):
        name = filename.split('\\')[-1].split('.')[0]
        print(name)
        name_for_edges = indir + labels + "skeledges_" + filename.split( SPLIT_CHAR )[-1].split('.')[0] + '.txt'
        name_for_points = indir + labels + "skelpoints_" + filename.split( SPLIT_CHAR )[-1].split('.')[0] + '.txt'
        name_for_shape = indir + shapes + "coords_" + filename.split( SPLIT_CHAR )[-1].split('.')[0] + '.txt'
        mat_points = read_number_file( name_for_points )
        mat_edges = read_number_file( name_for_edges )
        mat_shape = read_number_file( name_for_shape )
        
        min_x = mat_shape[0]
        max_x = mat_shape[0]
        min_y = mat_shape[1]
        max_y = mat_shape[1]
        
        for i in range(int(len(mat_shape) / 2)):
            if(mat_shape[2 * i + 0] < min_x):
                min_x = mat_shape[2 * i + 0] 
            if(mat_shape[2 * i + 0] > max_x):
                max_x = mat_shape[2 * i + 0] 
            if(mat_shape[2 * i + 1] < min_y):
                min_y = mat_shape[2 * i + 1]
            if(mat_shape[2 * i + 1] > max_y):
                max_y = mat_shape[2 * i + 1] 
        height = int(round(max_y - min_y))
        width = int(round(max_x - min_x))   
        
        for i in range(int(len(mat_points)/2)):
            mat_points[i * 2] = mat_points[i * 2] - min_x
            mat_points[i * 2 + 1] = mat_points[i * 2 + 1] - min_y
        for i in range(int(len(mat_shape)/2)):
            mat_shape[i * 2] = mat_shape[i * 2] - min_x
            mat_shape[i *2 + 1] = mat_shape[i * 2 + 1] - min_y

        write_shape(mat_shape, height, width, outdirCropped + shapes + name)
        
        file_points = open( outdirCropped + labels + "skelpoints_" + name + '.txt', "w" )
        file_edges = open( outdirCropped + labels + "skeledges_" + name + '.txt', "w" )  
        file_shape = open( outdirCropped + shapes + "coords_" + name + '.txt', "w" ) 
        for i in range( int( len(mat_points)/2 )):
            file_points.write(str("{0:4.8f}".format(mat_points[i*2]))+"\t"+str("{0:4.8f}".format(mat_points[i*2+1]))+"\n")
        file_points.close()
        for i in range(int(len(mat_edges)/2)):
            file_edges.write(str(int(mat_edges[i*2]))+"\t"+str(int(mat_edges[i*2+1]))+"\n")
        file_edges.close()
        for i in range(int(len(mat_shape)/2)):
            file_shape.write(str("{0:4.8f}".format(mat_shape[i*2]))+"\t"+str("{0:4.8f}".format(mat_shape[i*2+1]))+"\n")
        file_shape.close()

    print('all cropping done!')

    


# Resize the skeleton
def resize_skeleton(top, left,  ratio, mat_points):
    
    mat_points_updated = list(mat_points)

    for i in range(int(len(mat_points))):
        mat_points_updated[i] = mat_points[i] * ratio
    
    shiftCol = left + 1
    shiftRow = top + 1

    for i in range(int(len(mat_points_updated) / 2)):
        mat_points_updated[i * 2] += shiftRow 

    for i in range(int(len(mat_points_updated) / 2)):
        mat_points_updated[i * 2 + 1] += shiftCol
        
    for i in range(int(len(mat_points_updated)/2)):
        tmp = mat_points_updated[i * 2] 
        mat_points_updated[i * 2] = mat_points_updated[i * 2 + 1] + 1
        mat_points_updated[i * 2 + 1] = tmp + 1
   
    return(mat_points_updated)




# Resize the shape
def resize_shape( top, left, ratio, mat_shape):
    
    scaled_shape = list(mat_shape)
    for i in range(int(len(mat_shape))):
        scaled_shape[i] =  ratio * mat_shape[i]
        #scaled_shape[i] =  ratio * int(round(mat_shape[i]))
    shiftCol = left + 1
    shiftRow = top + 1   
    

    for i in range(int(len(mat_shape) / 2)):
            scaled_shape[i * 2 + 1] += shiftCol
    for i in range(int(len(mat_shape) / 2)):
            scaled_shape[i * 2 + 0] += shiftRow

            
    return(scaled_shape)




# Always give smaller side first as input's desired_size the output size will be two pixels bigger in each dimension
# Resizes an image with artifacts to the desired size
def resize_image_rectangle(current_image, desired_size):
    
    current_image_size = current_image.shape[0:2]
    switch = 0
    if(current_image_size[0] < current_image_size[1]):
        ratio_0 = float(desired_size[0])/current_image_size[0]
        ratio_1 = float(desired_size[1])/current_image_size[1]
    else:
        ratio_0 = float(desired_size[0])/current_image_size[1]
        ratio_1 = float(desired_size[1])/current_image_size[0]
        switch = 1
    ratio = ratio_1
    if(ratio_0 < ratio_1):
        ratio = ratio_0
        
    new_size = [0, 0]    
    new_size[0] = int(current_image_size[0] * ratio)
    new_size[1] = int(current_image_size[1] * ratio)
    
    im = cv2.resize(current_image, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST )
    delta_w = desired_size[(1+switch)%2] - new_size[1]
    delta_h = desired_size[(switch)%2] - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(im, top+1, bottom+1, left+1, right+1, cv2.BORDER_CONSTANT, value=color)
    
    return (top, left, ratio, new_image)




# Always give smaller side first as input's desired_size 
# Finds the ratio to resize the current shape to the desired size
def find_ratio_rectangle(current_shape, desired_size):

    min_x = current_shape[0]
    max_x = current_shape[0]
    min_y = current_shape[1]
    max_y = current_shape[1]
    for i in range(int(len(current_shape) / 2)):
        if(current_shape[2 * i + 0] < min_x):
            min_x = current_shape[2 * i + 0] 
        if(current_shape[2 * i + 0] > max_x):
            max_x = current_shape[2 * i + 0] 
        if(current_shape[2 * i + 1] < min_y):
            min_y = current_shape[2 * i + 1]
        if(current_shape[2 * i + 1] > max_y):
            max_y = current_shape[2 * i + 1] 
    size_x = max_x - min_x;
    size_y = max_y - min_y;
    current_image_size = (size_x, size_y)
    
    switch = 0
    if(current_image_size[0] < current_image_size[1]):
        ratio_0 = float(desired_size[0] - 1) / current_image_size[0]
        ratio_1 = float(desired_size[1] - 1) / current_image_size[1]
    else:
        ratio_0 = float(desired_size[0] - 1) / current_image_size[1]
        ratio_1 = float(desired_size[1] - 1) / current_image_size[0]
        switch = 1
    ratio = ratio_1
    if(ratio_0 < ratio_1):
        ratio = ratio_0
    ratio = ratio = floor(ratio * 100000000000)/100000000000
    
    delta_w = desired_size[(1 + switch) % 2] - (current_image_size[1] * ratio)
    delta_h = desired_size[(switch) % 2] - (current_image_size[0] * ratio)
    top = delta_h//2
    left = delta_w//2

    return (top, left, ratio)

 
    
    
# Reading the shape of an image
def read_number_file(filename1):
    
    with open(filename1) as file_with_numbers:
        content = file_with_numbers.readlines()
    content = [x.strip() for x in content]
    mat = []
    for line_with_two_numbers in content:
        number_no = line_with_two_numbers.split()
        if(len(number_no) == 2):
            mat.append(float(number_no[0]))
            mat.append(float(number_no[1]))
        
    return mat




# Writing an image skeleton
def write_skeleton(mat_points, mat_edges, height, width, name):
 
    blank_image = np.zeros((height,width,3), np.uint8) 
    for i in range(int(len(mat_edges) / 2)):
        start_edge = int(mat_edges[i * 2 + 0]) - 1
        end_edge = int(mat_edges[i * 2 + 1]) - 1
        start_point_x = int(round(mat_points[2 * start_edge + 0])) - 1
        start_point_y = int(round(mat_points[2 * start_edge + 1])) - 1
        end_point_x = int(round(mat_points[2 * end_edge + 0])) - 1
        end_point_y = int(round(mat_points[2 * end_edge + 1])) - 1
        cv2.line(blank_image,(start_point_y,start_point_x),(end_point_y,end_point_x),(255,255,255),1)

    # cv2.imwrite(name, blank_image)
    pil_im = Image.fromarray(blank_image)
    pil_im.save(name + '.png',"PNG")
    

    
# Writing a shape image
def write_shape(mat_shape, height, width, name):
    
    mat_shape.append( mat_shape[0])
    mat_shape.append( mat_shape[1])
    blank_image = np.zeros((height,width,3), np.uint8)
    array_shape = np.array(mat_shape).reshape(int(len(mat_shape) / 2), 2)
    cv2.fillPoly(blank_image, np.array([array_shape], dtype=np.int32), (255,255,255))
    
    # cv2.imwrite(name, blank_image)
    pil_im = Image.fromarray(blank_image)
    pil_im.save(name + '.png',"PNG")
    
print('What is the one dimension to resize?')
dimension_0 = int(input()) - 2
print('What is the other dimension to resize?')
dimension_1 = int(input()) - 2
if(dimension_1 < dimension_0):
    min_dimension = dimension_1
    dimension_1 = dimension_0
    dimension_0 = min_dimension

if( dimension_0 < 0 or dimension_1 < 0 or dimension_0 > 2048 or dimension_1 > 2048):
    dimension_0 = 254
    dimension_1 = 254
        
# Crop then resize all images in cropped directory
crop_shape()


for filename in glob.glob( outdirCropped + shapes + '/*.png' ):

        
    #Read image
    name = filename.split( SPLIT_CHAR )[-1].split('.')[0] 
    print( name )
    name_for_edges = outdirCropped + labels + "skeledges_" + filename.split( SPLIT_CHAR )[-1].split('.')[0] + '.txt'
    name_for_points = outdirCropped + labels + "skelpoints_" + filename.split( SPLIT_CHAR )[-1].split('.')[0] + '.txt'
    name_for_shape = outdirCropped + shapes + "coords_" + filename.split( SPLIT_CHAR )[-1].split('.')[0] + '.txt'
    mat_points = read_number_file( name_for_points )
    mat_edges_old = read_number_file( name_for_edges )
    mat_edges = [x + 1 for x in mat_edges_old] #if numbering starts from 0, comment if numbering starts from 1
    mat_shape = read_number_file( name_for_shape )
    pil_image = Image.open(filename).convert('RGB') 
    open_cv_image = np.array(pil_image) 
    
    # Resize images
    # to a square output size)
    (top_256, left_256, ratio_square, img_256) = resize_image_rectangle(open_cv_image, (dimension_0,dimension_1)) # resize on the image space with artifacts
    (top_256, left_256, ratio_square) = find_ratio_rectangle(mat_shape, (dimension_0, dimension_1)) # find the amount to resize
    (mat_shape_256) = resize_shape(top_256, left_256, ratio_square, mat_shape) # resize the shape
    (mat_skel_256) = resize_skeleton(top_256, left_256, ratio_square, mat_points) #resize the skeleton
    
    # to a rectangular output size
    (top_2048, left_2048, ratio_rect, img_2048_1024) = resize_image_rectangle(open_cv_image, (desired_size_rect_0,desired_size_rect_1))
    (top_2048, left_2048, ratio_rect) = find_ratio_rectangle(mat_shape, (desired_size_rect_0, desired_size_rect_1))    
    (mat_skel_2048_1024) = resize_skeleton(top_2048, left_2048, ratio_rect, mat_points)  
    (mat_shape_2048_1024) = resize_shape(top_2048, left_2048, ratio_rect, mat_shape)
   

    # Rotate images and coordinates if needed
    if(img_2048_1024.shape[1] < img_2048_1024.shape[0]):
        img_2048_1024 = np.flip(img_2048_1024, 1)
        img_2048_1024 = np.rot90(img_2048_1024)
        img_256 = np.flip(img_256, 1)
        img_256 = np.rot90(img_256)     

        
        for i in range(int(len(mat_skel_256) / 2)):
            tmp = mat_skel_256[i * 2] 
            mat_skel_256[i * 2] = mat_skel_256[i * 2 + 1] 
            mat_skel_256[i * 2 + 1] = tmp 
            
        for i in range(int(len(mat_shape_256) / 2)):
            tmp = mat_shape_256[i * 2] 
            mat_shape_256[i * 2] = mat_shape_256[i * 2 + 1]  
            mat_shape_256[i * 2 + 1] =  tmp      
            
        for i in range(int(len(mat_skel_2048_1024) / 2)):
            tmp = mat_skel_2048_1024[i * 2] 
            mat_skel_2048_1024[i * 2] =  mat_skel_2048_1024[i * 2 + 1] 
            mat_skel_2048_1024[i * 2 + 1] = tmp 
            
        for i in range(int(len(mat_shape_2048_1024) / 2)):
            tmp = mat_shape_2048_1024[i * 2]
            mat_shape_2048_1024[i * 2] = mat_shape_2048_1024[i * 2 + 1]
            mat_shape_2048_1024[i * 2 + 1] = tmp

            
    # Write the output to files.
    
    name_sk_square = outdirResized_256 + labels + "skel_" + name
    
    name_sh_square = outdirResized_256 + shapes +  name
    

    write_skeleton(mat_skel_256, mat_edges, dimension_0 + 2, dimension_1 + 2, name_sk_square)
    write_shape(mat_shape_256, dimension_0 + 2,dimension_1 + 2, name_sh_square)

    file_skel_points_256 = open(outdirResized_256 + labels + "skelpoints_" + name + '.txt', "w") 
    for i in range(int(len(mat_skel_256)/2)):
        file_skel_points_256.write(str("{0:4.8f}".format(mat_skel_256[i*2]))+"\t"+str("{0:4.8f}".format(mat_skel_256[i*2+1]))+"\n")
    file_skel_points_256.close()
    
    
    file_skel_edges = open(outdirResized_256 + labels + "skeledges_" + name + '.txt', "w") 
    for i in range(int(len(mat_edges)/2)):
        file_skel_edges.write(str(int(mat_edges[i*2] - 1))+"\t"+str(int(mat_edges[i*2+1] - 1))+"\n")
    file_skel_edges.close()
    
    
    file_shape_points_256 = open(outdirResized_256 + shapes + "coords_" + name + '.txt', "w") 
    for i in range(int(len(mat_shape_256)/2)):
        file_shape_points_256.write(str("{0:4.8f}".format(mat_shape_256[i*2]))+"\t"+str("{0:4.8f}".format(mat_shape_256[i*2+1]))+"\n")   
    file_shape_points_256.close()
    

print('all resizing done!')


   
    
from PIL import Image
import os
import cv2
import glob
import numpy as np
import csv
from math import floor
from shutil import rmtree
from math import sqrt
from random import randint


SPLIT_CHAR = '\\'  # For Windows
#SPLIT_CHAR = '/' #For Linux
inDir = 'Input/Labels_Shapes/'
outDir = 'Output/'
yourRes = 'Input/Results_User/'
labels = 'labels/'  # Contains the information for the skeleton of the images
shapes = 'shapes/'  # Contains the information for the shapes of the images
to_test = 'to_test/'  # Contains the information for the shapes of the images
score_file = yourRes.split('/')[1] + '_' + 'Score.csv'  # Computes the scores


# Create the folders where the files will be written
try:
    if os.path.exists(outDir):
        rmtree(outDir)
except OSError, e:
    print e

if not os.path.exists(outDir):
    os.makedirs(outDir)
if not os.path.exists(outDir + to_test):
    os.makedirs(outDir + to_test)
if not os.path.exists(outDir + labels):
    os.makedirs(outDir + labels)
if not os.path.exists(outDir + shapes):
    os.makedirs(outDir + shapes)

# Default image size
DEFAULT_SIZE = 256


# Reading the shape of an image
def read_number_file(filename1):
    with open(filename1) as file_with_numbers:
        content = file_with_numbers.readlines()
    content = [x.strip() for x in content]
    mat = []
    for line_with_two_numbers in content:
        number_no = line_with_two_numbers.split()
        if (len(number_no) == 2):
            mat.append(float(number_no[0]))
            mat.append(float(number_no[1]))

    return mat


# Always give smaller side first as input's desired_size
# Finds the ratio to resize the current shape to the desired size
def find_ratio_rectangle(current_shape, current_skeleton, desired_size):
    min_x = current_shape[0]
    max_x = current_shape[0]
    min_y = current_shape[1]
    max_y = current_shape[1]
    for i in range(int(len(current_shape) / 2)):
        if (current_shape[2 * i + 0] < min_x):
            min_x = current_shape[2 * i + 0]
        if (current_shape[2 * i + 0] > max_x):
            max_x = current_shape[2 * i + 0]
        if (current_shape[2 * i + 1] < min_y):
            min_y = current_shape[2 * i + 1]
        if (current_shape[2 * i + 1] > max_y):
            max_y = current_shape[2 * i + 1]
    size_x = max_x - min_x;
    size_y = max_y - min_y;
    current_image_size = (size_x, size_y)
    for i in range(int(len(current_shape) / 2)):
        current_shape[2 * i + 0] -= min_x
        current_shape[2 * i + 1] -= min_y
    for i in range(int(len(current_skeleton) / 2)):
        current_skeleton[2 * i + 0] -= min_x
        current_skeleton[2 * i + 1] -= min_y
    switch = 0
    if (current_image_size[0] < current_image_size[1]):
        ratio_0 = float(desired_size[0] - 3) / current_image_size[0]
        ratio_1 = float(desired_size[1] - 3) / current_image_size[1]
    else:
        ratio_0 = float(desired_size[0] - 3) / current_image_size[1]
        ratio_1 = float(desired_size[1] - 3) / current_image_size[0]
        switch = 1
    ratio = ratio_1
    if (ratio_0 < ratio_1):
        ratio = ratio_0
    ratio = floor(ratio * 100000000000) / 100000000000

    delta_w = desired_size[(1 + switch) % 2] - (current_image_size[1] * ratio)
    delta_h = desired_size[(switch) % 2] - (current_image_size[0] * ratio)
    top = delta_h // 2
    left = delta_w // 2

    return (top, left, ratio, current_shape, current_skeleton, switch)


# Resize the skeleton
def resize_skeleton(top, left, ratio, mat_points):
    mat_points_updated = list(mat_points)

    for i in range(int(len(mat_points))):
        mat_points_updated[i] = mat_points[i] * ratio

    shiftCol = left + 1
    shiftRow = top + 1

    for i in range(int(len(mat_points_updated) / 2)):
        mat_points_updated[i * 2] += shiftRow

    for i in range(int(len(mat_points_updated) / 2)):
        mat_points_updated[i * 2 + 1] += shiftCol

    for i in range(int(len(mat_points_updated) / 2)):
        tmp = mat_points_updated[i * 2]
        mat_points_updated[i * 2] = mat_points_updated[i * 2 + 1] + 1
        mat_points_updated[i * 2 + 1] = tmp + 1

    return (mat_points_updated)


# Resize the shape,  the shape should be  cropped
def resize_shape(top, left, ratio, mat_shape):
    scaled_shape = list(mat_shape)
    for i in range(int(len(mat_shape))):
        scaled_shape[i] = ratio * mat_shape[i]
        # scaled_shape[i] =  ratio * int(round(mat_shape[i]))
    shiftCol = left + 1
    shiftRow = top + 1

    for i in range(int(len(mat_shape) / 2)):
        scaled_shape[i * 2 + 1] += shiftCol
    for i in range(int(len(mat_shape) / 2)):
        scaled_shape[i * 2 + 0] += shiftRow

    return (scaled_shape)


# Writing an image skeleton, should be  cropped
def write_skeleton(mat_points, mat_edges, height, width, name):
    blank_image = np.zeros((height, width, 3), np.uint8)
    for i in range(int(len(mat_edges) / 2)):
        start_edge = int(mat_edges[i * 2 + 0]) - 1
        end_edge = int(mat_edges[i * 2 + 1]) - 1
        start_point_x = int(round(mat_points[2 * start_edge + 0])) - 1
        start_point_y = int(round(mat_points[2 * start_edge + 1])) - 1
        end_point_x = int(round(mat_points[2 * end_edge + 0])) - 1
        end_point_y = int(round(mat_points[2 * end_edge + 1])) - 1
        cv2.line(blank_image, (start_point_y, start_point_x), (end_point_y, end_point_x), (255, 255, 255), 1)

    # cv2.imwrite(name, blank_image)
    pil_im = Image.fromarray(blank_image)
    pil_im.save(name, "PNG")


# Writing a shape image
def write_shape(mat_shape, height, width, name):
    mat_shape.append(mat_shape[0])
    mat_shape.append(mat_shape[1])
    blank_image = np.zeros((height, width, 3), np.uint8)
    array_shape = np.array(mat_shape).reshape(int(len(mat_shape) / 2), 2)
    cv2.fillPoly(blank_image, np.array([array_shape], dtype=np.int32), (255, 255, 255))

    # cv2.imwrite(name, blank_image)
    pil_im = Image.fromarray(blank_image)
    pil_im.save(name, "PNG")


# Preprocess to match the data
def preprocess():
    Error = 0
    # Check if all images are of the same size
    for filename in glob.glob(yourRes + '/*.png'):
        image_pil = Image.open(filename).convert('RGB')
        candidate_label = np.array(image_pil)
        dim_x = candidate_label.shape[0]
        dim_y = candidate_label.shape[1]
    found = False
    for filename in glob.glob(yourRes + '/*.png'):
        image_pil = Image.open(filename).convert('RGB')
        candidate_label = np.array(image_pil)
        if (dim_x != candidate_label.shape[0] or dim_y != candidate_label.shape[1]):
            found = True
    if (found):
        print("ERROR: The files are not all of the same size")
        Error += 1
    for filename in glob.glob(inDir + '/*.png'):

        # Read image correct label
        image_pil = Image.open(filename).convert('RGB')
        correct_label = np.array(image_pil)
        image_pil.close()
        # Read the corresponding image from the dataset of the User
        name = filename.split(SPLIT_CHAR)[-1].split('.')[0]
        file_to_read = yourRes + name + '.png'
        name_for_updated_label = outDir + labels + name + '.png'
        name_for_updated_shape = outDir + shapes + name + '.png'
        name_for_updated_user_result = outDir + to_test + name + '.png'
        print(name)
        # Check if the file exists in User's folder
        if os.path.isfile(file_to_read):

            image_pil = Image.open(file_to_read).convert('RGB')
            candidate_label = np.array(image_pil)
            current_image_size = candidate_label.shape[0:2]

            # Check if the size is adequate
            if (current_image_size[0] > 64 and current_image_size[1] > 64):
                # Check if the Image is  given in the default size 256 x 256 and update accordingly the labeling
                if (current_image_size[0] != DEFAULT_SIZE or current_image_size[1] != DEFAULT_SIZE):
                    print("Image for testing is resized for comparison")
                    size = [current_image_size[0], current_image_size[1]]
                    if (size[1] < size[0]):
                        tmp = size[1]
                        size[1] = int(size[0])
                        size[0] = tmp
                        candidate_label = np.flip(candidate_label, 1)
                        candidate_label = np.rot90(candidate_label)
                    name_for_edges = inDir + "skeledges_" + name + '.txt'
                    name_for_points = inDir + "skelpoints_" + name + '.txt'
                    name_for_shape = inDir + "coords_" + name + '.txt'
                    mat_points = read_number_file(name_for_points)
                    mat_edges_old = read_number_file(name_for_edges)
                    mat_edges = [x + 1 for x in
                                 mat_edges_old]  # if numbering starts from 0, comment if numbering starts from 1
                    mat_shape = read_number_file(name_for_shape)
                    (top_256, left_256, ratio_square, mat_shape, mat_points, switch) = find_ratio_rectangle(mat_shape,
                                                                                                            mat_points,
                                                                                                            (size[0],
                                                                                                             size[
                                                                                                                 1]))  # find the amount to resize
                    (mat_skel_256) = resize_skeleton(top_256, left_256, ratio_square, mat_points)  # resize the skeleton
                    (mat_shape_256) = resize_shape(top_256, left_256, ratio_square, mat_shape)  # resize the shape

                    if (switch == 0):
                        for i in range(int(len(mat_skel_256) / 2)):
                            tmp = mat_skel_256[i * 2]
                            mat_skel_256[i * 2] = mat_skel_256[i * 2 + 1]
                            mat_skel_256[i * 2 + 1] = tmp

                        for i in range(int(len(mat_shape_256) / 2)):
                            tmp = mat_shape_256[i * 2]
                            mat_shape_256[i * 2] = mat_shape_256[i * 2 + 1]
                            mat_shape_256[i * 2 + 1] = tmp

                    write_skeleton(mat_skel_256, mat_edges, size[0], size[1], name_for_updated_label)
                    write_shape(mat_shape_256, size[0], size[1], name_for_updated_shape)
                else:
                    # Write_image Convert given image to
                    image_pil = Image.fromarray(correct_label)
                    correct_skeleton = cv2.inRange(correct_label, (0, 0, 255, 0), (0, 0, 255, 0))
                    correct_shape = cv2.inRange(correct_label, (255, 255, 255, 0), (255, 255, 255, 0))
                    correct_shape = cv2.add(correct_shape, correct_skeleton)
                    im_pil = Image.fromarray(correct_skeleton)
                    im_pil.save(name_for_updated_label)
                    im_pil = Image.fromarray(correct_shape)
                    im_pil.save(name_for_updated_shape)
                image_pil.close()
                # Convert the given Image to Black and White
                image_pil = Image.open(file_to_read).convert('RGB')
                user_result = np.array(image_pil)
                image_pil.close()
                # If blue pixels for the skeleton
                user_result_1 = cv2.inRange(candidate_label, (0, 0, 255, 0), (0, 0, 255, 0))
                im_pil = Image.fromarray(user_result_1)
                im_pil.save(name_for_updated_user_result)
            else:
                print ("ERROR: File is considered too small to capture skeleton details, it influences your error score.")
                Error += 1
        else:  # If the file did not exist update accordingly user's error
            print("ERROR: File does not exist in your data, it influences your error score.")
            Error += 1
    return Error


number_of_preprocessing_errors = preprocess()  # Returns the number of errors from preprocessing
No_Files = 0
Error = 0

# Write the result in a csv File and print it
with open(score_file, 'w') as csvfile:
    fieldnames = ['Name', 'Label pix', 'Shape pix',
                  'Out of shape', 'FP', 'FN', 'TP', 'TN',
                  'pix', 'F1', 'balanced accuracy'
                  ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
csvfile.close()

total_F1 = 0
total_balanced_accuracy = 0
for filename in glob.glob(outDir + to_test + '/*.png'):
    No_Files += 1
    FP = 0
    FN = 0
    OUT = 0

    # Read image
    image_pil = Image.open(filename).convert('RGB')
    components = np.array(image_pil)
    IMAGE_SIZE = components.shape[0] * components.shape[1]
    image_pil.close()
    components_gray = cv2.cvtColor(components, cv2.COLOR_BGR2GRAY)

    name = filename.split(SPLIT_CHAR)[-1].split('.')[0]

    name_for_updated_label = outDir + labels + name + '.png'
    name_for_updated_shape = outDir + shapes + name + '.png'
    name_for_updated_user_result = outDir + to_test + name + '.png'

    image_pil = Image.open(name_for_updated_label).convert('RGB')
    label = np.array(image_pil)
    label_gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    image_pil.close()

    image_pil = Image.open(name_for_updated_shape).convert('RGB')
    shape = np.array(image_pil)
    shape_gray = cv2.cvtColor(shape, cv2.COLOR_BGR2GRAY)
    image_pil.close()

    image_pil = Image.open(name_for_updated_user_result).convert('RGB')
    answer = np.array(image_pil)
    answer_gray = cv2.cvtColor(answer, cv2.COLOR_BGR2GRAY)
    image_pil.close()

    shape_pix = cv2.countNonZero(shape_gray)
    skl_pix = cv2.countNonZero(label_gray)

    not_in_shape = cv2.bitwise_not(shape)
    outside = cv2.bitwise_and(answer, not_in_shape)
    im_pil = Image.fromarray(outside)

    not_in_label = cv2.bitwise_not(label)
    not_in_answer = cv2.bitwise_not(answer)
    false_positives = cv2.bitwise_and(answer, not_in_label)
    false_negatives = cv2.bitwise_and(label, not_in_answer)

    true_positives = cv2.bitwise_and(answer, label)
    true_negatives = cv2.bitwise_and(not_in_answer, not_in_label)

    false_positives_gray = cv2.cvtColor(false_positives, cv2.COLOR_BGR2GRAY)
    false_negatives_gray = cv2.cvtColor(false_negatives, cv2.COLOR_BGR2GRAY)
    true_positives_gray = cv2.cvtColor(true_positives, cv2.COLOR_BGR2GRAY)
    true_negatives_gray = cv2.cvtColor(true_negatives, cv2.COLOR_BGR2GRAY)
    outside_gray = cv2.cvtColor(outside, cv2.COLOR_BGR2GRAY)
    FP = cv2.countNonZero(false_positives_gray)
    FN = cv2.countNonZero(false_negatives_gray)
    OUT = cv2.countNonZero(outside_gray)
    TP = cv2.countNonZero(true_positives_gray)
    TN = cv2.countNonZero(true_negatives_gray)

    label_gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    answer_gray = cv2.cvtColor(answer, cv2.COLOR_BGR2GRAY)
    im2, contours_answer, hierarchy = cv2.findContours(answer_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    im2, contours_label, hierarchy = cv2.findContours(label_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    recall = 1.0 * TP / skl_pix
    precision = 1.0 * TP / (TP + FP)
    TPR = 1.0 * TP / skl_pix
    TNR = 1.0 * TN / (IMAGE_SIZE - skl_pix)

    F1 = (1 + 1.0) * (precision * recall) / (1 * 1 * precision + recall)  # https://en.wikipedia.org/wiki/F1_score
    balanced_accuracy = (TPR + TNR) / 2.0  # https://en.wikipedia.org/wiki/Precision_and_recall
    total_F1 += F1
    total_balanced_accuracy += balanced_accuracy

    with open(score_file, 'a') as csvfile:
        fieldnames = ['Name', 'Label pix', 'Shape pix',
                      'Out of shape', 'FP', 'FN', 'TP', 'TN',
                      'pix', 'F1', 'balanced accuracy'
                      ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'balanced accuracy': balanced_accuracy,
                         'F1': F1,
                         'pix': IMAGE_SIZE,
                         'Label pix': skl_pix, 'Shape pix': shape_pix, 'Name': name,
                         'Out of shape': OUT, 'FP': FP, 'FN': FN, 'TP': TP, 'TN': TN})
    csvfile.close()

# Compute the average error from all files in the dataset
if (No_Files != 0 and number_of_preprocessing_errors == 0):

    total_F1 /= No_Files
    total_balanced_accuracy /= No_Files
    print('Your total score is:', total_F1)
    with open(score_file, 'a') as csvfile:
        fieldnames = ['Name', 'Label pix', 'Shape pix',
                      'Out of shape', 'FP', 'FN', 'TP', 'TN',
                      'pix', 'F1', 'balanced accuracy'
                      ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'F1': total_F1, 'balanced accuracy': total_balanced_accuracy})
    csvfile.close()
else:
    print("Your total score is: 0. Correct the errors of the preprocessing of the files and try again")

print('The End\n')
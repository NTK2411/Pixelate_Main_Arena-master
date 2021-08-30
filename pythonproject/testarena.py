import gym
import pix_main_arena
import numpy as np
import os
import time
import math
import pybullet as p
import cv2
import cv2.aruco as aruco
import json
# import image_processing as impr
# import dict

###################

def color_recognise(h, s, v):
    '''

    :param h: hue of color
    :param s: saturation
    :param v: value

    WHITE  = (0,0,255)
    RED    = (0,255,255)
    YELLOW = (30,255,255)
    GREEN  = (60,255,255)
    CYAN   = (90,255,255)
    BLUE   = (120,255,255)
    PINK   = (150,255,255)
    DULL GREEN = (76,255,104)

    :return:
    WHITE  = 0
    RED    = 1
    YELLOW = 2
    GREEN  = 3
    CYAN   = 4 #
    BLUE   = 5
    PINK   = 6 #
    DULL GREEN = 7 #
    '''

    color_code = -1
    ####CALIBRATED FROM STAGE1 GOOGLE DRIVE####
    # if (h == 0 and s == 0 and v == 255):
    #     color_code = 0
    # if (h == 0 and s == 255 and v == 255):
    #     color_code = 1
    # if (h == 30 and s == 255 and v == 255):
    #     color_code = 2
    # if (h == 60 and s == 255 and v == 255):
    #     color_code = 3
    # if (h == 90 and s == 255 and v == 255):
    #     color_code = 4
    # if (h == 120 and s == 255 and v == 255):
    #     color_code = 5
    # if (h == 150 and s == 255 and v == 255):
    #     color_code = 6
    # if (h == 76 and s == 255 and v == 104):
    #     color_code = 7

    ####CALIBRATED FROM SAMPLE ARENA####
    if (h == 0 and s == 0 and v > 100):
        color_code = 0
    if (h == 0 and s == 255 and v > 100):
        color_code = 1
    if (h == 30 and s == 255 and v > 100):
        color_code = 2
    if (h == 60 and s == 255 and v > 100):
        color_code = 3
    if (h == 90 and s == 255 and v > 100):
        color_code = 4
    if (h == 120 and s == 255 and v > 100):
        color_code = 5
    if (h == 150 and s > 100 and v > 100):
        color_code = 6
    if (h == 60 and s == 255 and v == 91):
        color_code = 7

    return color_code

def shape_detection(roi, h, s, v):
    roi = cv2.resize(roi, (250, 250))
    lower = np.array([h - 3, s - 3, v - 3])
    upper = np.array([h, s, v])
    '''
    :param roi: IMAGE OF SINGLE TILE
    :param h: HUE OF BLUE           120
    :param s: SATURATION OF BLUE    255
    :param v: VALUE OF BLUE         255

    :return:
    NOT FOUND = -1
    TRAINGLE    = 1
    SQUARE      = 2
    CIRCLE      = 3
    '''
    shape_code = -1
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(hsv, lower, upper)

    triangle_corners = np.zeros((3, 2), int)
    triangle_direction = np.zeros(2, int)  # (y,x)

    bgr_onlyblue = cv2.bitwise_and(bgr, bgr, mask=mask)
    gray_onlyblue = cv2.cvtColor(bgr_onlyblue, cv2.COLOR_BGR2GRAY)
    _, threshold_binary = cv2.threshold(gray_onlyblue, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = np.zeros(2)
    w = roi.shape[0]
    h = roi.shape[1]

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        # cv2.drawContours(roi, [approx], 0, (0), 1)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx)>2:
            for i in range(3):
                triangle_corners[i][0] = approx.ravel()[2 * i]
                triangle_corners[i][1] = approx.ravel()[2 * i+1]

        # print(cv2.contourArea(cnt), end=' ')
        if (area[0] < cv2.contourArea(cnt) and cv2.contourArea(cnt) < w * h - 1000):
            area[0] = cv2.contourArea(cnt)
            area[1] = len(approx)
            # print(area[1])


    ##############################
    # cv2.imshow('bgr', bgr)
    # cv2.imshow('marked', roi)
    # cv2.waitKey(10000)
    #############################


    if area[1] == 3:
        print('waah bete waah')
        shape_code = 1
        # triangle direction detection
        # triangle_corners
        diff = np.zeros(6, int)
        diff[0] = abs(triangle_corners[2][0] - triangle_corners[1][0])  # 0
        diff[1] = abs(triangle_corners[0][0] - triangle_corners[2][0])  # 1
        diff[2] = abs(triangle_corners[0][0] - triangle_corners[1][0])  # 2

        diff[3] = abs(triangle_corners[2][1] - triangle_corners[1][1])  # 0
        diff[4] = abs(triangle_corners[0][1] - triangle_corners[2][1])  # 1
        diff[5] = abs(triangle_corners[0][1] - triangle_corners[1][1])  # 2

        indexofmin = np.where(diff == np.amin(diff))

        print(indexofmin[0][0], diff[indexofmin[0][0]])
        print(diff)

        if (indexofmin[0][0] <= 2 and indexofmin[0][0] >= 0):  # left,right
            triangle_direction[0] = 0  # (y,x)
            if indexofmin[0][0] == 0:
                triangle_direction[1] = np.sign(
                    triangle_corners[0][0] - (triangle_corners[2][0] + triangle_corners[1][0]) / 2)  # (x,y)

            if indexofmin[0][0] == 1:
                triangle_direction[1] = np.sign(
                    triangle_corners[1][0] - (triangle_corners[2][0] + triangle_corners[1][0]) / 2)  # (x,y)

            if indexofmin[0][0] == 2:
                triangle_direction[1] = np.sign(
                    triangle_corners[2][0] - (triangle_corners[0][0] + triangle_corners[1][0]) / 2)  # (x,y)

        if (indexofmin[0][0] <= 5 and indexofmin[0][0] >= 3):  # up,down
            triangle_direction[1] = 0  # (y,x)
            if indexofmin[0][0] == 3:
                triangle_direction[0] = np.sign(
                    triangle_corners[0][1] - (triangle_corners[2][1] + triangle_corners[1][1]) / 2)  # (x,y)

            if indexofmin[0][0] == 4:
                triangle_direction[0] = np.sign(
                    triangle_corners[1][1] - (triangle_corners[2][1] + triangle_corners[1][1]) / 2)  # (x,y)

            if indexofmin[0][0] == 5:
                triangle_direction[0] = np.sign(
                    triangle_corners[2][1] - (triangle_corners[0][1] + triangle_corners[1][1]) / 2)  # (x,y)



    elif area[1] == 4:
        shape_code = 2
        triangle_direction = (0, 0)

    elif area[1] > 4:
        shape_code = 3
        triangle_direction = (0, 0)
    return shape_code, triangle_direction

def image_processing1(image, arena_dimensions):
    # cv2.imshow('image_procssing1-read', image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv',hsv)
    # fig, ax = plt.subplots(figsize=(10, 7))
    # plt.hist(image.flat,bins=100, range=(0,255))
    # plt.show()

    arena_information = np.zeros((arena_dimensions[0], arena_dimensions[1], 2), int)
    shapes_information = np.zeros((arena_dimensions[0], arena_dimensions[1], 2), int)
    '''
    ARGUMENTS:
        Y COORDINATE (INT)
        X COORDINATE (INT)
        COLOR CODE   (INT)
        SHAPE CODE   (INT)

    BLACK  = -1
    WHITE  = 0
    RED    = 1
    YELLOW = 2
    GREEN  = 3
    CYAN   = 4
    BLUE   = 5
    PINK   = 6
    DULL GREEN = 7

    SHAPES
    NULL        = 0
    TRIANGLE    = 1
    SQUARE      = 2
    CIRCLE      = 3
    '''
    for j in range(arena_dimensions[0]):
        for i in range(arena_dimensions[1]):
            arena_information[j][i][0] = -1

    tiles_centres = np.full((arena_dimensions[0], arena_dimensions[1], 2), -1, int)
    # print(tiles_centres)

    col = image.shape[0] // arena_dimensions[0]
    width = image.shape[1] // arena_dimensions[1]

    # for i in range(arena_dimensions[0]):
    #     image = cv2.line(image, (0, i * col), (image.shape[0], i * col), (0, 255, 0), 2)
    #
    # for i in range(arena_dimensions[1]):
    #     image = cv2.line(image, (i * width, 0), (i * width, image.shape[1]), (0, 255, 0), 2)

    l = np.array([0, 0, 90])
    u = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, l, u)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    ###REDUNDANT MORPHOLOGICAL OPERATIONS
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_mask = []
    for cnt in contours:
        area_mask.append(int(cv2.contourArea(cnt)))

    ###IN USE MORPHOLOGICAL OPERATIONS
    contours2, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_opcl = []
    box_length = 0
    count = 0

    for cnt in contours2:
        area_opcl.append(int(cv2.contourArea(cnt)))

        box_length = math.floor(math.sqrt(area_opcl[0]))
        print(box_length)

        x, y, w, h = cv2.boundingRect(cnt)
        centre_x = x + w // 2
        centre_y = y + h // 2
        # cv2.putText(image,"FOUND",(x,centre_y),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,0,0))

        ##################START ASSIGNING VALUES OF ARRAY
        print(hsv[y + 1, x + 1, 0], hsv[y + 1, x + 1, 1], hsv[y + 1, x + 1, 2])
        print(color_recognise(hsv[y + 1, x + 1, 0], hsv[y + 1, x + 1, 1], hsv[y + 1, x + 1, 2]))

        colorcode = str(color_recognise(hsv[y + 1, x + 1, 0], hsv[y + 1, x + 1, 1], hsv[y + 1, x + 1, 2]))
        # colorcode = 'STEVE'

        if (color_recognise(hsv[centre_y, centre_x, 0], hsv[centre_y, centre_x, 1], hsv[centre_y, centre_x, 2]) == 5):
            # cv2.putText(image, '5', (centre_x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))
            roi = image[y + 1:y + box_length, x + 1:x + box_length]
            # text= str(count)
            # cv2.imshow(text,roi)
            # cv2.waitKey(10000)

            # shapecode = str(count)
            # shapecode=str(shape_detection(roi, 120, 255, 227))#227 to 255

            # print(shape_detection(roi,120,255,255),count)
            count += 1
            # cv2.putText(image, shapecode, (centre_x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0))

        for i in range(arena_dimensions[0]):
            if (i * col < centre_y and (i + 1) * col > centre_y):
                y_coordinate = i
        for i in range(arena_dimensions[1]):
            if (i * width < centre_x and (i + 1) * width > centre_x):
                x_coordinate = i

        colorcode = str(y_coordinate) + str(x_coordinate)
        # cv2.putText(image,colorcode, (x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))

        ###upar pooree testing karlee fraaaanzzz

        arena_information[y_coordinate][x_coordinate][0] = color_recognise(hsv[y + 1, x + 1, 0], hsv[y + 1, x + 1, 1],
                                                                           hsv[y + 1, x + 1, 2])
        if (color_recognise(hsv[centre_y, centre_x, 0], hsv[centre_y, centre_x, 1], hsv[centre_y, centre_x, 2]) == 5):
            roi = image[y + 1:y + box_length, x + 1:x + box_length]
            arena_information[y_coordinate][x_coordinate][1], shapes_information[y_coordinate][
                x_coordinate] = shape_detection(roi, 120, 255, 227)  ###THIS VALUE NEEDS TO BE ADJUSTED

        tiles_centres[y_coordinate][x_coordinate] = (centre_y, centre_x)
        pos = [centre_x, centre_y]
        # add_steay_pos_error(pos,hsv)
        # cv2.circle(image, (pos[0],pos[1]), 3, 255, -1)
        ##################
        # cv2.putText(image, colorcode, (x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))

    mask = cv2.medianBlur(mask, 1)
    # cv2.imshow('BLACK REMOVED', mask)
    # cv2.imshow('MARKED', image)
    # cv2.imshow('closing', closing)

    length1 = len(area_mask)
    length2 = len(area_opcl)
    print(length1, length2, area_opcl[0])
    # for i in range(98):
    #     print(area_mask[i])

    # for i in range(length):
    #     dif = area_opcl[i]-area_mask[i]
    #     if dif != 0:
    #         print(i," : ",dif)
    cv2.waitKey(0)
    return arena_information, box_length, tiles_centres, shapes_information
####################
def graph_dict(arena_information,arena_dimensions,weight,shape_information,posi,posi2):
    y_max=arena_dimensions[0]
    x_max=arena_dimensions[1]
    y_ignore, x_ignore= coord_con_int(posi)
    y_ignore2, x_ignore2 = coord_con_int(posi2)
    # y_ignore = int(posi[1])
    # x_ignore = int(posi[2])
    # y_ignore2 = int(posi2[1])
    # x_ignore2 = int(posi2[2])
    d={}
    for j in range(y_max):
        for i in range(x_max):

            if arena_information[j][i][0] in[0,1,2,3,5]:
                if arena_information[j][i][1] != 1:
                    # if arena_information[j][i][0] in [0, 1, 2, 3, 5]:
                    d[coord_con_str([j,i])] = {}
                    # if arena_information[j][i][0] in [4, 6, 7]:
                    #     if j == y_ignore and i == x_ignore:
                    #         d['d' + str(j) + str(i)] = {}

                    #LOWER TILE
                    if j + 1 < y_max:
                        color_code=arena_information[j+1][i][0]
                        shape_code=arena_information[j+1][i][1]
                        shape_dir=shape_information[j+1][i]
                        if color_code in[0,1,2,3,5]:
                            if shape_code == 1:
                                # print("anti parallel10")
                                move_vector = np.array([1,0],int)
                                dot_product = move_vector[0]*shape_dir[0] + move_vector[1]*shape_dir[1]
                                if dot_product >= 0:
                                    print("mauj kardi "+str(j)+str(i))
                                    d[coord_con_str([j,i])][coord_con_str([j+1,i])] = weight[color_code]

                            else:
                                #yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                                # d['d' + str(j) + str(i)] = {}
                                d[coord_con_str([j,i])][coord_con_str([j+1,i])]=weight[color_code]
                        if color_code in [4,6,7]:
                            if j+1 == y_ignore and i == x_ignore:
                                d[coord_con_str([j,i])][coord_con_str([j+1,i])]=weight[color_code]
                            if j+1 == y_ignore2 and i == x_ignore2:
                                d[coord_con_str([j,i])][coord_con_str([j+1,i])]=weight[color_code]

                    #UPPER TILE\
                    if j-1 > -1:
                        color_code=arena_information[j-1][i][0]
                        shape_code=arena_information[j-1][i][1]
                        shape_dir = shape_information[j - 1][i]
                        if color_code in[0,1,2,3,5]:
                            if shape_code == 1:
                                # print("anti parallel-10")
                                move_vector = np.array([-1, 0],int)
                                dot_product = move_vector[0]*shape_dir[0] + move_vector[1]*shape_dir[1]
                                if dot_product >= 0:
                                    print("mauj kardi "+str(j)+str(i))
                                    d[coord_con_str([j,i])][coord_con_str([j-1,i])] = weight[color_code]

                            else:
                                # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                                # d['d' + str(j) + str(i)] = {}
                                d[coord_con_str([j,i])][coord_con_str([j-1,i])] = weight[color_code]
                        if color_code in [4,6,7]:
                            if j-1 == y_ignore and i == x_ignore:
                                d[coord_con_str([j,i])][coord_con_str([j-1,i])] = weight[color_code]
                            if j - 1 == y_ignore2 and i == x_ignore2:
                                d[coord_con_str([j,i])][coord_con_str([j-1,i])] = weight[color_code]
                    #RIGHT TILE
                    if i + 1 < x_max:
                        color_code=arena_information[j][i+1][0]
                        shape_code=arena_information[j][i+1][1]
                        shape_dir = shape_information[j][i + 1]
                        if color_code in[0,1,2,3,5]:
                            if shape_code == 1:
                                # print("anti parallel01")
                                move_vector = np.array([0, 1],int)
                                dot_product = move_vector[0]*shape_dir[0] + move_vector[1]*shape_dir[1]
                                if (dot_product >= 0):
                                    print("mauj kardi "+str(j)+str(i))
                                    d[coord_con_str([j,i])][coord_con_str([j,i+1])] = weight[color_code]

                            else:
                                # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                                # d['d' + str(j) + str(i)] = {}
                                d[coord_con_str([j,i])][coord_con_str([j,i+1])] = weight[color_code]
                        if color_code in [4,6,7]:
                            if j == y_ignore and i+1 == x_ignore:
                                d[coord_con_str([j,i])][coord_con_str([j,i+1])] = weight[color_code]
                            if j == y_ignore2 and i+1 == x_ignore2:
                                d[coord_con_str([j,i])][coord_con_str([j,i+1])] = weight[color_code]

                    #LEFT TILE
                    if i - 1 > -1:
                        color_code=arena_information[j][i-1][0]
                        shape_code=arena_information[j][i-1][1]
                        shape_dir = shape_information[j][i - 1]
                        if color_code in[0,1,2,3,5]:
                            if shape_code == 1:
                                # print("anti parallel0-1")
                                move_vector = np.array([0, -1],int)
                                dot_product = move_vector[0]*shape_dir[0] + move_vector[1]*shape_dir[1]
                                if dot_product >= 0:
                                    print("mauj kardi "+str(j)+str(i))
                                    d[coord_con_str([j,i])][coord_con_str([j,i-1])] = weight[color_code]

                            else:
                                # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                                # d['d' + str(j) + str(i)] = {}
                                d[coord_con_str([j,i])][coord_con_str([j,i-1])] = weight[color_code]
                        if color_code in [4,6,7]:
                            if j == y_ignore and i-1 == x_ignore:
                                d[coord_con_str([j,i])][coord_con_str([j,i-1])] = weight[color_code]
                            if j == y_ignore2 and i-1 == x_ignore2:
                                d[coord_con_str([j,i])][coord_con_str([j,i-1])] = weight[color_code]
                if arena_information[j][i][1] == 1:
                    d[coord_con_str([j,i])]= {}
                    shape_code = arena_information[j][i][1]
                    shape_dir = shape_information[j][i]
                    color_code = arena_information[j+shape_dir[0]][i+shape_dir[1]][0]
                    d[coord_con_str([j,i])][coord_con_str([j + shape_dir[0],i + shape_dir[1]])] = weight[color_code]
                # d['d'+str(j)+str(i)]=d+str(j)+str(i)

            if arena_information[j][i][0] in [4, 6, 7]:
                if j == y_ignore and i == x_ignore:
                    d[coord_con_str([j,i])] = {}
                    # LOWER TILE
                    if j + 1 < y_max:
                        color_code = arena_information[j + 1][i][0]
                        shape_code = arena_information[j + 1][i][1]
                        shape_dir = shape_information[j + 1][i]
                        if color_code in [0, 1, 2, 3, 5]:
                            if shape_code == 1:
                                # print("anti parallel10")
                                move_vector = np.array([1, 0], int)
                                dot_product = move_vector[0] * shape_dir[0] + move_vector[1] * shape_dir[1]
                                if dot_product >= 0:
                                    print("mauj kardi " + str(j) + str(i))
                                    d[coord_con_str([j,i])][coord_con_str([j+1,i])] = weight[color_code]

                            else:
                                # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                                # d['d' + str(j) + str(i)] = {}
                                d[coord_con_str([j,i])][coord_con_str([j+1,i])] = weight[color_code]
                        if color_code in [4, 6, 7]:
                            if j + 1 == y_ignore and i == x_ignore:
                                d[coord_con_str([j,i])][coord_con_str([j+1,i])] = weight[color_code]
                            if j + 1 == y_ignore2 and i == x_ignore2:
                                d[coord_con_str([j,i])][coord_con_str([j+1,i])] = weight[color_code]

                    # UPPER TILE\
                    if j - 1 > -1:
                        color_code = arena_information[j - 1][i][0]
                        shape_code = arena_information[j - 1][i][1]
                        shape_dir = shape_information[j - 1][i]
                        if color_code in [0, 1, 2, 3, 5]:
                            if shape_code == 1:
                                # print("anti parallel-10")
                                move_vector = np.array([-1, 0], int)
                                dot_product = move_vector[0] * shape_dir[0] + move_vector[1] * shape_dir[1]
                                if dot_product >= 0:
                                    print("mauj kardi " + str(j) + str(i))
                                    d[coord_con_str([j,i])][coord_con_str([j-1,i])] = weight[color_code]

                            else:
                                # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                                # d['d' + str(j) + str(i)] = {}
                                d[coord_con_str([j,i])][coord_con_str([j-1,i])] = weight[color_code]
                        if color_code in [4, 6, 7]:
                            if j - 1 == y_ignore and i == x_ignore:
                                d[coord_con_str([j,i])][coord_con_str([j-1,i])] = weight[color_code]
                            if j - 1 == y_ignore2 and i == x_ignore2:
                                d[coord_con_str([j,i])][coord_con_str([j-1,i])] = weight[color_code]

                    # RIGHT TILE
                    if i + 1 < x_max:
                        color_code = arena_information[j][i + 1][0]
                        shape_code = arena_information[j][i + 1][1]
                        shape_dir = shape_information[j][i + 1]
                        if color_code in [0, 1, 2, 3, 5]:
                            if shape_code == 1:
                                # print("anti parallel01")
                                move_vector = np.array([0, 1], int)
                                dot_product = move_vector[0] * shape_dir[0] + move_vector[1] * shape_dir[1]
                                if (dot_product >= 0):
                                    print("mauj kardi " + str(j) + str(i))
                                    d[coord_con_str([j,i])][coord_con_str([j,i+1])] = weight[color_code]

                            else:
                                # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                                # d['d' + str(j) + str(i)] = {}
                                d[coord_con_str([j,i])][coord_con_str([j,i+1])] = weight[color_code]
                        if color_code in [4, 6, 7]:
                            if j == y_ignore and i + 1 == x_ignore:
                                d[coord_con_str([j,i])][coord_con_str([j,i+1])] = weight[color_code]
                            if j == y_ignore2 and i + 1 == x_ignore2:
                                d[coord_con_str([j,i])][coord_con_str([j,i+1])] = weight[color_code]

                    # LEFT TILE
                    if i - 1 > -1:
                        color_code = arena_information[j][i - 1][0]
                        shape_code = arena_information[j][i - 1][1]
                        shape_dir = shape_information[j][i - 1]
                        if color_code in [0, 1, 2, 3, 5]:
                            if shape_code == 1:
                                # print("anti parallel0-1")
                                move_vector = np.array([0, -1], int)
                                dot_product = move_vector[0] * shape_dir[0] + move_vector[1] * shape_dir[1]
                                if dot_product >= 0:
                                    print("mauj kardi " + str(j) + str(i))
                                    d[coord_con_str([j,i])][coord_con_str([j,i-1])] = weight[color_code]

                            else:
                                # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                                # d['d' + str(j) + str(i)] = {}
                                d[coord_con_str([j,i])][coord_con_str([j,i-1])] = weight[color_code]
                        if color_code in [4, 6, 7]:
                            if j == y_ignore and i - 1 == x_ignore:
                                d[coord_con_str([j,i])][coord_con_str([j,i-1])] = weight[color_code]
                            if j == y_ignore2 and i - 1 == x_ignore2:
                                d[coord_con_str([j,i])][coord_con_str([j,i-1])] = weight[color_code]
                if j == y_ignore2 and i == x_ignore2:
                    d[coord_con_str([j,i])] = {}
                    # LOWER TILE
                    if j + 1 < y_max:
                        color_code = arena_information[j + 1][i][0]
                        shape_code = arena_information[j + 1][i][1]
                        shape_dir = shape_information[j + 1][i]
                        if color_code in [0, 1, 2, 3, 5]:
                            if shape_code == 1:
                                # print("anti parallel10")
                                move_vector = np.array([1, 0], int)
                                dot_product = move_vector[0] * shape_dir[0] + move_vector[1] * shape_dir[1]
                                if dot_product >= 0:
                                    print("mauj kardi " + str(j) + str(i))
                                    d[coord_con_str([j,i])][coord_con_str([j+1,i])] = weight[color_code]

                            else:
                                # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                                # d['d' + str(j) + str(i)] = {}
                                d[coord_con_str([j,i])][coord_con_str([j+1,i])] = weight[color_code]
                        if color_code in [4, 6, 7]:
                            if j + 1 == y_ignore and i == x_ignore:
                                d[coord_con_str([j,i])][coord_con_str([j+1,i])] = weight[color_code]
                            if j + 1 == y_ignore2 and i == x_ignore2:
                                d[coord_con_str([j,i])][coord_con_str([j+1,i])] = weight[color_code]

                    # UPPER TILE\
                    if j - 1 > -1:
                        color_code = arena_information[j - 1][i][0]
                        shape_code = arena_information[j - 1][i][1]
                        shape_dir = shape_information[j - 1][i]
                        if color_code in [0, 1, 2, 3, 5]:
                            if shape_code == 1:
                                # print("anti parallel-10")
                                move_vector = np.array([-1, 0], int)
                                dot_product = move_vector[0] * shape_dir[0] + move_vector[1] * shape_dir[1]
                                if dot_product >= 0:
                                    print("mauj kardi " + str(j) + str(i))
                                    d[coord_con_str([j,i])][coord_con_str([j-1,i])] = weight[color_code]

                            else:
                                # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                                # d['d' + str(j) + str(i)] = {}
                                d[coord_con_str([j,i])][coord_con_str([j-1,i])] = weight[color_code]
                        if color_code in [4, 6, 7]:
                            if j - 1 == y_ignore and i == x_ignore:
                                d[coord_con_str([j,i])][coord_con_str([j-1,i])] = weight[color_code]
                            if j - 1 == y_ignore2 and i == x_ignore2:
                                d[coord_con_str([j,i])][coord_con_str([j-1,i])] = weight[color_code]

                    # RIGHT TILE
                    if i + 1 < x_max:
                        color_code = arena_information[j][i + 1][0]
                        shape_code = arena_information[j][i + 1][1]
                        shape_dir = shape_information[j][i + 1]
                        if color_code in [0, 1, 2, 3, 5]:
                            if shape_code == 1:
                                # print("anti parallel01")
                                move_vector = np.array([0, 1], int)
                                dot_product = move_vector[0] * shape_dir[0] + move_vector[1] * shape_dir[1]
                                if (dot_product >= 0):
                                    print("mauj kardi " + str(j) + str(i))
                                    d[coord_con_str([j,i])][coord_con_str([j,i+1])] = weight[color_code]

                            else:
                                # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                                # d['d' + str(j) + str(i)] = {}
                                d[coord_con_str([j,i])][coord_con_str([j,i+1])] = weight[color_code]
                        if color_code in [4, 6, 7]:
                            if j == y_ignore and i + 1 == x_ignore:
                                d[coord_con_str([j,i])][coord_con_str([j,i+1])] = weight[color_code]
                            if j == y_ignore2 and i + 1 == x_ignore2:
                                d[coord_con_str([j,i])][coord_con_str([j,i+1])] = weight[color_code]

                    # LEFT TILE
                    if i - 1 > -1:
                        color_code = arena_information[j][i - 1][0]
                        shape_code = arena_information[j][i - 1][1]
                        shape_dir = shape_information[j][i - 1]
                        if color_code in [0, 1, 2, 3, 5]:
                            if shape_code == 1:
                                # print("anti parallel0-1")
                                move_vector = np.array([0, -1], int)
                                dot_product = move_vector[0] * shape_dir[0] + move_vector[1] * shape_dir[1]
                                if dot_product >= 0:
                                    print("mauj kardi " + str(j) + str(i))
                                    d[coord_con_str([j,i])][coord_con_str([j,i-1])] = weight[color_code]

                            else:
                                # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                                # d['d' + str(j) + str(i)] = {}
                                d[coord_con_str([j,i])][coord_con_str([j,i-1])] = weight[color_code]
                        if color_code in [4, 6, 7]:
                            if j == y_ignore and i - 1 == x_ignore:
                                d[coord_con_str([j,i])][coord_con_str([j,i-1])] = weight[color_code]
                            if j == y_ignore2 and i - 1 == x_ignore2:
                                d[coord_con_str([j,i])][coord_con_str([j,i-1])] = weight[color_code]

    return d
#####################
def path_dijkstra(graph,start,goal):
    shortest_distance = {}
    track_predecessor = {}
    unseenNodes = graph
    infinity = 999999
    track_path = []

    for node in unseenNodes:
        shortest_distance[node] = infinity
    shortest_distance[start] = 0

    while unseenNodes:

        min_distance_node = None

        for node in unseenNodes:
            if min_distance_node is None:
                min_distance_node = node
            elif shortest_distance[node] < shortest_distance[min_distance_node]:
                min_distance_node = node

        path_options = graph[min_distance_node].items()

        for child_node, weight in path_options:

            if weight + shortest_distance[min_distance_node] < shortest_distance[child_node]:
                shortest_distance[child_node] = weight + shortest_distance[min_distance_node]
                track_predecessor[child_node] = min_distance_node

        unseenNodes.pop(min_distance_node)

    currentNode = goal

    while currentNode != start:
        try:
            track_path.insert(0,currentNode)
            currentNode = track_predecessor[currentNode]
        except KeyError:
            print("Path is not reachable")
            break
    if shortest_distance[goal] != infinity:
        print("Shortest distance is " + str(shortest_distance[goal]))
        print("Optimal Path is " + str(track_path))
        return track_path,int(shortest_distance[goal])
##############################################################################
def aruco_pos(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

    parameters = aruco.DetectorParameters_create()

    corners, ids, rej = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # corners = np.squeeze(corners)
    ids = np.squeeze(ids)

    # test_img = aruco.drawDetectedMarkers(img, corners)
    # print('ID = ', ids)
    # print(corners)
    # cv2.imshow('MARKERS', img)
    # cv2.waitKey(0)
    v1= np.zeros(2, int)
    v2 = np.zeros(2, int)
    v3 = np.zeros(2, int)
    v4 = np.zeros(2, int)
    centre = np.zeros(2,int)
    if ids == None:
        print('ID = ', ids)
    if ids!= None:
        # v1,v2,v3,v4 = np.zeros(2,int)
        v1 = np.subtract(corners[0][0][0], corners[0][0][1])    #LEFT
        v2 = np.subtract(corners[0][0][1], corners[0][0][2])    #BACKWARD
        v3 = np.subtract(corners[0][0][2], corners[0][0][3])    #RIGHT
        v4 = np.subtract(corners[0][0][3], corners[0][0][0])    #FORWARD
        centre = np.array([(corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])//4,
                           (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])//4],int)

    v1 = v1.astype(int)
    v2 = v2.astype(int)
    v3 = v3.astype(int)
    v4 = v4.astype(int)
    return v1,v2,v3,v4,centre
############################
def cur_pos(pos,arena_dimensions,roi):
    y_coordinate = -1
    x_coordinate = -1
    width = roi.shape[0] // arena_dimensions[0]
    col = roi.shape[1] // arena_dimensions[1]
    centre_x = pos[0]
    centre_y = pos[1]

    y_coordinate = centre_y // col
    x_coordinate = centre_x //width

    # for i in range(arena_dimensions[0]):
    #     if (i * col <= centre_y and (i + 1) * col > centre_y):
    #         print("hey",i * col,centre_y,(i + 1) * col)
    #         y_coordinate = i
    # for i in range(arena_dimensions[1]):
    #     if (i * width <= centre_x and (i + 1) * width > centre_x):
    #         x_coordinate = i
    # print(pos)
    position = np.array([y_coordinate,x_coordinate])
    return position
#############################
'''
def move(path_list,posinit,v0,ar_dim):
    visited = 0
    total_tiles = len(path_list)
    pos = posinit
    count =0
    v2=v0
    ##switch v2 cuz then it will give values in coordinates of pixels
    temp = v2[0]
    v2[0] = v2[1]
    v2[1] = temp

    posi = 'd' + str(999999) + str(999999)
    while posi != path_list[-1]:
        count+=1
        p.stepSimulation()
        env.move_husky(0, 0, 0, 0)
        if math.remainder(count, 100) == 0 or count in range(5):
            # img = env.camera_feed()
            # w = img.shape[0]
            # h = img.shape[1]
            # roi = img[x - 35:w - x + 35, y - 35:h - y + 35]
            # v1, v2, v3, v4, pos = aruco_pos(roi)
            #
            # ##switch v2 cuz then it will give values in coordinates of pixels
            # temp = v2[0]
            # v2[0]=v2[1]
            # v2[1]=temp
            #
            # print(v1, v2, v3, v4, pos)
            #
            #
            # yx = cur_pos(pos, ar_dim, roi)
            # posi = 'd' + str(yx[0]) + str(yx[1])
            # print("yaha dekhle", posi)

        if posi != path_list[-2]:
            # target = path_list[visited]
            # move_vector = np.array([(int(target[1]) - int(posi[1])), (int(target[2]) - int(posi[2]))],int)
            # print(target,posi,move_vector)
            # ####direction _aligned?
            # comparison = np.sign(move_vector) == np.sign(v2)
            # print("ab idhar", move_vector,np.sign(v2))
            if comparison.all():
                env.move_husky(0.2, 0.2, 0.2, 0.2)
            ####if not aligned then take cross product
            else:
                count2=0
                cross_pd = move_vector[1]*v2[0]- move_vector[0]*v2[1]
                print(cross_pd)
                if (cross_pd > 0):
                    while (move_vector[0] != np.sign(v2[0]) or move_vector[1] != np.sign(v2[1])):
                        p.stepSimulation()
                        count2 += 1
                        if math.remainder(count2, 100) == 0 or count2 in range(100):
                            img = env.camera_feed()
                            w = img.shape[0]
                            h = img.shape[1]
                            roi = img[x - 35:w - x + 35, y - 35:h - y + 35]
                            v1, v2, v3, v4, pos = aruco_pos(roi)

                            ##switch v2 cuz then it will give values in coordinates of pixels
                            temp = v2[0]
                            v2[0] = v2[1]
                            v2[1] = temp

                        env.move_husky(-0.2, 0.2, -0.15, 0.15)
                if(cross_pd<0):
                    while(move_vector[0] != np.sign(v2[0]) or move_vector[1] != np.sign(v2[1])):
                        p.stepSimulation()
                        count2+=1
                        if math.remainder(count2, 100) == 0 or count2 in range(100):
                            img = env.camera_feed()
                            w = img.shape[0]
                            h = img.shape[1]
                            roi = img[x - 35:w - x + 35, y - 35:h - y + 35]
                            v1, v2, v3, v4, pos = aruco_pos(roi)

                            ##switch v2 cuz then it will give values in coordinates of pixels
                            temp = v2[0]
                            v2[0] = v2[1]
                            v2[1] = temp

                        env.move_husky(0.2, -0.2, 0.15, -0.15)
        if(posi == target and posi != path_list[-2]):
            visited+=1
        if(posi != path_list[-2]):
            bool = env.remove_cover_plate(int(path_list[-1][2]),int(path_list[-1][1]))
            pass
'''
###################################
def isInside(circle_x, circle_y, rad, x, y):
    # Compare radius of circle
    # with distance of its center
    # from given point
    # print(math.sqrt(abs((x - circle_x) * (x - circle_x) + (y - circle_y) * (y - circle_y) )))#- rad * rad
    if ((x - circle_x) * (x - circle_x) +(y - circle_y) * (y - circle_y) <= rad * rad):
        return True
    else:
        return False

def move2(path_list,pos,v0,tiles_centre,ar_dim,tile_length,ar_info,sh_info,):
    logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS,
                                "D:\Pixelate_Main_Arena-master\pythonproject\\timings2.json")  #####################

    visited = 0
    count = 0
    v2 = v0
    ##switch v2 cuz then it will give values in coordinates of pixels
    temp = v2[0]
    v2[0] = v2[1]
    v2[1] = temp
    yx = pos
    posi = 'dnn'
    while posi != path_list[-1]:
        print("LAST TILE,tiles visited:",posi,visited)
        count+=1
        p.stepSimulation()
        env.move_husky(0, 0, 0, 0)
        #aruco location at every 100 timesteps
        # if count % 1000 == 0 or count in range(5):
        img = env.camera_feed()
        v1, v2, v3, v4, pos = aruco_pos(img)
        for i in range(2):
            pos[i] -= 40  # x,y---acc to roi2#40 cuz, img processing has been done on it
        add_steay_pos_error(pos, img)
        temp = v2[0]
        v2[0] = v2[1]
        v2[1] = temp
        # w = img.shape[0]
        # h = img.shape[1]
        # x=40
        # y=40
        # roi = img[x - 35:w - x + 35, y - 35:h - y + 35]
        # roi2 = img[x:w - x, y:h - y]

        # ##switch v2 cuz then it will give values in coordinates of pixels

        # print(v1, v2, v3, v4, pos)
        #BISMARCK  # 7711
        # posi = cur_pos(targetpos, ar_dim, roi2)
        if (posi == 'dnn'):
            yx = cur_pos(pos, ar_dim, img)
            posi = coord_con_str(yx)
        #BEFORE 2ND LAST TILE
        if posi != path_list[-2]:
            target = path_list[visited]
            print(target)
            targetyx = coord_con_int(target)                         #yx
            targetpos = tiles_centre[targetyx[0]][targetyx[1]]       #yx
            ####direction _aligned?
            # print("wah",target,targetpos,pos)
            move_vector = np.array([(int(targetpos[0])-int(pos[1])),(int(targetpos[1])-int(pos[0]))],int) #yx
            cross_pd = v2[1]*move_vector[0]-v2[0]*move_vector[1]
            dot_pd = v2[1] * move_vector[1] + v2[0] * move_vector[0]
            print("cross pd(ML): ",cross_pd)
            if (cross_pd <=50 and cross_pd >=-50):
                count+=1
                if dot_pd<0:
                    while (count % 100 != 1):
                        count += 1
                        env.move_husky(-0.2, -0.2, -0.2, -0.2)
                        p.stepSimulation()
                else:
                    while (count % 1000 != 1):
                        count += 1
                        if count2 % 1000 == 0:
                            print("Jai mahakal69",count)
                        env.move_husky(0.2, 0.2, 0.2, 0.2)
                        p.stepSimulation()
                    # pass
            else:
                count2 = 0
                if (cross_pd<-50):

                    while(cross_pd<-50):

                        p.stepSimulation()
                        count2 += 1
                        if count2 % 1000 == 0 or count2 ==1:
                            img = env.camera_feed()
                            v1, v2, v3, v4, pos = aruco_pos(img)
                            for i in range(2):
                                pos[i] -= 40  # x,y---acc to roi2#40 cuz, img processing has been done on it
                            add_steay_pos_error(pos, img)

                            ##switch v2 cuz then it will give values in coordinates of pixels
                            temp = v2[0]
                            v2[0] = v2[1]
                            v2[1] = temp
                            move_vector = np.array([(int(targetpos[0]) - int(pos[1])), (int(targetpos[1]) - int(pos[0]))],
                                               int)
                            cross_pd = v2[1]*move_vector[0]-v2[0]*move_vector[1]
                            print("Takeshi Castle1", cross_pd,count2)
                        if (cross_pd > -200):
                            env.move_husky(-0.0375, 0.05, -0.0375, 0.05)
                        else:
                            env.move_husky(-0.15, 0.2, -0.15, 0.2)
                if (cross_pd > 50):
                    print("namaste2")
                    while (cross_pd > 50):
                        p.stepSimulation()
                        count2 += 1
                        if count2 % 1000 == 0 or count2 ==1:
                            img = env.camera_feed()
                            v1, v2, v3, v4, pos = aruco_pos(img)
                            for i in range(2):
                                pos[i] -= 40  # x,y---acc to roi2#40 cuz, img processing has been done on it
                            add_steay_pos_error(pos, img)

                            ##switch v2 cuz then it will give values in coordinates of pixels
                            temp = v2[0]
                            v2[0] = v2[1]
                            v2[1] = temp
                            move_vector = np.array([(int(targetpos[0]) - int(pos[1])), (int(targetpos[1]) - int(pos[0]))],
                                               int)
                            cross_pd = v2[1] * move_vector[0] - v2[0] * move_vector[1]
                            print("Takeshi Castle2", cross_pd, count2)
                        if (cross_pd < 200):
                            env.move_husky(0.05, -0.0375, 0.05, -0.0375)
                        else:
                            env.move_husky(0.2, -0.15, 0.2, -0.15)


        #################
        next_node_reached = isInside(tiles_centre[targetyx[0]][targetyx[1]][1], tiles_centre[targetyx[0]][targetyx[1]][0]
                                     , tile_length / 6, pos[0], pos[1])
        print("Target Pos, Aruco w/error pos: ",tiles_centre[targetyx[0]][targetyx[1]],pos)
        print(tile_length)
        #################

        if (next_node_reached == 1):
            print("next Node", next_node_reached,posi,path_list[-2])
            # time.sleep(2)
            posi = coord_con_str(targetyx)#'d' + str(targetyx[1]) + str(targetyx[0])
            visited += 1
        if (posi == path_list[-2]):
            print("cute Husky ", count)
            env.move_husky(0,0,0,0)
            p.stepSimulation()

            plate_y,plate_x  = coord_con_int(path_list[-1])
            bool = env.remove_cover_plate(plate_y, plate_x)
            time.sleep(1)
            for r in range(100):
                p.stepSimulation()
            print("karye pragati pe hai")
            time.sleep(1)
            visited = len(path_list)-1
            img = env.camera_feed()
            ####################
            target = path_list[-1]
            # target = path_list[visited]
            # targetpos = tiles_centre[int(target[1])][int(target[2])]
            # targetyx = cur_pos(targetpos,ar_dim,roi2)
            targetyx = coord_con_int(target)  # yx
            targetpos = tiles_centre[targetyx[0]][targetyx[1]]  # yx
            # roi3 = img[tiles_centre[targetyx[1]][targetyx[0]][0]-(tile_length // 2)-1:tiles_centre[targetyx[1]][targetyx[0]][0]-(tile_length // 2)+1,
            #        tiles_centre[targetyx[1]][targetyx[0]][1]-(tile_length // 2)-1:tiles_centre[targetyx[1]][targetyx[0]][1]-(tile_length // 2)+1]
            xc, yc = 40, 40
            w = img.shape[0]
            h = img.shape[1]
            roi3 = img[xc:w - xc, yc:h - yc]

            # roi = roi3[y + 1:y + box_length, x + 1:x + box_length]
            # arena_info_temp, _,_,_ = image_processing1(roi3, ar_dim)
            #
            # arena_info[targetyx[0]][targetyx[1]][1] =arena_info_temp[targetyx[0]][targetyx[1]][1]
            #######
            # shape_detection(roi, 120, 255, 227)
            temp_pos = coord_con_int(target)
            # print(tiles_centre)
            print(temp_pos,tiles_centre[targetyx[0]][targetyx[1]])
            til_recg_x = tiles_centre[targetyx[0]][targetyx[1]][0] - tile_length//2
            til_recg_y = tiles_centre[targetyx[0]][targetyx[1]][1] - tile_length//2
            print(til_recg_x,til_recg_y)
            roi4 = roi3[til_recg_x:til_recg_x+tile_length, til_recg_y:til_recg_y+tile_length]
            roi4 = cv2.resize(roi4, (250, 250))
            # cv2.imshow("roi4",roi4)
            shape_pat,_ = shape_detection(roi4, 120, 255, 227)
            arena_info[targetyx[0]][targetyx[1]][1] =shape_pat
            ########
            # print(arena_info_temp)
            print("HERE IS THE FREAKIN SHAPE:",shape_pat)
            # cv2.imshow('n', roi3)
            cv2.waitKey(0)
            time.sleep(0)
            # cv2.imshow('roi3',roi3)
            # cv2.waitKey(0)
            # y_coordinate=targetyx[0]
            # x_coordinate=targetyx[1]
            # ar_info[y_coordinate][x_coordinate][1], sh_info[y_coordinate][
            #     x_coordinate] = shape_detection(roi3, 120, 255, 227)

            ####################
            #NOW MOVE HAPPENS HERE from 2nd last to last tile
            ####################
            target = path_list[-1]
            targetyx = coord_con_int(target)  # yx
            targetpos = tiles_centre[targetyx[0]][targetyx[1]]  # yx
            count=0
            ###
            while posi != path_list[-1]:
                print("target: ",target)
                print(tiles_centre[targetyx[0]][targetyx[1]],pos)
                # targetpos = tiles_centre[int(target[1])][int(target[2])]
                # targetyx = cur_pos(targetpos,ar_dim,roi2)
                # targetpos = tiles_centre[y][x]
                # targetyx = cur_pos(targetpos, ar_dim, roi2)
                count += 1
                p.stepSimulation()
                env.move_husky(0, 0, 0, 0)
                # if count % 100 == 0 or count == 1:
                    # img = env.camera_feed()
                    # w = img.shape[0]
                    # h = img.shape[1]
                    # roi = img[x - 35:w - x + 35, y - 35:h - y + 35]
                    # roi2 = img[x:w - x, y:h - y]
                    # v1, v2, v3, v4, pos = aruco_pos(roi)
                    # for i in range(2):
                    #     pos[i] -= 35
                    # add_steay_pos_error(pos, roi2)
                    # # ##switch v2 cuz then it will give values in coordinates of pixels
                    # temp = v2[0]
                    # v2[0] = v2[1]
                    # v2[1] = temp
                img = env.camera_feed()
                v1, v2, v3, v4, pos = aruco_pos(img)
                for i in range(2):
                    pos[i] -= 40  # x,y---acc to roi2#40 cuz, img processing has been done on it
                add_steay_pos_error(pos, img)
                temp = v2[0]
                v2[0] = v2[1]
                v2[1] = temp
                # yx = cur_pos(pos, ar_dim, roi2)
                # posi = coord_con_str(yx)
                move_vector = np.array([(int(targetpos[0]) - int(pos[1])), (int(targetpos[1]) - int(pos[0]))], int)
                cross_pd = v2[1] * move_vector[0] - v2[0] * move_vector[1]
                dot_pd = v2[1] * move_vector[1] + v2[0] * move_vector[0]
                if (cross_pd <= 50 and cross_pd >= -50):
                    count += 1
                    if dot_pd < 0:
                        while (count % 100 != 1):
                            count += 1
                            print("back lene de", count)
                            env.move_husky(-0.2, -0.2, -0.2, -0.2)
                            p.stepSimulation()
                    else:
                        while (count % 1000 != 1):
                            count += 1
                            print("Jai mahakal", count)
                            env.move_husky(0.2, 0.2, 0.2, 0.2)
                            p.stepSimulation()
                    # pass
                    next_node_reached = isInside(tiles_centre[targetyx[0]][targetyx[1]][1],
                                                 tiles_centre[targetyx[0]][targetyx[1]][0]
                                                 , tile_length / 6, pos[0], pos[1])
                    if (next_node_reached == 1):
                        print("April's fool")
                        posi = coord_con_str(targetyx)
                else:
                    # roi = img[x - 35:w - x + 35, y - 35:h - y + 35]
                    # roi2 = img[x:w - x, y:h - y]
                    # v1, v2, v3, v4, pos = aruco_pos(roi)
                    #
                    # for i in range(2):
                    #     pos[i] -= 35
                    # add_steay_pos_error(pos, roi2)

                    img = env.camera_feed()
                    v1, v2, v3, v4, pos = aruco_pos(img)
                    for i in range(2):
                        pos[i] -= 40  # x,y---acc to roi2#40 cuz, img processing has been done on it
                    add_steay_pos_error(pos, img)
                    temp = v2[0]
                    v2[0] = v2[1]
                    v2[1] = temp
                    next_node_reached = isInside(tiles_centre[targetyx[0]][targetyx[1]][1],
                                                 tiles_centre[targetyx[0]][targetyx[1]][0]
                                                 , tile_length / 6, pos[0], pos[1])
                    if (next_node_reached == 1):
                        posi = coord_con_str(targetyx)


                    if posi != path_list[-1]:

                        count2 = 0
                        # cross_pd = move_vector[1] * v2[0] - move_vector[0] * v2[1]
                        # print(cross_pd)
                        if (cross_pd < -50):

                            while (cross_pd < -50):

                                p.stepSimulation()
                                count2 += 1
                                if count2 % 500 == 0 or count2 ==1:
                                    # precision = 1 + (count2) * 0.00003
                                    # img = env.camera_feed()
                                    # x =40
                                    # y =40
                                    # w = img.shape[0]
                                    # h = img.shape[1]
                                    # roi = img[x - 35:w - x + 35, y - 35:h - y + 35]
                                    # roi2 = img[x:w - x, y:h - y]
                                    # v1, v2, v3, v4, pos = aruco_pos(roi)
                                    # for i in range(2):
                                    #     pos[i] -= 35
                                    # add_steay_pos_error(pos, roi2)
                                    # # _, _, _, _, pos = aruco_pos(roi2)

                                    img = env.camera_feed()
                                    v1, v2, v3, v4, pos = aruco_pos(img)
                                    for i in range(2):
                                        pos[i] -= 40  # x,y---acc to roi2#40 cuz, img processing has been done on it
                                    add_steay_pos_error(pos, img)

                                    ##switch v2 cuz then it will give values in coordinates of pixels
                                    temp = v2[0]
                                    v2[0] = v2[1]
                                    v2[1] = temp
                                # if (cross_pd < 300):
                                #     if count2 % 50 == 0:
                                #         img = env.camera_feed()
                                #         v1, v2, v3, v4, pos = aruco_pos(img)
                                #         for i in range(2):
                                #             pos[i] -= 40  # x,y---acc to roi2#40 cuz, img processing has been done on it
                                #         add_steay_pos_error(pos, img)

                                    # ##switch v2 cuz then it will give values in coordinates of pixels
                                    # temp = v2[0]
                                    # v2[0] = v2[1]
                                    # v2[1] = temp
                                move_vector = np.array(
                                    [(int(targetpos[0]) - int(pos[1])), (int(targetpos[1]) - int(pos[0]))],
                                    int)
                                # per_vector_robot = np.array([-v2[1], v2[0]], int)
                                # env.move_husky(-0.05, 0.0375, -0.05, 0.0375)#maybe this was causing error
                                # env.move_husky(-0.15, 0.2, -0.15, 0.2)
                                ############################################################################################
                                env.move_husky(-0.15, 0.2, -0.15, 0.2)
                                if (cross_pd > -100):
                                    env.move_husky(-0.05, 0.0375, -0.05, 0.0375)
                                ############################################################################################
                                cross_pd = v2[1] * move_vector[0] - v2[0] * move_vector[1]
                                print("Takeshi Castle1", cross_pd,count2)
                                print(tiles_centre[targetyx[0]][targetyx[1]], pos)
                        if (cross_pd > 50):
                            print("namaste2")
                            while (cross_pd > 50):
                                p.stepSimulation()
                                count2 += 1
                                if count2 % 500 == 0 or count2 ==1:
                                    # precision = 1 + (count2)*0.00003
                                    # img = env.camera_feed()
                                    # x = 40
                                    # y = 40
                                    # w = img.shape[0]
                                    # h = img.shape[1]
                                    # roi = img[x - 35:w - x + 35, y - 35:h - y + 35]
                                    # roi2 = img[x:w - x, y:h - y]
                                    # v1, v2, v3, v4, pos = aruco_pos(roi)
                                    # for i in range(2):
                                    #     pos[i] -= 35
                                    # add_steay_pos_error(pos, roi2)
                                    # # _, _, _, _, pos = aruco_pos(roi2)

                                    img = env.camera_feed()
                                    v1, v2, v3, v4, pos = aruco_pos(img)
                                    for i in range(2):
                                        pos[i] -= 40  # x,y---acc to roi2#40 cuz, img processing has been done on it
                                    add_steay_pos_error(pos, img)

                                    ##switch v2 cuz then it will give values in coordinates of pixels
                                    temp = v2[0]
                                    v2[0] = v2[1]
                                    v2[1] = temp
                                # if (cross_pd < 300):
                                #     if count2 % 50 == 0:
                                #         img = env.camera_feed()
                                #         v1, v2, v3, v4, pos = aruco_pos(img)
                                #         for i in range(2):
                                #             pos[i] -= 40  # x,y---acc to roi2#40 cuz, img processing has been done on it
                                #         add_steay_pos_error(pos, img)

                                    # ##switch v2 cuz then it will give values in coordinates of pixels
                                    # temp = v2[0]
                                    # v2[0] = v2[1]
                                    # v2[1] = temp


                                move_vector = np.array(
                                    [(int(targetpos[0]) - int(pos[1])), (int(targetpos[1]) - int(pos[0]))],
                                    int)
                                # per_vector_robot = np.array([-v2[1], v2[0]], int)
                                # env.move_husky(0.05, -0.0375, 0.05, -0.0375)
                                ############################################################################################
                                env.move_husky(0.2, -0.15, 0.2, -0.15)
                                if(cross_pd<100):
                                    env.move_husky(0.05, -0.0375, 0.05, -0.0375)
                                ############################################################################################
                                cross_pd = v2[1] * move_vector[0] - v2[0] * move_vector[1]
                                print("wah", target, targetpos, pos)
                                print("bete", move_vector, v2)
                                print("Takeshi Castle2", cross_pd)
                                print(tiles_centre[targetyx[0]][targetyx[1]], pos)






            # visited+=1
        if (posi == path_list[-1]):
            print("MOVE2 FUNCTION COMPLETED, DESTINATION REACHED")
            return path_list[-1]
        p.stopStateLogging(logId)

####################################
def hosp(arena_info, posi):
    y_max = arena_info.shape[0]
    x_max = arena_info.shape[1]

    # y = int(posi[1])
    # x = int(posi[2])
    # y=0
    # x=0
    y,x =coord_con_int(posi)
    color_code = arena_info[y][x][0]
    shape_code = arena_info[y][x][1]
    print("waka waka",color_code,shape_code)

    if color_code == 6:#pink
        for j in range(y_max):
            for i in range(x_max):
                if arena_info[j][i][0] == 4:#cyan
                    if arena_info[j][i][1] == shape_code:
                        yx = [j,i]
                        posi_return = coord_con_str(yx)
                        return posi_return
    # if color_code == 4:  # cyan
    #     pass
###################################
def coord_con_int(posi):

    y = ((int(posi[1]))*10)+(int(posi[2]))
    x = ((int(posi[3]))*10) + (int(posi[4]))
    return y,x

def coord_con_str(yx):
    x = yx[0]##actually y
    y = yx[1]##actually x
    if (len(str(x)) < 2 and len(str(y)) < 2):
        posi = ('d'+'0' + str(x) + '0' + str(y))
    elif (len(str(x)) < 2 and len(str(y)) == 2):
        posi = ('d'+'0' + str(x) + str(y))
    elif (len(str(x)) == 2 and len(str(y)) < 2):
        posi = ('d'+str(x) + '0' + str(y))
    elif (len(str(x)) == 2 and len(str(y)) == 2):
        posi = ('d'+str(x) + str(y))
    return posi #(y,x)

def add_steay_pos_error(pos,roi):
    #add_steay_pos_error(pos,roi2)
    w = roi.shape[0]
    h = roi.shape[1]
    pos[0] -= int(((((w / 12) / 2) - 1) * (pos[0] - w / 2)) * 0.0013)
    pos[1] -= int(((((h / 12) / 2) - 1) * (pos[1] - w / 2)) * 0.0013)

##################################
###################################
##################################
# parent_path = os.path.dirname(os.getcwd())
# os.chdir(parent_path)
# env = gym.make("pix_main_arena-v0")
# ar_dim = np.array([6,6])

weights = np.zeros(8,int)
'''
    COLOR = COLOR CODE, WEIGHT
    WHITE  = 0, 1
    RED    = 1, 4
    YELLOW = 2, 3
    GREEN  = 3, 2
    CYAN   = 4, 1?
    BLUE   = 5, no need but still say 1
    PINK   = 6, 1?
    DULL GREEN = 7, 1
'''
weights[0] = 1
weights[1] = 4
weights[2] = 3
weights[3] = 2
weights[4] = 1#asdf
weights[5] = 1#aise hee,blue ke liye wieghts is meaningless
weights[6] = 1
weights[7] = 1

#####################################################
#####################################################

#####################################################
#####################################################

#####################################################
#####################################################

#####################################################
#####################################################

#####################################################
#####################################################
zero_img = np.array(0,int)
i=0
vr = -0.5
v1 = np.zeros(2,int)
v2 = np.zeros(2,int)
v3 = np.zeros(2,int)
v4 = np.zeros(2,int)
pos = np.zeros(2,int)
posyx = np.zeros(2,int)
###########################
###########################
#AH SHIT, HERE WE GO AGAIN

parent_path = os.path.dirname(os.getcwd())
os.chdir(parent_path)
env = gym.make("pix_main_arena-v0")
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
# logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "D:\Pixelate_Main_Arena-master\pythonproject\\timings.json")#####################
# out_file = open(r"D:\Pixelate_Main_Arena-master\pythonproject\logs.json", "w")#########################################################################
ar_dim = np.array([12,12])

weights = np.zeros(8,int)
'''
    COLOR = COLOR CODE, WEIGHT
    WHITE  = 0, 1
    RED    = 1, 4
    YELLOW = 2, 3
    GREEN  = 3, 2
    CYAN   = 4, 1?
    BLUE   = 5, no need but still say 1
    PINK   = 6, 1?
    DULL GREEN = 7, 1
'''
weights[0] = 1
weights[1] = 4
weights[2] = 3
weights[3] = 2
weights[4] = 1#asdf
weights[5] = 1#aise hee,blue ke liye wieghts is meaningless
weights[6] = 1
weights[7] = 1

zero_img = np.array(0,int)
i=0
vr = -0.5
v1 = np.zeros(2,int)
v2 = np.zeros(2,int)
v3 = np.zeros(2,int)
v4 = np.zeros(2,int)
pos = np.zeros(2,int)
posyx = np.zeros(2,int)

##########

p.stepSimulation()
env.remove_car()
# bool = env.remove_cover_plate(5,11)
# p.stepSimulation()
img = env.camera_feed()
x, y = 40,40
w = img.shape[0]
h = img.shape[1]
roi = img[x:w - x, y:h - y]
# roi = img[x - 35:w - x + 35, y - 35:h - y + 35]
# roi2 = img[x:w - x, y:h - y]
v1, v2, v3, v4,pos = aruco_pos(roi)
add_steay_pos_error(pos,roi)

# cv2.imshow('as',roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
arena_info, tile_length, tile_centres,shape_info = image_processing1(roi, ar_dim)
print(tile_length)
patient = []
for j in range(ar_dim[0]):
    for i in range(ar_dim[1]):
        if arena_info[j][i][0] == 6:
            patient.append([j,i])
print(arena_info)
# print(tile_centres[5][11])
# til_recg_x = tile_centres[5][11][0] - tile_length//2
# til_recg_y = tile_centres[5][11][1] - tile_length//2
# print(til_recg_x,til_recg_y)
# roi4 = roi[til_recg_x:til_recg_x+tile_length, til_recg_y:til_recg_y+tile_length]
# roi4 = cv2.resize(roi4, (250, 250))
# cv2.imshow("roi4",roi4)
# cv2.imshow('as',roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(patient,coord_con_str(patient[0]))
graphi1 = graph_dict(arena_info, ar_dim, weights, shape_info,'d1111',coord_con_str(patient[0]))
graphi2 = graph_dict(arena_info, ar_dim, weights, shape_info,'d1111',coord_con_str(patient[1]))
print(graphi1)
print(graphi2)

pathi1, leni1 = path_dijkstra(graphi1, 'd1111', coord_con_str(patient[0]))
pathi2, leni2 = path_dijkstra(graphi2, 'd1111', coord_con_str(patient[1]))

print(leni1,leni2)

env.respawn_car()
time.sleep(1)

if(leni1>=leni2):#movement will start now
    last_pos = move2(pathi2,pos,v2,tile_centres,ar_dim,tile_length,arena_info,shape_info)
    # p.stopStateLogging(logId)

    # json.dump(logId, out_file, indent=6)
    #
    # out_file.close()
    print(arena_info)
    next_pos = hosp(arena_info, last_pos)
    # print(last_pos,next_pos)
    #1st Hospital
    graph_hos_1 = graph_dict(arena_info, ar_dim, weights, shape_info, last_pos, next_pos)
    path3, _ = path_dijkstra(graph_hos_1, last_pos, next_pos)
    last_pos2 = move2(path3, last_pos, v2, tile_centres, ar_dim, tile_length, arena_info, shape_info)
    #2nd patient
    graphi_pat_2 = graph_dict(arena_info, ar_dim, weights, shape_info, last_pos2, coord_con_str(patient[0]))
    path_pat_2, leni2 = path_dijkstra(graphi_pat_2, last_pos2, coord_con_str(patient[0]))
    last_pos3 = move2(path_pat_2, last_pos2, v2, tile_centres, ar_dim, tile_length, arena_info, shape_info)
    #2ndHospital
    next_pos3 = hosp(arena_info, last_pos3)
    graph_hos_2 = graph_dict(arena_info, ar_dim, weights, shape_info, last_pos3, next_pos3)
    path4, _ = path_dijkstra(graph_hos_2, last_pos3, next_pos3)
    last_pos4 = move2(path4, last_pos3, v2, tile_centres, ar_dim, tile_length, arena_info, shape_info)
    print("Project is dead, we have work to do")

if(leni1<leni2):
    last_pos = move2(pathi1,pos,v2,tile_centres,ar_dim,tile_length,arena_info,shape_info)
    next_pos = hosp(arena_info, last_pos)
    # p.stopStateLogging(logId)


    # json.dump(logId, out_file, indent=6)
    #
    # out_file.close()

    # print(arena_info)
    # print('d00: ',next_pos)
    # 1st Hospital
    graph_hos_1 = graph_dict(arena_info, ar_dim, weights, shape_info, last_pos, next_pos)
    path3, _ = path_dijkstra(graph_hos_1, last_pos, next_pos)
    last_pos2 = move2(path3, last_pos, v2, tile_centres, ar_dim, tile_length, arena_info, shape_info)
    #2nd patient
    graphi_pat_2 = graph_dict(arena_info, ar_dim, weights, shape_info, last_pos2, coord_con_str(patient[1]))
    path_pat_2, leni2 = path_dijkstra(graphi_pat_2, last_pos2, coord_con_str(patient[1]))
    last_pos3 = move2(path_pat_2, last_pos2, v2, tile_centres, ar_dim, tile_length, arena_info, shape_info)
    #2ndHospital
    next_pos3 = hosp(arena_info, last_pos3)
    graph_hos_2 = graph_dict(arena_info, ar_dim, weights, shape_info, last_pos3, next_pos3)
    path4, _ = path_dijkstra(graph_hos_2, last_pos3, next_pos3)
    last_pos4 = move2(path4, last_pos3, v2, tile_centres, ar_dim, tile_length, arena_info, shape_info)
    print("Project is dead, we have work to do")









time.sleep(20)










#############################
#############################
# while True:
#     i+=1
#     p.stepSimulation()
#     # env.move_husky(0.2, 0.022, 0.15, 0.015)
#     # env.move_husky(0.2,-0.2,0.15,-0.15)
#     env.move_husky(0,0,0,0)
#     # env.move_husky(v_front, v_front, v_back, v_back)
#     if i == 1 :
#         env.remove_car()
#         img = env.camera_feed()
#         x, y = 93, 93
#         w = img.shape[0]
#         h = img.shape[1]
#         roi = img[x:w - x, y:h - y]
#         # env.remove_cover_plate(0, 0)
#         arena_info, tile_length, tile_centres,shape_info = image_processing1(roi, ar_dim)
#         cv2.destroyAllWindows()
#         print(arena_info)
#         dict_graph = graph_dict(arena_info, ar_dim, weights, shape_info,'d55','d00')
#         print(dict_graph)
#         s_path, s_len = path_dijkstra(dict_graph, 'd55', 'd00')
#         # dict_graph = graph_dict(arena_info, ar_dim, weights, shape_info,'d05','d00')
#         # s_path2, s_len2 = path_dijkstra(dict_graph, 'd00', 'd05')
#         roi = img[x - 35:w - x + 35, y - 35:h - y + 35]
#         zero_img = roi.copy()
#         # cv2.waitKey(1)
#         cv2.destroyAllWindows()
#         env.respawn_car()
#         # time.sleep()
#
#
#         img = env.camera_feed()
#         w = img.shape[0]
#         h = img.shape[1]
#         roi = img[x:w - x, y:h - y]
#         v1, v2, v3, v4,pos = aruco_pos(roi)
#         posyx = cur_pos(pos, ar_dim, zero_img)
#         # print(v1, v2, v3, v4,pos )
#
#     # print(dict)
    # if i<20000:

#     if math.remainder(i, 100) == 0:
#         img = env.camera_feed()
#         w = img.shape[0]
#         h = img.shape[1]
#         roi = img[x-35:w - x+35, y-35:h - y+35]
#         v1, v2, v3, v4, pos = aruco_pos(roi)
#         # print(v1,v2,v3,v4, pos)
#     # move(s_path,pos,v2,ar_dim)
#     last_pos = move2(s_path, pos, v2, tile_centres, ar_dim, tile_length,arena_info,shape_info)
#     env.remove_cover_plate(0, 0)
#     # arena_info, tile_length, tile_centres, shape_info = image_processing1(roi, ar_dim)
#     next_pos = hosp(arena_info, last_pos)
#     print(arena_info)
#     print('d00: ',next_pos)
#     dict_graph = graph_dict(arena_info, ar_dim, weights, shape_info, last_pos, next_pos)
#     s_path2, s_len2 = path_dijkstra(dict_graph, last_pos, next_pos)
#     last_pos2 = move2(s_path2, pos, v2, tile_centres, ar_dim, tile_length, arena_info, shape_info)
#
#
#
#     # print(dict_graph)
#     # s_path, s_len = path_dijkstra(dict_graph, 'd55', 'd05')
#     # move2(s_path2, pos, v2, tile_centres, ar_dim, tile_length,arena_info,shape_info)
#     # break
#         # if pos[0]-tile_centres[5][0][1]>0:
#         #     env.move_husky(0.2, 0.2, 0.2, 0.2)
#         # env.move_husky(0.2, 0.019, 0.2,0.019)
#     # env.move_husky(0.2, -0.2, 0.15, -0.15)
#     # if v4[0]>0:
#     #     # print('rotate',pos,cur_pos(pos,ar_dim,zero_img))
#     #     env.move_husky(0.2, -0.18, 0.15, -0.15)
#     # else:
#     #     # print('forward',cur_pos(pos,ar_dim,zero_img))
#     #     env.move_husky(0.2, 0.2, 0.2, 0.2)
#
#     # print(int(s_path[0][1])+int(s_path[0][2]))
#     # print(s_path[-2])
#
#         # env.move_husky(v_front, v_front, v_back, v_back)
# #     cv2.imshow("img", img)
# #     cv2.waitKey(1)
# cv2.destroyAllWindows()
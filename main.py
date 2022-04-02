import sys
from cmath import sqrt

from numpy import real
from skimage.feature import (graycomatrix, graycoprops)
import cv2

max_contrast = -sys.maxsize - 1
max_correlation = -sys.maxsize - 1
max_energy = -sys.maxsize - 1
max_homogeneity = -sys.maxsize - 1
max_value = -sys.maxsize - 1

min_contrast = sys.maxsize
min_correlation = sys.maxsize
min_energy = sys.maxsize
min_homogeneity = sys.maxsize
min_value = sys.maxsize

# def clear_different_material_pixels(material):
#     pre1 = f'FMD/image/{material}/{material}{"_moderate" if (material != "foliage") else "_final"}_'
#     pre2 = f'FMD/mask/{material}/{material}{"_moderate" if (material != "foliage") else "_final"}_'
#     post = '_new.jpg'
#     for i in range(50):
#         num=i+1
#         path1 = pre1 + ("00" if (int(num / 10) == 0) else "0") + str(num) + post
#         path2 = pre2 + ("00" if (int(num / 10) == 0) else "0") + str(num) + post
#         img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
#         img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
#         # print(path1)
#         cv2.imshow('gray scale1', img1)
#         cv2.waitKey(1000)
#         for r in range(len(img2)):
#             for c in range(len(img2[r])):
#                 if(img2[r][c]==0): img1[r][c]=0
#                 print(len(img2[r]))
#         cv2.imshow('gray scale1',img1)
#         cv2.waitKey(1000)

def calcGLCMFeatureVector(material, type):  # type 1 mean train 2 means test
    global max_correlation, max_contrast, max_energy, max_homogeneity, max_value
    global min_correlation, min_contrast, min_energy, min_homogeneity, min_value
    materials_feature_vectors = []
    if (type == 1):
        num = 0
        end = 35
    else:
        num = 35
        end = 50
    pre1 = f'FMD/image/{material}/{material}{"_moderate" if (material != "foliage") else "_final"}_'
    pre2 = f'FMD/mask/{material}/{material}{"_moderate" if (material != "foliage") else "_final"}_'
    post = '_new.jpg'
    while num < end:
        # print(num)
        num = num + 1
        path1 = pre1 + ("00" if (int(num / 10) == 0) else "0") + str(num) + post
        path2 = pre2 + ("00" if (int(num / 10) == 0) else "0") + str(num) + post
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('gray scale1', img1)
        # cv2.waitKey(50)
        for r in range(len(img2)):
            for c in range(len(img2[r])):
                if (img2[r][c] == 0): img1[r][c] = 0
        # cv2.imshow('gray scale1', img1)
        # cv2.waitKey(50)
        # print(path)
        # cv2.imshow('gray scale1',img)
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()
        g = graycomatrix(img1, [1], [0])
        contrast = graycoprops(g, "contrast")
        correlation = graycoprops(g, "correlation")
        energy = graycoprops(g, "energy")
        homogeneity = graycoprops(g, "homogeneity")
        dissimilarity = graycoprops(g, "dissimilarity")
        ASM = graycoprops(g, "ASM")

        # if contrast > max_contrast: max_contrast = real(sqrt(contrast))
        # if correlation > max_correlation: max_correlation = correlation
        # if energy > max_energy: max_energy = energy
        # if homogeneity > max_homogeneity: max_homogeneity = homogeneity
        #
        # if contrast < min_contrast: min_contrast = real(sqrt(contrast))
        # if correlation < min_correlation: min_correlation = correlation
        # if energy < min_energy: min_energy = energy
        # if homogeneity < min_homogeneity: min_homogeneity = homogeneity
        value = float(real(sqrt(contrast)) / 26 + correlation + energy + homogeneity)
        feature_vector = (real(sqrt(contrast)) / 26, correlation, energy, homogeneity, dissimilarity, ASM, value)
        # if value > max_value: max_value = value
        # if value < min_value: min_value = value
        # print(feature_vector)
        # print(value)
        materials_feature_vectors.append(feature_vector)
    return materials_feature_vectors


Materials = ["fabric", "foliage", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood"]


def calcGLCMFeatureVectorForAll(type):
    all_materials_feature_vectors = []
    for material in Materials:
        # print(material)
        all_materials_feature_vectors.append(calcGLCMFeatureVector(material, type))
    return all_materials_feature_vectors


# [numMaterial][numPhoto][numProperty]
def calculateAccuracy(trainData, testData):
    accuracy = 0
    for testMatNum, testMaterial in enumerate(Materials):
        testPhoto = 0
        while testPhoto < 15:
            minDiff = sys.maxsize
            choosenMaterial = ""
            for trainMatNum, trainMaterial in enumerate(Materials):
                train = 0
                while train < 35:
                    # print(testMatNum)
                    # print(test)
                    diff = abs(trainData[trainMatNum][train][6] - testData[testMatNum][testPhoto][6])
                    if diff < minDiff:
                        # if trainMaterial=="glass": print(diff)
                        minDiff = diff
                        choosenMaterial = trainMaterial
                    train += 1
            testPhoto += 1
            if choosenMaterial == testMaterial:
                accuracy += 1
            # else: print(choosenMaterial + "vvv" + testMaterial)
            # print(accuracy)

    accuracy /= 1.5  # (/1.5)==(*100/150)
    acc = str(accuracy)
    # print("Accuracy: " + acc + "%")
    return acc

# clear_different_material_pixels("glass")
trainData = calcGLCMFeatureVectorForAll(1)
testData = calcGLCMFeatureVectorForAll(2)
# print(len(testData[0][0]))
accuracy = calculateAccuracy(trainData, testData)
print("Accuracy: " + accuracy + "%")
# print(max_contrast)
# print(max_correlation)
# print(max_energy)
# print(max_homogeneity)
# print(max_value)
# print()
# print(min_contrast)
# print(min_correlation)
# print(min_energy)
# print(min_homogeneity)
# print(min_value)
# foliage_final_001_new.jpg
# foliage_001_new.jpg

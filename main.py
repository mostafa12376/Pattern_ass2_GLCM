from skimage.feature import (graycomatrix, graycoprops)
import cv2


def calcTrainGLCMFeatureVector(material, type): #type 1 mean train 2 means test
    materials_feature_vectors = []
    if (type==1):
        num= 0
        end = 35
    else:
        num= 35
        end= 50
    while num<end :
        #print(num)
        num = num + 1
        path = f'FMD/image/{material}/{material}_moderate_{("00" if (int(num / 10) == 0) else "0") + str(num)}_new.jpg'
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # print(path)
        # cv2.imshow('gray scale1',img)
        # cv2.waitKey(200)
        # cv2.destroyAllWindows()
        g = graycomatrix(img, [1], [0])
        contrast = graycoprops(g, "contrast")
        correlation = graycoprops(g, "correlation")
        energy = graycoprops(g, "energy")
        homogeneity = graycoprops(g, "homogeneity")
        feature_vector = (contrast, correlation, energy, homogeneity)
        materials_feature_vectors.append(feature_vector)
    return materials_feature_vectors


Materials = {"fabric", "foliage", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood"}


def calcTrainGLCMFeatureVectorForAll(type):
    all_materials_feature_vectors = []
    for material in Materials:
        all_materials_feature_vectors.append(calcTrainGLCMFeatureVector(material, type))
    return all_materials_feature_vectors

import math
import matplotlib.pyplot as plt
import numpy as np
import pathlib, os, shutil
import random
import requests
import warnings

from zipfile import ZipFile
from PIL import Image

###
from google_images_download import google_images_download
from csv import reader


######################################
# Manning Art Classifier LiveProject Review
# David G. Trimm
# 4/9/21


###########

# Gloibal Variables

pathToDataset  = None
pathToDownload = None
pathToTrain = None
pathToTest = None
pathToValid = None
pathToModels = None

def setup():
    warnings.filterwarnings('ignore')


###
# Sets the root folder for image sets and creates directories
# if they don't exist
#
# dataset
#   downloads
#   train
#   test
#   valid
#   models

def prepDirectories():
    global pathToDataset, pathToDownload, pathToTrain, pathToTest, pathToValid, pathToModels
    
    pathToDataset  = pathlib.Path.cwd()
    pathToDownload = pathToDataset / 'downloads'

    pathToTrain = pathToDataset / 'train'
    if not pathToTrain.exists():
        pathToTrain.mkdir()

    pathToTest = pathToDataset / 'test'
    if not pathToTest.exists():
        pathToTest.mkdir()

    pathToValid = pathToDataset / 'valid'
    if not pathToValid.exists():
        pathToValid.mkdir()

    pathToModels = pathToDataset / 'models'
    if not pathToModels.exists():
        pathToModels.mkdir()




def categorizeArt():
    # list the folders required under pathToDataset folder
    # (using a list to reduce the lines of code)
    artCategories = getFolderNamesInDirectory(pathToDownload, ".")  #collects the list of folders
    print("Total no. of categories = ", len(artCategories))  #displays the number of classes (= Art categories)
    print("Categories: ", artCategories)  #displays the list of classes
    return artCategories



def extractData(zFile='downloads.zip',
                sourceDir=pathToDataset,
                destDir = pathToDownload):
    # extract the zip file 'zfile' found in the 'fromDir' directory
    # into the 'toDir' directory
    #print(sourceDir)
    images_file = sourceDir / zFile

    with ZipFile(images_file, 'r') as zipObj:
       zipObj.extractall(destDir)


def splitFiles(artCategories, nTrain=0.5, nTest=0.3, nValid=0.2 ):
    # split each category of art in the provided artCategories list into
    # test, train and valid directories
    numCategories = len(artCategories)
    
    for artCategory in artCategories:

        path_source = pathToDownload / artCategory
        os.chdir(path_source)
        files = getFilesInDirectory(path_source,"*.jpg")
        
        # Determines the splitting index
        index = splitIndex(len(files), [nTrain, nTest, nValid])
        
        # Split the files across the 3 datasets
        split_images = np.split(np.array(files), index)
        print(f"{artCategory} n={len(files)} index={index} images-> {[len(x) for x in split_images]}")

        # Sets the target folders and moves the files
        path_target_train = pathToTrain / artCategory
        if not path_target_train.exists():
            path_target_train.mkdir()
        for img_file in split_images[0]:
            shutil.move(img_file, path_target_train / img_file)    
                
        path_target_test = pathToTest / artCategory
        if not path_target_test.exists():
            path_target_test.mkdir()
        for img_file in split_images[1]:
            shutil.move(img_file, path_target_test / img_file)    

        path_target_valid = pathToValid / artCategory
        if not path_target_valid.exists():
            path_target_valid.mkdir()
        for img_file in split_images[2]:
            shutil.move(img_file, path_target_valid / img_file)


def cleanImages():
    cleanFilesAtLocation(pathToTrain)
    cleanFilesAtLocation(pathToValid)
    cleanFilesAtLocation(pathToTest)

            
########################################
# Utilities

# Retrieves the list of files with a directory which match a
# pattern (extension)
def getFilesInDirectory(pathToDir, extension = "*.*"):
    return sorted([fld.name for fld in pathToDir.glob(extension) if fld.is_file()])

    
# Retrieves the list of folders with a directory
# note - requirement for prefix is not clear yet, so not implemented
def getFolderNamesInDirectory(pathToDir, prefix = ""):
    return sorted([fld.name for fld in pathToDir.iterdir() if fld.is_dir()])


def resetFiles():
    # reset the files in the directory to the state before this module
    # was run
    for d in ['downloads', 'test', 'train', 'valid', 'models']:
        try:
            shutil.rmtree(pathToDataset / d)
        except:
            pass

def splitIndex(n, splits):
    # based on a total population size and a profile of "splits" produce
    # the index list which can create those splits using the np.split function.
    # Example of a splits list would be [0.2, 0.4, 0.4] which would produce
    # three splits the first containing 20% of the total 'n' the second
    # containing 40% and so on. 
    profile = [int(split * n) for split in splits]
    return sorted(profile[:len(profile)-1])


def cleanFilesAtLocation(location):
    artCategories = categorizeArt()

    # For each art category
    for artCategory in artCategories:

        # Sets the source folder
        path_source = location / artCategory

        # Sets the datasets
        files = getFilesInDirectory(path_source, '*.jpg')    # lists all the 'jpg' images in the folder

        for file in files:
            filePath = path_source / file
            try:
                with Image.open(filePath) as img:
                    pass
            except IOError:
                print( filePath )
                os.remove(filePath)



############################################################
# Download images from Google Image
# I don't think this library (google_images_download) is working any more
# per https://github.com/hardikvasa/google-images-download/issues/301

def download_google_images(query, ext="jpg",
                           limit=10,
                           print_urls=False,
                           usage_rights="Labeled-for-reuse",
                           size="Large"):
    response = google_images_download.googleimagesdownload() 
    # aspect ratio denotes the height width ratio of images to download. ("tall, square, wide, panoramic")  
    # specific_site
    # usage_rights: labeled-for-reuse
    arguments = {"keywords": query,     # keywords is the search query
                 "format": ext,         # format: svg, png, jpg
                 "limit": limit,        # limit: is the number of images to be downloaded
                 "print_urls": print_urls,   # print_urls: is to print the image file url
                 "size": size,          # size: large, medium, icon 
                 "usage_rights": usage_rights }       

    try:
        path = response.download(arguments)
    except FileNotFoundError:
        print( "Exception raised" )



###
def download_listed_images(filepath):
    # Download images from a list of urls contained in a file 'filepath'
    # specified as an input parameter. The name of the filepath should be
    # the category of the art listed in that file (e.g. cubism.txt)
    # output will be in the (e.g.) dataset/downloads/cubism directory

    # Check Art Category folder exists
    categoryFolder = pathToDownload / filepath[:-4]
    if not categoryFolder.exists():
        categoryFolder.mkdir()
    
    # grab the list of URLs from the input file, then initialize the total number of images downloaded so far
    with open(filepath) as f:
        urls = f.read().strip().split("\n")
        
    urlCounter = 0

    # loop the URLs
    for url in urls:
        try:
            # try to download the image
            req = requests.get(url, timeout=60)

            # save the image to disk
            pathToDownloadedImage = categoryFolder.joinpath("{}.jpg".format(str(urlCounter).zfill(8)))
            with open(pathToDownloadedImage, "wb") as downloaded_image:
                downloaded_image.write(req.content)
            
            # update the counter
            print("[INFO] downloaded: {}".format(pathToDownloadedImage))
            urlCounter += 1
            
        # handle if any exceptions are thrown during the download process
        except:
            print(f"[INFO] error downloading {pathToDownloadedImage}...skipping")

########################################


if __name__=="__main__":
    setup()
    resetFiles()
    prepDirectories()
    extractData(zFile='downloads.zip',
                sourceDir=pathToDataset,
                destDir = pathToDataset)
    download_listed_images("cubism.txt")
    download_listed_images("surrealism.txt")
    categories = categorizeArt()
    splitFiles(categories)
    cleanImages()
    

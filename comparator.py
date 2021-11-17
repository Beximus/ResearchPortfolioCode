# some code here adapted from the examples used here: https://github.com/haltakov/natural-language-image-search

from pathlib import Path
import argparse
import clip
import torch
from PIL import Image, ImageDraw, ImageFont
import math
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) 

# ----------- function computes clip vectors from a folder of images-----------

# Function that computes the feature vectors for a batch of images
def compute_clip_features(photos_batch):
    # Load all the photos from the files
    photos = [Image.open(photo_file) for photo_file in photos_batch]
    
    # Preprocess all photos
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

    with torch.no_grad():
        # Encode the photos batch to compute the feature vectors and normalize them
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return photos_features.cpu().numpy()

# ----------- function computes clip vectors from a single file -----------

# Function that computes the feature vectors for a batch of images
def compute_clip_features_single(picture):
    # Load in the picture
    photo = Image.open(picture)
    
    # Preprocess all photos
    picture_preprocessed = torch.stack([preprocess(photo)]).to(device)

    with torch.no_grad():
        # Encode the picture to compute the feature vectors and normalize
        pictureFeatures = model.encode_image(picture_preprocessed)
        pictureFeatures /= pictureFeatures.norm(dim=1,keepdim=True)
    
    # Transfer the feature vectors to the CPU and convert to numpy
    return pictureFeatures.cpu().numpy()

# ----------- function creates folders to store data -----------

def folderMaker(containerPath):
    # ---------- CREATE FOLDER TO SAVE ANALYSIS TO ----------
    try:
        os.mkdir(containerPath)
    except OSError:
        print("creation of the directory %s failed" %containerPath)
    else:
        print("successfully created the directory %s" %containerPath)

# ----------- function preprocesses images -----------

def preprocessImages(features_path,photos_files,photos_path):
    # Define the batch size so that it fits on your GPU. You can also do the processing on the CPU, but it will be slower.
    batch_size = 16

    # Compute how many batches are needed
    batches = math.ceil(len(photos_files) / batch_size)

    # Process each batch
    for i in range(batches):
        print(f"Processing batch {i+1}/{batches}")

        batch_ids_path = features_path / f"{i:010d}.csv"
        batch_features_path = features_path / f"{i:010d}.npy"
        
        # Only do the processing if the batch wasn't processed yet
        if not batch_features_path.exists():
            try:
                # Select the photos for the current batch
                batch_files = photos_files[i*batch_size : (i+1)*batch_size]

                # Compute the features and save to a numpy file
                batch_features = compute_clip_features(batch_files)
                np.save(batch_features_path, batch_features)

                # Save the photo IDs to a CSV file
                photo_ids = [photo_file.name for photo_file in batch_files]
                photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
                photo_ids_data.to_csv(batch_ids_path, index=False)
            except:
                # Catch problems with the processing to make the process more robust
                print(f'Problem with batch {i}')
    
    # Load all numpy files
    features_list = [np.load(features_file) for features_file in sorted(features_path.glob("*.npy"))]

    # Concatenate the features and store in a merged file
    features = np.concatenate(features_list)
    np.save(features_path / "features.npy", features)

    # Load all the photo IDs
    photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(features_path.glob("*.csv"))])
    photo_ids.to_csv(features_path / "photo_ids.csv", index=False)

    # save features as a csv for latent space

    pd.DataFrame(features).to_csv("testingFeaturesTesting.csv")
    

# ----------- function searches Images -----------
def searchImages(features_path,search_query,numberResults):
    # Read the photos table
    photos = pd.read_csv(features_path / "photo_ids.csv", sep='\t', header=0)

    # # Load the features and the corresponding IDs
    photo_features = np.load(features_path / "features.npy")
    photo_ids = pd.read_csv(features_path / "photo_ids.csv")
    photo_ids = list(photo_ids['photo_id'])

    # encode query into feature vector
    with torch.no_grad():
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    
    text_features = text_encoded.cpu().numpy()
    similarities = list((text_features @ photo_features.T).squeeze(0))
    best_photos = sorted(zip(similarities, range(photo_features.shape[0])), key=lambda x: x[0], reverse=True)
    
    results = []

    for i in range(numberResults):
        idx = best_photos[i][1]
        photo_id = photo_ids[idx]
        photo_probability = best_photos[i][0]
        result =[photo_id,photo_probability]
        results.append(result)
    
    return results

def distanceFinder(features_path,search_query,numberResults,compare_features):
    #aksrjfnaskdjfnask
    referencePhotos = pd.read_csv(features_path/"photo_ids.csv",sep='\t', header=0)
    referenceFeatures = np.load(features_path/"features.npy")
    referenceIds = pd.read_csv(features_path/"photo_ids.csv")
    referenceIds = list(referenceIds['photo_id'])
    comparisonPath = Path(compare_features,"preprocessed_feature_compare.npy")
    comparisonPicture = np.load(comparisonPath)
    comparisonValues = []
    print(len(referenceFeatures))
    counter = 0
    for count, item in enumerate(referenceFeatures):
        dist = np.linalg.norm(comparisonPicture-item)
        keepImageinfo = [dist,referenceIds[count]]
        comparisonValues.append(keepImageinfo)
    comparisonValues.sort(key=lambda x:x[0], reverse=True)
    print(comparisonValues)
    return comparisonValues
        
        
        
    

# image resizer 

def resizeImage(imagePath,newImagePath):
    BaseSize = 720
    img = Image.open(imagePath)
    curWidth = img.size[0]
    curHeight = img.size[1]
    if(curWidth>=curHeight):
        percent = (BaseSize/float(curWidth))
        height = int((float(curHeight)*float(percent)))
        width = BaseSize
    else:
        percent = (BaseSize/float(curHeight))
        width = int((float(curWidth)*float(percent)))
        height = BaseSize
    
    resized = img.resize((width,height), Image.ANTIALIAS)
    resized.save(newImagePath)


# draw search results document

def drawResults(analysis_path,photos_path,searchResults,search_query):
    documentLength = 50+(770*len(searchResults))
    documentWidth = 820
    analysisResultsPath = Path(analysis_path)/"analysisResults.jpg"
    res = Image.new('RGB',(documentWidth,documentLength),(255,255,255))
    writeText = ImageDraw.Draw(res)
    fnt = ImageFont.truetype('fonts/OpenSans-Regular.ttf',25)
    writeText.text((50,10), "Query: "+search_query, font=fnt, fill=(0,0,0))
    for indx, pictures in enumerate(searchResults):
        newname = "resized_"+str(pictures[1])+"_"+str(pictures[0])
        imagePath = Path(photos_path)/pictures[0]
        newImagePath = Path(analysis_path)/newname
        resizeImage(imagePath,newImagePath)
        offset = 50+(770*indx)
        textoffset = 770+(770*indx)
        smallerPicture = Image.open(newImagePath)
        yoffset = smallerPicture.size[1]
        yoffset = int((720 - yoffset)/2)
        res.paste(smallerPicture,(50,offset+yoffset))
        fnt = ImageFont.truetype('fonts/OpenSans-Regular.ttf',12)
        writeText.text((50,textoffset),str(indx+1)+". filename: "+str(pictures[0])+" , Probability: "+ str(float(100*pictures[1]))+"%",font=fnt,fill=(0,0,0))
    
    res.save(analysisResultsPath)

def writeCSV(searchResults,csvsavepath):
    container = []
    images = []
    probabilities = []
    for results in searchResults:
        images.append(results[0])
        probabilities.append(results[1])
    container.append(images)
    container.append(probabilities)
    analysis = { 'Images': images, 'Probabilities': probabilities}
    df = pd.DataFrame(analysis, columns=['Images','Probabilities'])
    print(df)
    df.to_csv(csvsavepath,index = False, header=True)


def comparator(compare_path,compare_features):
    #stuff goes here
    comparison = compute_clip_features_single(compare_path)
    # print(compare_features)
    if (os.path.exists(str(compare_features))):
        print("this image has been compared before")
    else:
        folderMaker(compare_features)
    new_compare_features = Path(compare_features,"preprocessed_feature_compare.npy")
    # print(new_compare_features)
    # print(comparison)
    compared_preprocessed_features = compute_clip_features_single(compare_path)
    np.save(new_compare_features,compared_preprocessed_features)



# Main Section

def main():
    #put main section here
    parser = argparse.ArgumentParser(description="Pick Best Image from Search")
    parser.add_argument('--folder', default='testImages/', help="input folder")
    parser.add_argument('--query', default='picture of something', help="search query")
    parser.add_argument('--matches', default=3, help ="number of matches wanted")
    parser.add_argument('--compareImage', default="false", help="default false set true if image is being compared")
    parser.add_argument('--compareFile',default=None,help="path for image to compare to")
    args = parser.parse_args()

    # set search query
    search_query = str(args.query)
    print("Looking for best matches for: " + search_query)

    # Set the path to the photos
    photos_path = Path(args.folder)

    # set how many matches should be returned
    numberResults = int(args.matches)

    # List all JPGs in the folder
    photos_files = list(photos_path.glob("*.jpg"))
    # photos_files = list(photos_path.glob("*.png"))

    # Print some statistics
    print(f"Photos found: {len(photos_files)}")

    # Create folder for preprocess data
    features_path = Path(photos_path)/"features"
    
    if (os.path.exists(str(features_path))):
        print("Image features have already been preprocessed")
    else:
        print("Preprocessing Images")
        folderMaker(features_path)
        preprocessImages(features_path,photos_files,photos_path)
        print("Image preprocessing successful")


    

    # Preprocessing comparison image if there is one
    if (args.compareImage=="true"):
        comparator_path = Path(photos_path)/"comparison"
        if(os.path.exists(str(comparator_path))):
            print("Comparisons have been run on this dataset before")
        else:
            print("Comparing Image results for the first time")
            folderMaker(comparator_path)
        compare_path = Path(args.compareFile)
        pathstem = compare_path.stem
        compare_features = Path(comparator_path,pathstem)
        comparator(compare_path,compare_features)
        distances = distanceFinder(features_path,search_query,numberResults,compare_features)

             
    
    # Create folder for analysis images
    analysis_path = Path(photos_path)/"analysis"

    if (os.path.exists(str(analysis_path))):
        print("This set has been queried before")
    else:
        print("This set has never been queried")
        folderMaker(analysis_path)

    newAnalysisPath = Path(analysis_path)/search_query

    if (os.path.exists(str(newAnalysisPath))):
        print("The query: "+search_query+" has been explored before - rerunning analysis")
    else:
        print("The query: "+search_query+" has not been explored before")
        folderMaker(newAnalysisPath)

    print("Results can be found in: "+str(newAnalysisPath))

    # Run Image Search
    searchResults = searchImages(features_path,search_query,numberResults)
    # print(searchResults)
    csvsavepath = search_query+"_results.csv" 
    csvsavepath = Path(newAnalysisPath)/csvsavepath

    writeCSV(searchResults,csvsavepath)

    drawResults(newAnalysisPath,photos_path,searchResults,search_query)

    




if __name__ == '__main__':
    main()
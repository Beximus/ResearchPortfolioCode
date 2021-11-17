import argparse
import numpy as np
import torch
import clip
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import json
import glob
import os
import csv
from datetime import datetime

# --------- this imports the classses by default ---------- turning the template into search classes has not yet been arranged

with open('imagenetClasses.json','r') as f:
    imagenet_classes = json.load(f)

with open('imagenetTemplates.json','r') as g:
    imagenet_templates = json.load(g)

# --------- this imports the classses by default ----------

# ----------- ZERO SHOT CLASSIFIER NEEDS CHANGING LATER -----------
def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cpu() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cpu()
    return zeroshot_weights
# ----------- ZERO SHOT CLASSIFIER NEEDS CHANGING LATER -----------

# ---------- CSV WRITER ----------
def csvWriter(nameOfFile,rowData):
    with open(nameOfFile, mode='a') as currentFile:
            currentFile = csv.writer(currentFile, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
            currentFile.writerow(rowData)
# ---------- CSV WRITER ----------

# ---------- CREATE IMAGE CARD ----------

def imageCard(analysisResults,imagePath,analysispath,currentImagePath):
    # --------- PATH FOR NEW IMAGE ----------
    newImagePath = os.path.join(analysispath+imagePath)
    img = Image.open(currentImagePath)
    newbasesize = 480
    bgH = 520
    bgW = 1000
    cardBackground = Image.new('RGB', (bgW,bgH),(255,255,255))
    currentwidth = img.size[0]
    currentheight = img.size[1]

    if(currentwidth>=currentheight):
        percent = (newbasesize/float(currentwidth))
        height = int((float(currentheight)*float(percent)))
        width = newbasesize
        offset = (20,int(float((newbasesize-height)/2)+20))
    else:
        percent = (newbasesize/float(currentheight))
        width = int((float(currentwidth)*float(percent)))
        height = newbasesize
        offset = (int((float(newbasesize-width)/2)+20),20)
    
    img = img.resize((width,height), Image.ANTIALIAS)
    cardBackground.paste(img,offset)

    writeText = ImageDraw.Draw(cardBackground)
    fnt = ImageFont.truetype('fonts/OpenSans-Regular.ttf',30)
    writeText.text((540,20),"filename: "+imagePath,font=fnt,fill=(0,0,0))
    fnt = ImageFont.truetype('fonts/OpenSans-Italic.ttf', 20)
    for iteration, index in enumerate(analysisResults):
        writeText.text((540,(50*iteration)+100),analysisResults[iteration],font=fnt,fill=(0,0,0))
    # print(indicies)

    cardBackground.save(newImagePath)

def folderMaker(containerPath):
    # ---------- CREATE FOLDER TO SAVE ANALYSIS TO ----------
    try:
        os.mkdir(containerPath)
    except OSError:
        print("creation of the directory %s failed" %containerPath)
    else:
        print("successfully created the directory %s" %containerPath)

# ----------- MAIN SECTION -----------

def main():
    # import the classes to main
    global imagenet_classes
    # make sure that the experiment date and time is stored -- if no name is specified date and time is used
    now = datetime.now()
    defaultcsvname = now.strftime("%d-%m-%Y-%H-%M-%S")+".csv"

    # set up arguments for the script

    parser = argparse.ArgumentParser(description="score images against classifiers")
    parser.add_argument('--input-glob', default='CLIP.png',
                        help="input folder")
    parser.add_argument("--labels", default=None,
                        help="comma separated list of labels")
    parser.add_argument("--name",default=defaultcsvname,
                        help="name the export folder")
    parser.add_argument("--labelFiles", default=None,
                        help="a json of labels")

    args = parser.parse_args()

    # for whatever reason this does not work without cuda (could be my machine)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32")

    labels = []
    if args.labels is not None:
        labels = args.labels.split(",")
        imagenet_classes = imagenet_classes + labels
        # imagenet_classes = labels
    
    # lets you add a json file of labels

    if args.labelFiles is not None:
        labelFile = glob.glob(args.labelFiles)
        with open(labelFile[0],'r') as l:
            input_labels = json.load(l)
        imagenet_classes = imagenet_classes + input_labels
        print(imagenet_classes)

    filename= args.name 

    InputFolder= glob.glob(args.input_glob)

    ImagefolderPath = os.path.abspath(InputFolder[0])
    print(ImagefolderPath)
    
    imagesToProcess = os.listdir(ImagefolderPath)
    print(str(len(imagesToProcess))+ " files found to process")

    # ---------- MAKES FOLDER TO SAVE ANALYSIS TO -----------
    analysispath = os.path.join(ImagefolderPath+"/analysis/")
    folderMaker(analysispath)
    # ---------- FILE PATH TO SAVE ANALYSIS CSV ----------
    filename = os.path.join(analysispath+filename)
    
    print("Building labels")
    zeroshot_weights = zeroshot_classifier(model, imagenet_classes, imagenet_templates)
    
    for picture in imagesToProcess:
        
        # --------- PATH CONSTRUCTOR FOR PICTURE ANALYSIS ----------
        currentPicture = os.path.basename(picture)
        currentPath = os.path.join(ImagefolderPath+"/"+currentPicture)


        # --------- CSV CONSTRUCTOR ----------

        currentCSVrow = []
        currentCSVrow.append(currentPicture)

        # ----------- PREPROCESS IMAGE ----------
        
        image = preprocess(Image.open(currentPath)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits_per_image = 100. * image_features @ zeroshot_weights
            
            # logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1)

        values, indices = probs[0].topk(5)
        results = []

        print(f"\nTop scores for file {currentPicture}")
        for i in range(len(indices)):
            j = indices[i]
            print(f"{imagenet_classes[j]:>16s} {100 * probs[0][j]:5.2f} ({i+1})")
            results.append(f"{imagenet_classes[j]:>16s} {100 * probs[0][j]:5.2f} ({i+1})")
            currentCSVrow.append(f"{imagenet_classes[j]:>16s} {100 * probs[0][j]:5.2f} ({i+1})")

        # if len(imagenet_classes) > 1000:
        #     print("\nScores for provided labels")
        #     currentCSVrow.append("Scores for provided labels:")
        #     for j in range(1000,len(imagenet_classes)):
        #         print(f"{imagenet_classes[j]:>16s} {100 * probs[0][j]:5.2f}")
        #         currentCSVrow.append(f"{imagenet_classes[j]:>16s} {100 * probs[0][j]:5.2f}")
        print(currentCSVrow)
        
        imageCard(results,currentPicture,analysispath,currentPath)

        csvWriter(filename,currentCSVrow)
        

if __name__ == '__main__':
    main()

# ----------- MAIN SECTION -----------
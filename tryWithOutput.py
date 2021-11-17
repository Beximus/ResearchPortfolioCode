import argparse
import numpy as np
import torch
import clip
from tqdm import tqdm
from PIL import Image

import json
import glob
import os
import csv
from datetime import datetime

with open('imagenetClasses.json','r') as f:
    imagenet_classes = json.load(f)

with open('imagenetTemplates.json','r') as g:
    imagenet_templates = json.load(g)

# print(imagenet_classes)
# print(imagenet_templates)

# ----------- ZERO SHOT CLASSIFIER NEEDS CHANGING LATER -----------
def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights
# ----------- ZERO SHOT CLASSIFIER NEEDS CHANGING LATER -----------

# ---------- CSV WRITER ----------
def csvWriter(nameOfFile,rowData):
    with open(nameOfFile, mode='a') as currentFile:
            currentFile = csv.writer(currentFile, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
            currentFile.writerow(rowData)



# ----------- MAIN SECTION -----------

def main():
    global imagenet_classes
    now = datetime.now()
    defaultcsvname = now.strftime("%d-%m-%Y-%H-%M-%S")+".csv"

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32")

    labels = []
    if args.labels is not None:
        labels = args.labels.split(",")
        imagenet_classes = imagenet_classes + labels
        # imagenet_classes = labels
    
    if args.labelFiles is not None:
        labelFile = glob.glob(args.labelFiles)
        with open(labelFile[0],'r') as l:
            input_labels = json.load(l)
        imagenet_classes = imagenet_classes + input_labels
        print(imagenet_classes)

    filename= args.name 
    
    print("Building labels")
    zeroshot_weights = zeroshot_classifier(model, imagenet_classes, imagenet_templates)

    InputFolder= glob.glob(args.input_glob)

    ImagefolderPath = os.path.abspath(InputFolder[0])
    print(ImagefolderPath)
    
    imagesToProcess = os.listdir(ImagefolderPath)
    print(str(len(imagesToProcess))+ " files found to process")
    

    for picture in imagesToProcess:
        currentPicture = os.path.basename(picture)
        currentPath = os.path.join(ImagefolderPath+"/"+currentPicture)
        
        currentCSVrow = []
        currentCSVrow.append(currentPicture)
        
        image = preprocess(Image.open(currentPath)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits_per_image = 100. * image_features @ zeroshot_weights
            
            # logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1)

        values, indices = probs[0].topk(5)

        print(f"\nTop scores for file {currentPicture}")
        for i in range(len(indices)):
            j = indices[i]
            print(f"{imagenet_classes[j]:>16s} {100 * probs[0][j]:5.2f} ({i+1})")
            currentCSVrow.append(f"{imagenet_classes[j]:>16s} {100 * probs[0][j]:5.2f} ({i+1})")

        # if len(imagenet_classes) > 1000:
        #     print("\nScores for provided labels")
        #     currentCSVrow.append("Scores for provided labels:")
        #     for j in range(1000,len(imagenet_classes)):
        #         print(f"{imagenet_classes[j]:>16s} {100 * probs[0][j]:5.2f}")
        #         currentCSVrow.append(f"{imagenet_classes[j]:>16s} {100 * probs[0][j]:5.2f}")
        print(currentCSVrow)
        
        csvWriter(filename,currentCSVrow)
        

if __name__ == '__main__':
    main()
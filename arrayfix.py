import json
import os

with open('sports.json','r') as f:
    sportslist = json.load(f)

def main():
    listosports=[]
    sports = sportslist
    for sport in sports:
        if sport not in listosports:
            listosports.append(sport)
            # print(sport)
    with open('zports.json','w') as g:
        json.dump(listosports,g)



main()
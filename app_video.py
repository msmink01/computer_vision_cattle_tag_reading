import cv2
import argparse
import time
import pandas as pd
from readTags import read_tags_in_webcam_image, create_cow_model
import os


def cam(args):
    print(f"Reading video: {args.input}, DrinkingOnly: {args.drinking_only}, while skipping {args.skip} frames.")
    print(args)
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = os.path.getmtime(args.input)
    skip = args.skip + 1

    outputs = dict()
    cow_model = create_cow_model(args.cow_model)
    
    frame_number = 0
    #start_time = time.time()
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        frame_time = frame_number * (1 / fps)
        frame_number += 1
        if frame_number % skip != 0:
            continue
        
        
        out, box_image = read_tags_in_webcam_image(frame, cow_model, args.digit_model, args.drinking_only)
        if out:
            outputs[frame_number] = (frame_time, out)
        
        
        cow_image = cv2.cvtColor(box_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", cow_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    tags = dict() # dictionary of tag to frame pairs
    for fnum in outputs:
        for k in outputs[fnum][1]:
            tag = outputs[fnum][1][k]
            t = outputs[fnum][0]
            if tag[0] in tags:
                tags[tag[0]].append((fnum, tag[1], t, tag[3], tag[4])) # "text": (frame_number, textConfidence, time, nearConfidence, drinkingConfidence
            else:
                tags[tag[0]] = [(fnum, tag[1], t, tag[3], tag[4])]
    
    timeForTags = dict() # dictionary of tag to time pairs
    for tag in tags:
        listOfTimes = tags[tag]
        i = 0
        while(i < len(listOfTimes)):
            #print(f"i = {i}")
            totalConf = float(listOfTimes[i][1])
            totalDConf = float(listOfTimes[i][4])
            numFrames = 1
            start = listOfTimes[i][2]
            end = listOfTimes[i][2]
            while(i+1 < len(listOfTimes) and listOfTimes[i+1][2] - listOfTimes[i][2] < args.drinking_thresh):
                totalConf += float(listOfTimes[i+1][1])
                totalDConf += float(listOfTimes[i+1][4])
                numFrames += 1
                end = listOfTimes[i+1][2]
                i += 1
            if tag in timeForTags:
                timeForTags[tag].append((start, end, totalConf / numFrames, totalDConf / numFrames))
            else:
                timeForTags[tag] = [(start, end, totalConf / numFrames, totalDConf / numFrames)]
            i += 1
    
    tagNames = []
    tagIndexOfOccurrence = []
    averageTextConfidence = []
    averageDrinkingConfidence = []
    timeStart = []
    timeEnd = []
    for tag_key in timeForTags:
        for i in range(len(timeForTags[tag_key])):
            reading = timeForTags[tag_key][i]
            tagNames.append(tag_key)
            tagIndexOfOccurrence.append(i)
            averageTextConfidence.append(reading[2])
            averageDrinkingConfidence.append(reading[3])
            timeStart.append(reading[0])
            timeEnd.append(reading[1])
            
    df = pd.DataFrame({"tagNames": tagNames, "tagIndexOfOccurrence": tagIndexOfOccurrence, "averageTextConfidence": averageTextConfidence, "averageDrinkingConfidence": averageDrinkingConfidence, "timeStart": timeStart, "timeEnd": timeEnd}) 

    print(df)
    if args.output != None:
        df.to_csv(args.output)
        print("Finished making csv.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help="path to video to present")
    parser.add_argument('--cow_model', required=True, type=str, help="path to cow model to use")
    parser.add_argument('--digit_model', required=True, type=str, help="path to digit model to use")
    parser.add_argument('--drinking_only', default=False, action='store_true', help="include only drinking tags or not")
    parser.add_argument('--drinking_thresh', default=2, type=float, help="how many seconds can be between detections and still be considered one detection")
    parser.add_argument('--output', type=str, default=None, help="csv file to output to")
    parser.add_argument('--skip', type=int, default = 0, help="Number of frames to skip between reads")

    stuff = parser.parse_args()
    
    cam(stuff)
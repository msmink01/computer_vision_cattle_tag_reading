import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from extrautils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate, PredictedImage, PredictedImages, PredictedVideoBatch, PredictedWebcamImage
from model import Model

import cv2
import numpy as np
import pandas as pd
import os
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from difflib import SequenceMatcher

CLASSES = [
    'background', 'far', 'near', 'drinking'
]
YOLO_CLASSES = ['near', 'far', 'drinking']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

cow_threshold = 0.5
does_tag_belong_threshold = 0.9
tag_reading_threshold = 0.4

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Create FasterRCNN model
def create_model(num_classes, min_size=800):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size = min_size)
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

# Create cow model
def create_cow_model(cowModelPath):
    global CLASSES, device
    cow_model = create_model(num_classes=len(CLASSES), min_size = 800).to(device)
    cow_model.load_state_dict(torch.load(
        cowModelPath, map_location=device
    ))
    cow_model.eval()
    return cow_model

def create_cow_model_yolo(cowModelPath):
    cow_model = torch.hub.load("yolov5", "custom", cowModelPath, source="local").to(device)
    return cow_model

def get_cow_predictions(image_path, cowModelPath, yolo=True):
    global device, CLASSES, YOLO_CLASSES
    if yolo:
        cow_model = create_cow_model_yolo(cowModelPath)
        imgs = [Image.open(image_path)]
        results = cow_model(imgs)
        
        out = results.xyxy[0]
        #print(out)
        #print(out[:, 5].cpu().numpy())
        #for i in out[:, 5].cpu().numpy():
        #    print(i)
        pred_classes = [YOLO_CLASSES[int(i)] for i in out[:, 5].cpu().numpy()]
        pred_bboxes = out[:, :4].detach().cpu().numpy()
        pred_scores = out[:, 4].detach().cpu().numpy()
        pred_labels = out[:, 5].detach().cpu().numpy()
        # print(pred_classes, pred_bboxes, pred_scores, pred_labels)
    else:
        
        cow_model = create_cow_model(cowModelPath)

        cow_image = cv2.imread(image_path)
        image_width = cow_image.shape[1]
        image_height = cow_image.shape[0]
        orig_img = cow_image.copy()
        # BGR to RGB
        cow_image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        cow_image /= 255.0
        cow_image = np.transpose(cow_image, (2, 0, 1)).astype(np.float32)
        cow_image = torch.tensor(cow_image, dtype=torch.float).cuda()
        # add batch dimension
        cow_image = torch.unsqueeze(cow_image, 0)

        with torch.no_grad():
            outputs = cow_model(cow_image)

        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_labels = outputs[0]['labels'].detach().cpu().numpy()
        #print(pred_classes, pred_bboxes, pred_scores, pred_labels)

    numOfPreds = len(pred_classes)
    for i in reversed(range(numOfPreds)):
        if pred_scores[i] < cow_threshold:
            pred_scores = np.delete(pred_scores,i)
            pred_bboxes = np.delete(pred_bboxes,i, axis=0)
            pred_classes.pop(i)
            pred_labels = np.delete(pred_labels,i)
            
    return pred_classes, pred_labels, pred_bboxes, pred_scores

def get_cow_predictions_video(capture, cowModelPath, yolo, skipFrames=16):
    if yolo:
        cow_model = create_cow_model_yolo(cowModelPath)
    else:
        cow_model = create_cow_model(cowModelPath)
    global device, CLASSES
    
    if (capture.isOpened() == False):
        print('Error while trying to read video. Please check path again')
        return
    
    capture.set(1, 0) # reset the frames to 0
    frame_number = 0
    while(True):
        frames = []
        frame_numbers = []
        for i in range(skipFrames * 8):
            ret, frame = capture.read()
            frame_number += 1 # 1st frame
            if i % skipFrames != 0: # Only do every Xth frame
                continue
            if ret == False:
                break
            if frame_number >= capture.get(cv2.CAP_PROP_FRAME_COUNT):
                break
            
            if yolo:
                cow_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cow_image = Image.fromarray(cow_image)
            else:
                cow_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
                cow_image /= 255.0
                cow_image = np.transpose(cow_image, (2, 0, 1)).astype(np.float32)
                cow_image = torch.tensor(cow_image, dtype=torch.float).cuda()
                # add batch dimension
                cow_image = torch.unsqueeze(cow_image, 0)
    
            frames.append(cow_image)
            frame_numbers.append(frame_number)
        
        toReturn = dict()    
        
        if yolo:
            results = cow_model(frames)
            outputs = results.xyxy
            
            for i in range(len(outputs)):
                out = outputs[i]
                fnum = frame_numbers[i]
                pred_classes = [YOLO_CLASSES[int(j)] for j in out[:, 5].cpu().numpy()]
                pred_bboxes = out[:, :4].detach().cpu().numpy()
                pred_scores = out[:, 4].detach().cpu().numpy()
                pred_labels = out[:, 5].detach().cpu().numpy()
                
                numOfPreds = len(pred_classes)
                for i in reversed(range(numOfPreds)):
                    if pred_scores[i] < cow_threshold:
                        pred_scores = np.delete(pred_scores,i)
                        pred_bboxes = np.delete(pred_bboxes,i, axis=0)
                        pred_classes.pop(i)
                        pred_labels = np.delete(pred_labels,i)
                
                toReturn[fnum] = (pred_classes, pred_labels, pred_bboxes, pred_scores)
                
        else:
        
            frames = torch.cat(frames, axis=0)
    
            with torch.no_grad():
                outputs = cow_model(frames)
        
        
            for i in range(len(outputs)):
                out = outputs[i]
                fnum = frame_numbers[i]
                pred_classes = [CLASSES[i] for i in out['labels'].cpu().numpy()]
                pred_bboxes = out['boxes'].detach().cpu().numpy()
                pred_scores = out['scores'].detach().cpu().numpy()
                pred_labels = out['labels'].detach().cpu().numpy()

                numOfPreds = len(pred_classes)
                for i in reversed(range(numOfPreds)):
                    if pred_scores[i] < cow_threshold:
                        pred_scores = np.delete(pred_scores,i)
                        pred_bboxes = np.delete(pred_bboxes,i, axis=0)
                        pred_classes.pop(i)
                        pred_labels = np.delete(pred_labels,i)
                
                toReturn[fnum] = (pred_classes, pred_labels, pred_bboxes, pred_scores)
        
        oldFrame = frame_number
        yield toReturn
        capture.set(1, oldFrame) # reset the frames to the place we left off
        #print(f"Reset cap to {oldFrame}")
        
        if frame_number >= capture.get(cv2.CAP_PROP_FRAME_COUNT):
            break
                
                
def get_cow_predictions_webcam_image(frame, cow_model, yolo):
    global device, CLASSES, YOLOCLASSES
    
    if yolo:
        cow_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgs = [Image.fromarray(cow_image)]
        results = cow_model(imgs)
        out = results.xyxy[0]
        
        pred_classes = [YOLO_CLASSES[int(i)] for i in out[:, 5].cpu().numpy()]
        pred_bboxes = out[:, :4].detach().cpu().numpy()
        pred_scores = out[:, 4].detach().cpu().numpy()
        pred_labels = out[:, 5].detach().cpu().numpy()
        
    else:

        image_width = frame.shape[1]
        image_height = frame.shape[0]
        orig_img = frame.copy()
        # BGR to RGB
        cow_image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        cow_image /= 255.0
        cow_image = np.transpose(cow_image, (2, 0, 1)).astype(np.float32)
        cow_image = torch.tensor(cow_image, dtype=torch.float).cuda()
        # add batch dimension
        cow_image = torch.unsqueeze(cow_image, 0)

        with torch.no_grad():
            outputs = cow_model(cow_image)

        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_labels = outputs[0]['labels'].detach().cpu().numpy()

    numOfPreds = len(pred_classes)
    for i in reversed(range(numOfPreds)):
        if pred_scores[i] < cow_threshold:
            pred_scores = np.delete(pred_scores,i)
            pred_bboxes = np.delete(pred_bboxes,i, axis=0)
            pred_classes.pop(i)
            pred_labels = np.delete(pred_labels,i)
            
    return pred_classes, pred_labels, pred_bboxes, pred_scores

def get_cow_predictions_dir(dir_path, cowModelPath, yolo):
    outcomes = dict()
    global device, CLASSES, YOLO_CLASSES
    
    
    l = os.listdir(dir_path)
    l = list(filter(lambda x: "jpg" in x or "png" in x or "jpeg" in x, l))
    
    if yolo:
        cow_model = create_cow_model_yolo(cowModelPath)
    else:
        cow_model = create_cow_model(cowModelPath)
        
    for image_path in l:
        if yolo:
            imgs = [Image.open(os.path.join(dir_path, image_path))]
            results = cow_model(imgs)

            out = results.xyxy[0]
            #print(out)
            #print(out[:, 5].cpu().numpy())
            #for i in out[:, 5].cpu().numpy():
            #    print(i)
            pred_classes = [YOLO_CLASSES[int(i)] for i in out[:, 5].cpu().numpy()]
            pred_bboxes = out[:, :4].detach().cpu().numpy()
            pred_scores = out[:, 4].detach().cpu().numpy()
            pred_labels = out[:, 5].detach().cpu().numpy()
            # print(pred_classes, pred_bboxes, pred_scores, pred_labels)
        else:
            cow_image = cv2.imread(dir_path + "/" + image_path)
            image_width = cow_image.shape[1]
            image_height = cow_image.shape[0]
            orig_img = cow_image.copy()
            # BGR to RGB
            cow_image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            # make the pixel range between 0 and 1
            cow_image /= 255.0
            cow_image = np.transpose(cow_image, (2, 0, 1)).astype(np.float32)
            cow_image = torch.tensor(cow_image, dtype=torch.float).cuda()
            # add batch dimension
            cow_image = torch.unsqueeze(cow_image, 0)

            with torch.no_grad():
                outputs = cow_model(cow_image)

            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
            pred_scores = outputs[0]['scores'].detach().cpu().numpy()
            pred_labels = outputs[0]['labels'].detach().cpu().numpy()

        numOfPreds = len(pred_classes)
        for i in reversed(range(numOfPreds)):
            if pred_scores[i] < cow_threshold:
                pred_scores = np.delete(pred_scores,i)
                pred_bboxes = np.delete(pred_bboxes,i, axis=0)
                pred_classes.pop(i)
                pred_labels = np.delete(pred_labels,i)
                
        outcomes[image_path] = (pred_classes, pred_labels, pred_bboxes, pred_scores)
            
    return outcomes
    

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def draw_boxes(boxes, classes, labels, scores, image, tag_indexes=None, preds=None, confs=None):
    global COLORS
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[int(labels[i])]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        if tag_indexes != None and classes[i] == "near" and str(i) in tag_indexes:
            cv2.putText(image, f'{i}: {classes[i]} - {truncate(scores[i], 3)}; \"{preds[tag_indexes.index(str(i))]}\" - {confs[tag_indexes.index(str(i))]}', (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, 
                    lineType=cv2.LINE_AA)
        else:
            cv2.putText(image, f'{i}: {classes[i]} - {truncate(scores[i], 3)}', (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, 
                    lineType=cv2.LINE_AA)
    return image
    
    
    
    
    
def show_cow_predictions(image_path, cowModelPath, yolo=True):
    global device, CLASSES, YOLO_CLASSES
    cow_image = cv2.imread(image_path)
    orig_img = cow_image.copy()
    if yolo:
        cow_model = create_cow_model_yolo(cowModelPath)
        imgs = [Image.open(image_path)]
        results = cow_model(imgs)
        
        out = results.xyxy[0]
        #print(out)
        #print(out[:, 5].cpu().numpy())
        #for i in out[:, 5].cpu().numpy():
        #    print(i)
        pred_classes = [YOLO_CLASSES[int(i)] for i in out[:, 5].cpu().numpy()]
        pred_bboxes = out[:, :4].detach().cpu().numpy()
        pred_scores = out[:, 4].detach().cpu().numpy()
        pred_labels = out[:, 5].detach().cpu().numpy()
        # print(pred_classes, pred_bboxes, pred_scores, pred_labels)
    else:
        
        cow_model = create_cow_model(cowModelPath)

        #cow_image = cv2.imread(image_path)
        image_width = cow_image.shape[1]
        image_height = cow_image.shape[0]
        #orig_img = cow_image.copy()
        # BGR to RGB
        cow_image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        cow_image /= 255.0
        cow_image = np.transpose(cow_image, (2, 0, 1)).astype(np.float32)
        cow_image = torch.tensor(cow_image, dtype=torch.float).cuda()
        # add batch dimension
        cow_image = torch.unsqueeze(cow_image, 0)

        with torch.no_grad():
            outputs = cow_model(cow_image)

        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_labels = outputs[0]['labels'].detach().cpu().numpy()
        #print(pred_classes, pred_bboxes, pred_scores, pred_labels)

    numOfPreds = len(pred_classes)
    for i in reversed(range(numOfPreds)):
        if pred_scores[i] < cow_threshold:
            pred_scores = np.delete(pred_scores,i)
            pred_bboxes = np.delete(pred_bboxes,i, axis=0)
            pred_classes.pop(i)
            pred_labels = np.delete(pred_labels,i)
    
    box_image = draw_boxes(pred_bboxes, pred_classes, pred_labels, pred_scores, orig_img)
    #print(box_image)
    #print(type(box_image))
    #matplotlib.use( 'tkagg' )
    #print(matplotlib.get_backend())
    fig, ax1 = plt.subplots(1, figsize = (16,9))
    ax1.imshow(box_image)
    #fig.show()

def IoU(drinkingArray, nearArray):
    # compute intersection piece
    xA = max(drinkingArray[0], nearArray[0])
    yA = max(drinkingArray[1], nearArray[1])
    xB = min(drinkingArray[2], nearArray[2])
    yB = min(drinkingArray[3], nearArray[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # compute areas of boxes by themselves
    drinkingArea = (drinkingArray[2] - drinkingArray[0] + 1) * (drinkingArray[3] - drinkingArray[1] + 1)
    nearArea = (nearArray[2] - nearArray[0] + 1) * (nearArray[3] - nearArray[1] + 1)
    
    # compute IoU
    return interArea / float(drinkingArea + nearArea - interArea)

def does_tag_belong(drinkingArray, nearArray):
    # compute intersection piece
    xA = max(drinkingArray[0], nearArray[0])
    yA = max(drinkingArray[1], nearArray[1])
    xB = min(drinkingArray[2], nearArray[2])
    yB = min(drinkingArray[3], nearArray[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    nearArea = (nearArray[2] - nearArray[0] + 1) * (nearArray[3] - nearArray[1] + 1)
    
    return interArea / float(nearArea)


def determine_tags_to_be_read(p_classes, p_labels, p_bboxes, p_scores, drinkingOnly=True):
    numPreds = len(p_classes)
    drinkings = []
    nears = []
    for i in range(numPreds):
        if p_classes[i] == 'drinking':
            drinkings.append((i, p_classes[i], p_labels[i], p_bboxes[i], p_scores[i]))
        if p_classes[i] == 'near':
            nears.append((i, p_classes[i], p_labels[i], p_bboxes[i], p_scores[i], False))

    tagsToRead = []
    for drinks in drinkings:
        for i in range(len(nears)):
            near = nears[i]
            if does_tag_belong(drinks[3], near[3]) >= does_tag_belong_threshold:
                # Tag does belong to a drinking cow 
                #print(f'Adding tag: {near}')
                #print(f"Nears looks like: {nears}")
                n = list(near)
                n[5] = drinks[4]
                near = tuple(n)
                nears[i] = near
                tagsToRead.append(near)
                
    # Get rid of possible duplicates
    alreadyGotten = []
    for i in reversed(range(len(tagsToRead))):
        if tagsToRead[i][0] not in alreadyGotten:
            alreadyGotten.append(tagsToRead[i][0])
        else:
            tagsToRead.pop(i)
    
    if drinkingOnly:
        return tagsToRead
    else:
        return nears


def single_img(opt, tagsToBeRead, doPrint=True):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    if doPrint:
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    if doPrint:
        print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = PredictedImage(tagsToBeRead, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    
    # lists to output findings
    img_name_list = []
    predicted_text_list = []
    predicted_confidences_list = []

    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            #log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            if doPrint:
                print(f'{dashed_line}\n{head}\n{dashed_line}')
            #log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                if doPrint:
                    print(type(img_name), type(pred), confidence_score)
                    print(f'{str(img_name):25s}\t{pred:25s}\t{confidence_score:0.4f}')
                
                img_name_list.append(f'{img_name}')
                predicted_text_list.append(f'{pred}')
                predicted_confidences_list.append(f'{confidence_score:0.4f}')
    
                #log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            #log.close()
    return img_name_list, predicted_text_list, predicted_confidences_list

def webcam_img(opt, frame, tagsToBeRead, doPrint=True):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    if doPrint:
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    if doPrint:
        print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = PredictedWebcamImage(frame, tagsToBeRead, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    
    # lists to output findings
    img_name_list = []
    predicted_text_list = []
    predicted_confidences_list = []

    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            #log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            if doPrint:
                print(f'{dashed_line}\n{head}\n{dashed_line}')
            #log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                if doPrint:
                    print(type(img_name), type(pred), confidence_score)
                    print(f'{str(img_name):25s}\t{pred:25s}\t{confidence_score:0.4f}')
                
                img_name_list.append(f'{img_name}')
                predicted_text_list.append(f'{pred}')
                predicted_confidences_list.append(f'{confidence_score:0.4f}')
    
                #log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            #log.close()
    return img_name_list, predicted_text_list, predicted_confidences_list

def multiple_images(opt, toBeReadDict, doPrint):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    if doPrint:
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    if doPrint:
        print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = PredictedImages(toBeReadDict, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    
    # lists to output findings
    img_name_list = []
    predicted_text_list = []
    predicted_confidences_list = []

    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            #log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            if doPrint:
                print(f'{dashed_line}\n{head}\n{dashed_line}')
            #log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                if pred_max_prob.numel() == 0:
                    # print("SKIPPING", pred, pred_max_prob)
                    continue
                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                if doPrint:
                    print(type(img_name), type(pred), confidence_score)
                    print(f'{str(img_name):25s}\t{pred:25s}\t{confidence_score:0.4f}')
                
                img_name_list.append(f'{img_name}')
                predicted_text_list.append(f'{pred}')
                predicted_confidences_list.append(f'{confidence_score:0.4f}')
    
                #log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            #log.close()
    return img_name_list, predicted_text_list, predicted_confidences_list


def video_batch(opt, cap, toBeReadDict, doPrint):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    if doPrint:
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    if doPrint:
        print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = PredictedVideoBatch(cap, toBeReadDict, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    
    # lists to output findings
    img_name_list = []
    predicted_text_list = []
    predicted_confidences_list = []

    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            #log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            if doPrint:
                print(f'{dashed_line}\n{head}\n{dashed_line}')
            #log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                if pred_max_prob.numel() == 0:
                    # print("SKIPPING", pred, pred_max_prob)
                    continue
                # print(pred, pred_max_prob, pred_max_prob.numel())
                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                if doPrint:
                    print(type(img_name), type(pred), confidence_score)
                    print(f'{str(img_name):25s}\t{pred:25s}\t{confidence_score:0.4f}')
                
                img_name_list.append(f'{img_name}')
                predicted_text_list.append(f'{pred}')
                predicted_confidences_list.append(f'{confidence_score:0.4f}')
    
                #log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            #log.close()
    return img_name_list, predicted_text_list, predicted_confidences_list

class Options_single_image():
    def __init__(self, image, model):
        self.image_path = image
        self.workers = 0
        self.batch_size = 192
        self.saved_model = model
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = False
        self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.sensitive = False
        self.PAD = False
        self.Transformation = "TPS"
        self.FeatureExtraction = "ResNet"
        self.SequenceModeling = "BiLSTM"
        self.Prediction = "Attn"
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256
        self.num_gpu = torch.cuda.device_count()

    
def read_tags_in_image(image_path, cowModelPath, digitModelPath, drinkingOnly=True, yolo=True):
    classes, labels, bboxes, scores = get_cow_predictions(image_path, cowModelPath, yolo)
    toBeRead = determine_tags_to_be_read(classes, labels, bboxes, scores, drinkingOnly)
    opt = Options_single_image(image_path, digitModelPath)
    tag_indexes, preds, confs = single_img(opt, toBeRead, False)
    
    outcome = dict()
    for i in range(len(tag_indexes)):
        outcome[tag_indexes[i]] = (preds[i], confs[i])
        
    return outcome  
    
def read_tags_in_webcam_image(frame, cow_model, digitModelPath, drinkingOnly=True, yolo=True):
    classes, labels, bboxes, scores = get_cow_predictions_webcam_image(frame, cow_model, yolo)
    toBeRead = determine_tags_to_be_read(classes, labels, bboxes, scores, drinkingOnly)
    #print(toBeRead)
    opt = Options_single_image(None, digitModelPath)
    tag_indexes, preds, confs = webcam_img(opt, frame, toBeRead, False)
    
    outcome = dict()
    for i in range(len(tag_indexes)):
        outcome[tag_indexes[i]] = (preds[i], confs[i], toBeRead[i][3], toBeRead[i][4], toBeRead[i][5]) # predicted text, confidence of text, bounding box, confidence in near tag, confidence in drinking
        
    box_image = draw_boxes(bboxes, classes, labels, scores, frame, tag_indexes, preds, confs)
    
    return outcome, box_image

def read_tags_in_dir(dir_path, cowModelPath, digitModelPath, drinkingOnly=True, yolo=True):
    outcomes = get_cow_predictions_dir(dir_path, cowModelPath, yolo)
    toBeReadDict = dict()
    for img_key in outcomes:
        toBeReadDict[img_key] = determine_tags_to_be_read(outcomes[img_key][0], outcomes[img_key][1], outcomes[img_key][2], outcomes[img_key][3], drinkingOnly)
    #print(toBeReadDict)
    opt = Options_single_image(dir_path, digitModelPath)
    tag_indexes, preds, confs = multiple_images(opt, toBeReadDict, False)
    
    outcome = dict()
    for i in range(len(tag_indexes)):
        image_name, j = tag_indexes[i].split(" ")
        if image_name in outcome:
            outcome[image_name].append((j, preds[i], confs[i]))
        else:
            outcome[image_name] = [(j, preds[i], confs[i])]
        
    return outcome  
    
    
def read_tags_in_dir_to_df(dir_path, cowModelPath, digitModelPath, drinkingOnly=True, yolo=True):
    outcomes = get_cow_predictions_dir(dir_path, cowModelPath, yolo)
    toBeReadDict = dict()
    for img_key in outcomes:
        toBeReadDict[img_key] = determine_tags_to_be_read(outcomes[img_key][0], outcomes[img_key][1], outcomes[img_key][2], outcomes[img_key][3], drinkingOnly)
    #print(toBeReadDict)
    opt = Options_single_image(dir_path, digitModelPath)
    tag_indexes, preds, confs = multiple_images(opt, toBeReadDict, False)
    
    outcome = dict()
    for i in range(len(tag_indexes)):
        image_name, j = tag_indexes[i].split(" ")
        if image_name in outcome:
            outcome[image_name].append((j, preds[i], confs[i]))
        else:
            outcome[image_name] = [(j, preds[i], confs[i])]
        
    imgNames = []
    tagIndexWithinImage = []
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    tagConfidence = []
    isDrinking = []
    predictedText = []
    textConfidence = []
    for img_key in outcome:
        for i in range(len(outcome[img_key])):
            reading = outcome[img_key][i]
            #print(img_key, reading)
            #print(toBeReadDict[img_key][int(reading[0])])
            imgNames.append(img_key)
            tagIndexWithinImage.append(reading[0])
            x0.append(toBeReadDict[img_key][i][3][0])
            y0.append(toBeReadDict[img_key][i][3][1])
            x1.append(toBeReadDict[img_key][i][3][2])
            y1.append(toBeReadDict[img_key][i][3][3])
            tagConfidence.append(toBeReadDict[img_key][i][4])
            isDrinking.append(toBeReadDict[img_key][i][5])
            predictedText.append(reading[1])
            textConfidence.append(reading[2])
    
    
    return pd.DataFrame({"imgNames": imgNames, "tagIndexWithinImage": tagIndexWithinImage, "x0" : x0, "y0": y0, "x1": x1, "y1": y1, "tagConfidence": tagConfidence, "isDrinking": isDrinking, "predictedText": predictedText, "textConfidence": textConfidence})  
    
    
def read_tags_in_dir_to_csv(dir_path, cowModelPath, digitModelPath, output_path, drinkingOnly=True,  yolo=True):
    df = read_tags_in_dir_to_df(dir_path, cowModelPath, digitModelPath, drinkingOnly, yolo)
    df.to_csv(output_path)
    print("Done.")
    
    
def read_tags_in_video(video_path, cowModelPath, digitModelPath, drinkingOnly=True, yolo=True, skipFrames=16, drinkingThresh=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    mod_time = os.path.getmtime(video_path)
    
    outcome = dict()
    largeToBeRead = dict()
    for batch in get_cow_predictions_video(cap, cowModelPath, yolo, skipFrames):
        #print(batch)
        toBeReadDict = dict()
        for fnum in batch:
            #print(batch[fnum][1])
            t = determine_tags_to_be_read(batch[fnum][0], batch[fnum][1], batch[fnum][2], batch[fnum][3], drinkingOnly)
            largeToBeRead[fnum] = t
            toBeReadDict[fnum] = t
        #print(toBeReadDict)
        opt = Options_single_image(video_path, digitModelPath)
        tag_indexes, preds, confs = video_batch(opt, cap, toBeReadDict, False)
        #print(tag_indexes, preds, confs)
        
        for i in range(len(tag_indexes)):
            fnum, j = tag_indexes[i].split(" ")
            fnum = int(fnum)
            if fnum in outcome:
                outcome[fnum].append((j, preds[i], confs[i]))
            else:
                outcome[fnum] = [(j, preds[i], confs[i])]
                
        #print(outcome)
    cap.release()
    
    tags = dict() # dictionary of tag to frame pairs
    for fnum in outcome:
        for k in range(len(outcome[fnum])):
            tag = outcome[fnum][k]
            if tag[1] in tags:
                tags[tag[1]].append((fnum, tag[2], fnum * (1/fps), largeToBeRead[fnum][k][5]))
            else:
                tags[tag[1]] = [(fnum, tag[2], fnum * (1/fps), largeToBeRead[fnum][k][5])]
                
    timeForTags = dict()
    for tag in tags:
        listOfTimes = tags[tag]
        i = 0
        while(i < len(listOfTimes)):
            #print(f"i = {i}")
            totalConf = float(listOfTimes[i][1])
            totalDConf = float(listOfTimes[i][3])
            numFrames = 1
            start = listOfTimes[i][2]
            end = listOfTimes[i][2]
            while(i+1 < len(listOfTimes) and listOfTimes[i+1][2] - listOfTimes[i][2] < drinkingThresh):
                totalConf += float(listOfTimes[i+1][1])
                totalDConf += float(listOfTimes[i+1][3])
                numFrames += 1
                end = listOfTimes[i+1][2]
                i += 1
            if tag in timeForTags:
                timeForTags[tag].append((start, end, totalConf / numFrames, totalDConf / numFrames))
            else:
                timeForTags[tag] = [(start, end, totalConf / numFrames, totalDConf / numFrames)]
            i += 1
    return timeForTags




def read_tags_in_video_to_df(video_path, cowModelPath, digitModelPath, drinkingOnly=True, yolo=True, skipFrames=16, drinkingThresh=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    mod_time = os.path.getmtime(video_path)
    
    outcome = dict()
    largeToBeRead = dict()
    for batch in get_cow_predictions_video(cap, cowModelPath, yolo, skipFrames):
        #print(batch)
        toBeReadDict = dict()
        for fnum in batch:
            #print(batch[fnum][1])
            t = determine_tags_to_be_read(batch[fnum][0], batch[fnum][1], batch[fnum][2], batch[fnum][3], drinkingOnly)
            toBeReadDict[fnum] = t
            largeToBeRead[fnum] = t
        # print(toBeReadDict)
        opt = Options_single_image(video_path, digitModelPath)
        tag_indexes, preds, confs = video_batch(opt, cap, toBeReadDict, False)
        #print(tag_indexes, preds, confs)
        
        for i in range(len(tag_indexes)):
            fnum, j = tag_indexes[i].split(" ")
            fnum = int(fnum)
            if fnum in outcome:
                outcome[fnum].append((j, preds[i], confs[i]))
            else:
                outcome[fnum] = [(j, preds[i], confs[i])]
                
        #print(outcome)
    cap.release()
    
    tags = dict() # dictionary of tag to frame pairs
    for fnum in outcome:
        for k in range(len(outcome[fnum])):
            tag = outcome[fnum][k]
            if tag[1] in tags:
                tags[tag[1]].append((fnum, tag[2], fnum * (1/fps), largeToBeRead[fnum][k][5]))
            else:
                tags[tag[1]] = [(fnum, tag[2], fnum * (1/fps), largeToBeRead[fnum][k][5])]
                
    timeForTags = dict()
    for tag in tags:
        listOfTimes = tags[tag]
        i = 0
        while(i < len(listOfTimes)):
            #print(f"i = {i}")
            totalConf = float(listOfTimes[i][1])
            totalDConf = float(listOfTimes[i][3])
            numFrames = 1
            start = listOfTimes[i][2]
            end = listOfTimes[i][2]
            while(i+1 < len(listOfTimes) and listOfTimes[i+1][2] - listOfTimes[i][2] < drinkingThresh):
                totalConf += float(listOfTimes[i+1][1])
                totalDConf += float(listOfTimes[i+1][3])
                numFrames += 1
                end = listOfTimes[i+1][2]
                i += 1
            if tag in timeForTags:
                timeForTags[tag].append((start, end, totalConf / numFrames, totalDConf / numFrames))
            else:
                timeForTags[tag] = [(start, end, totalConf / numFrames, totalDConf / numFrames)]
            i += 1
        #print(timeForTags)

         
    tagNames = []
    tagIndexOfOccurrence = []
    averageTextConfidence = []
    averageDrinkingConfidence = []
    timeStart = []
    timeEnd = []
    dateStart = []
    dateEnd = []
    for tag_key in timeForTags:
        for i in range(len(timeForTags[tag_key])):
            reading = timeForTags[tag_key][i]
            tagNames.append(tag_key)
            tagIndexOfOccurrence.append(i)
            averageTextConfidence.append(reading[2])
            averageDrinkingConfidence.append(reading[3])
            timeStart.append(reading[0])
            timeEnd.append(reading[1])
            dateStart.append(datetime.datetime.fromtimestamp(mod_time + reading[0]))
            dateEnd.append(datetime.datetime.fromtimestamp(mod_time + reading[1]))
    
    #"x0" : x0, "y0": y0, "x1": x1, "y1": y1,
    return pd.DataFrame({"tagNames": tagNames, "tagIndexOfOccurrence": tagIndexOfOccurrence, "averageTextConfidence": averageTextConfidence, "averageDrinkingConfidence": averageDrinkingConfidence, "timeStart": timeStart, "timeEnd": timeEnd, "dateStart": dateStart, "dateEnd": dateEnd})  


def read_tags_in_video_to_csv(video_path, cowModelPath, digitModelPath, output_path, drinkingOnly=True, yolo=True, skipFrames=16, drinkingThresh=2):
    df = read_tags_in_video_to_df(video_path, cowModelPath, digitModelPath, drinkingOnly, yolo, skipFrames, drinkingThresh)
    df.to_csv(output_path)
    print("Done.")
    

def show_tags_in_img(image_path, cowModelPath, digitModelPath, drinkingOnly=True, yolo=True, saveFig = None):
    cow_image = cv2.imread(image_path)
    classes, labels, bboxes, scores = get_cow_predictions(image_path, cowModelPath, yolo)
    toBeRead = determine_tags_to_be_read(classes, labels, bboxes, scores, drinkingOnly)
    opt = Options_single_image(image_path, digitModelPath)
    tag_indexes, preds, confs = single_img(opt, toBeRead, False)
    
    #print(tag_indexes)
    box_image = draw_boxes(bboxes, classes, labels, scores, cow_image, tag_indexes, preds, confs)
    
    fig, ax1 = plt.subplots(1, figsize = (16,9))
    ax1.imshow(box_image)
    if saveFig:
        fig.savefig(saveFig)
#     return box_image

def _show_tags_in_img(image_path, cowModelPath, digitModelPath, drinkingOnly=True, yolo=True):
    cow_image = cv2.imread(image_path)
    classes, labels, bboxes, scores = get_cow_predictions(image_path, cowModelPath, yolo)
    toBeRead = determine_tags_to_be_read(classes, labels, bboxes, scores, drinkingOnly)
    opt = Options_single_image(image_path, digitModelPath)
    tag_indexes, preds, confs = single_img(opt, toBeRead, False)
    
    box_image = draw_boxes(bboxes, classes, labels, scores, cow_image, tag_indexes, preds, confs)
    
    return box_image

def show_tags_in_dir(dir_path, cowModelPath, digitModelPath, drinkingOnly=True, yolo=True):
    l = os.listdir(dir_path)
    l = list(filter(lambda x: "jpg" in x or "jpeg" in x or "png" in x, l))
    #print(l)
    
    global device, CLASSES
    
    boxImgs = []
    
    if len(l) >= 4:
        for i in range(4):
            image_path = l[i]
            boxImgs.append(_show_tags_in_img(dir_path + "/" + image_path, cowModelPath, digitModelPath, drinkingOnly, yolo))
    else:
        for i in range(len(l)):
            image_path = l[i]
            boxImgs.append(_show_tags_in_img(dir_path + "/" + image_path, cowModelPath, digitModelPath, drinkingOnly, yolo))
            
    fig, ax = plt.subplots(4, figsize=(16, 32))
    if len(boxImgs) >= 1:
        ax[0].imshow(boxImgs[0])
    if len(boxImgs) >= 2:
        ax[1].imshow(boxImgs[1])
    if len(boxImgs) >= 3:
        ax[2].imshow(boxImgs[2])
    if len(boxImgs) >= 4:
        ax[3].imshow(boxImgs[3])
        
    
        
def _overlap(keyStart, keyEnd, simStart, simEnd, timeSimThresh):
    if keyEnd + timeSimThresh < simStart or simEnd + timeSimThresh < keyStart:
        return False
    else:
        return True

def parse_df_to_df(df, simThresh=0.6,  timeSimThresh=1): # simThresh: How similar should the digits be to be considered similar enough, timeSimThresh: how big can the time interval be to still be considered the same tag
    # print(df.head())
    finalDict = dict()
    for tup in df.itertuples():
        notAdded = True
        # print(f"Row to look at: {tup}")
        for key in finalDict:
            # print(key)
            if SequenceMatcher(None, str(key[0]), str(tup[1])).ratio() >= simThresh and _overlap(key[2], key[3], tup[5], tup[6], timeSimThresh):
                if key[1] * len(str(key[0])) < tup[3] * len(str(tup[1])): # if confidence of old is smaller than new
                    finalDict[key].append(tup[0])
                    finalDict[(tup[1], tup[3], min(tup[5], key[2]), max(tup[6], key[3]), tup[4], min(tup[7], key[5]), max(tup[8], key[6]))] = finalDict[key]
                    finalDict.pop(key)
                    notAdded = False
                    break
                else: # if confidence of old is larger than new
                    finalDict[key].append(tup[0])
                    finalDict[(key[0], key[1], min(tup[5], key[2]), max(tup[6], key[3]), key[4], min(tup[7], key[5]), max(tup[8], key[6]))] = finalDict[key]
                    finalDict.pop(key)
                    notAdded = False
                    break
        if notAdded: # if none were found to be similar/at the same time add it separately
            finalDict[(tup[1], tup[3], tup[5], tup[6], tup[4], tup[7], tup[8])] = [tup[0]]
        # print(finalDict)
        
    tagNames = []
    alternateNames = []
    averageTextConfidence = []
    averageDrinkingConfidence = []
    timeStart = []
    timeEnd = []
    dateStart = []
    dateEnd = []
    for tup in finalDict:
        # print(tup)
        tagNames.append(tup[0])
        averageTextConfidence.append(tup[1])
        averageDrinkingConfidence.append(tup[4])
        timeStart.append(tup[2])
        timeEnd.append(tup[3])
        dateStart.append(tup[5])
        dateEnd.append(tup[6])
        others = []
        first = True
        for row_index in finalDict[tup]:
            if first:
                first = False
                continue
            others.append((df.at[row_index, "tagNames"], df.at[row_index, "averageTextConfidence"], df.at[row_index, "timeStart"], df.at[row_index, "timeEnd"]))
        alternateNames.append(others)
        
    return pd.DataFrame({"tagNames": tagNames, "averageTextConfidence": averageTextConfidence, "averageDrinkingConfidence": averageDrinkingConfidence, "timeStart": timeStart, "timeEnd": timeEnd, "dateStart": dateStart, "dateEnd": dateEnd, "alternateNames": alternateNames}) 
    
def parse_df_to_csv(df, new_csv_path, simThresh=0.6, timeSimThresh=1):
    parse_df_to_df(df, simThresh, timeSimThresh).to_csv(new_csv_path)
    print("Done.")
    
def parse_csv_to_df(csv_path, simThresh=0.6,  timeSimThresh=1):
    df = pd.read_csv(csv_path, index_col=0)
    return parse_df_to_df(df, simThresh, timeSimThresh)
    
def parse_csv_to_csv(old_csv_path, new_csv_path, simThresh=0.6, timeSimThresh=1):
    df = pd.read_csv(old_csv_path, index_col=0)
    # print(df)
    newdf = parse_df_to_df(df, simThresh, timeSimThresh)
    newdf.to_csv(new_csv_path)
    print("Done.")
    
    
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, help='path to image to read from')
    parser.add_argument('--cow_model', required=True, help="path to cow model to use")
    parser.add_argument('--digit_model', required=True, help="path to digit model to use")
    parser.add_argument('--drinking_only', default=True, help="include only drinking tags or not")

    stuff = parser.parse_args()

    print(read_tags_in_image(stff.image_path. stuff.cow_model, stuff.digit_model, stuff.drinking_only))
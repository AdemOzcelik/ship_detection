from mmdet.apis import inference_detector, init_detector
from mmengine.utils import ProgressBar, path
from mmengine.logging import print_log
from mmyolo.registry import VISUALIZERS
from mmyolo.utils.misc import get_file_list, show_data_classes
import mmcv
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
from matplotlib.patches import Rectangle
import glob
import tqdm
import cv2
import pandas as pd
import argparse
from mmyolo.utils import register_all_modules
from tabulate import tabulate
register_all_modules()
import warnings
warnings.filterwarnings("ignore")
def crop_images(im,crop_size):

    image_dict = {}
    for i in range(math.ceil(im.shape[0]/crop_size)):
        image_dict[str(i)]={}
        for j in range(math.ceil(im.shape[1]/crop_size)):
            im_1= im[crop_size*i:crop_size*(i+1),crop_size*j:crop_size*(j+1),:]
            image_dict[str(i)][str(j)]=im_1
    return image_dict

def save_results(output_path,crop_size,max_size,config,model,file_path):

    images = sorted(glob.glob(file_path+"*.png"))
    imgs = []
    annos = []
    score_thr = 0.5
    dict_file = {}
    for image in tqdm.tqdm(images,bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):

        img= Image.open(image)
        if len(img.size)==2:
            img=img.convert('RGB')
        im = np.array(img)
        result = inference_detector(model, im)
        pred_instances = result.pred_instances[
                    result.pred_instances.scores > score_thr]
        bboxes= pred_instances["bboxes"].tolist()
        
        for w in range(len(bboxes)):
            
            xmin,ymin,xmax,ymax = bboxes[w] 
            xmin= int(xmin)
            xmax= int(xmax)
            ymin= int(ymin)
            ymax= int(ymax)
            w = xmax - xmin
            h = ymax - ymin

            if w > max_size or h > max_size:
                cv2.rectangle(im, (xmin,ymin), (xmax, ymax), (0,0,255), 2)

                
        files = crop_images(im,crop_size)
        col = []
        for key,value in files.items():
            row = []
            for key1,value1 in value.items():

                file = files[key][key1]
                i=int(key)
                j=int(key1)
                try:
                    result = inference_detector(model, file)
                    pred_instances = result.pred_instances[
                                result.pred_instances.scores > score_thr]
                    bboxes= pred_instances["bboxes"].tolist()

                    for box in bboxes:

                        xmin,ymin,xmax,ymax = box
                        xmin= int(xmin + ((j)*crop_size))
                        xmax= int(xmax + ((j)*crop_size))
                        ymin= int(ymin + ((i)*crop_size))
                        ymax= int(ymax + ((i)*crop_size))
                        

                        cv2.rectangle(file, (xmin,ymin), (xmax, ymax), (0,0,255), 2)

                except:
                    a=1
                row.append(file)
            rows = np.concatenate(row, axis=1)
            col.append(rows)
        im = np.concatenate(col, axis=0)  
        
        im_id = output_path + "/" +image.split('/')[-1]
        cv2.imwrite(im_id,im)


def same_merge(x): 

    return ','.join(x)
        
def save_csv(output_path,crop_size,max_size,config,model,file_path):

    images = sorted(glob.glob(file_path + "*.png"))
    imgs = []
    annos = []
    score_thr = 0.5
    dict_file = {}
    for image in tqdm.tqdm(images,bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):

        img= Image.open(image)
        if len(img.size)==2:
            img=img.convert('RGB')
        im = np.array(img)
        result = inference_detector(model, im)
        pred_instances = result.pred_instances[
                    result.pred_instances.scores > score_thr]
        bboxes= pred_instances["bboxes"].tolist()
        
        for w in range(len(bboxes)):
            
            xmin,ymin,xmax,ymax = bboxes[w] 
            xmin= int(xmin)
            xmax= int(xmax)
            ymin= int(ymin)
            ymax= int(ymax)
            wi = xmax - xmin
            he = ymax - ymin

            if wi > max_size or he > max_size:
                
                score = round(float(pred_instances["scores"][w]),1)
                imgs.append(image.split("/")[-1])
                annos.append(str(score) + " " + str(xmin) + " " + str(ymin)+ " " +str(xmax)+ " " +str(ymax))

                
        files = crop_images(im,crop_size)
        for key,value in files.items():
            for key1,value1 in value.items():

                file = files[key][key1]
                i=int(key)
                j=int(key1)
    
                result = inference_detector(model, file)
                pred_instances = result.pred_instances[
                            result.pred_instances.scores > score_thr]
                bboxes= pred_instances["bboxes"].tolist()

                for w in range(len(bboxes)):

                    xmin,ymin,xmax,ymax = bboxes[w]
                    xmin= int(xmin + ((j)*crop_size))
                    xmax= int(xmax + ((j)*crop_size))
                    ymin= int(ymin + ((i)*crop_size))
                    ymax= int(ymax + ((i)*crop_size))

                    score = round(float(pred_instances["scores"][w]),1)
                    imgs.append(image.split("/")[-1])
                    annos.append(str(score) + " " + str(xmin) + " " + str(ymin)+ " " +str(xmax)+ " " +str(ymax))
                    
        data = {'id': imgs, 'label': annos}

        df = pd.DataFrame(data=data)
        df = pd.DataFrame(df.groupby("id").apply(lambda x: x.apply(same_merge,axis=0))["label"])
        df.to_csv(output_path + "/" + "ship.csv",index=True)

def main():
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',  default="csv", help='overlap help')
    parser.add_argument('--checkpoint',  default="work_dirs/bbox_mAP_epoch_11.pth", help='overlap help')
    parser.add_argument('--config',  default="configs/custom_dataset/yolov8_x_syncbn_fast_8xb16-500e_ship.py", help='overlap help')
    parser.add_argument('--output_path', default="output", help='output_path help')
    parser.add_argument('--file_path', default="/workspace/notebooks/mmlabs/ship_org/test/", help='file_path help')
    parser.add_argument('--crop_size', type=int, default=640, help='crop_size help')
    parser.add_argument('--max_size', type=int, default=100, help='max_size help')
    parser.add_argument('--score_thr', type=float, default=0.5, help='score_thr help')

    args = parser.parse_args()
    
    Image.MAX_IMAGE_PIXELS=None
    
    print(tabulate([['file path', args.file_path], ['mode', args.mode],["checkpoint: ",args.checkpoint],["config:",args.config],["score_thr:",args.score_thr],["output_path :", args.output_path],["crop_size:",args.crop_size],["max_size :",args.max_size]], headers=["parameters"],tablefmt='grid'))
    
    if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)    
          
    device = "cuda:0"
    model = init_detector(args.config, args.checkpoint, device=device, cfg_options=None)
    dataset_classes = model.dataset_meta.get('classes')

    if args.mode=="csv":
        save_csv(args.output_path,args.crop_size,args.max_size,args.config,model,args.file_path)

    if args.mode=="png":
        save_results(args.output_path,args.crop_size,args.max_size,args.config,model,args.file_path)

    print("done!!")
    
if __name__ == "__main__":
    main()

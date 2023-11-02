from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Rectangle
import glob
import json
import shutil
import tqdm
import cv2
import argparse
import os
from sklearn.model_selection import train_test_split
    
def crop_images(im,image_id,overlap,crop_size,file_path):

    for i in range(math.ceil(im.shape[0]/crop_size)):        
        for j in range(math.ceil(im.shape[1]/crop_size)):
            img= im[(crop_size*i-i*overlap):(crop_size*(i+1)-i*overlap),(crop_size*j-j*overlap):(crop_size*(j+1)-j*overlap),:]
            if img.shape[0]<crop_size and img.shape[1]<crop_size: 
                img_mask = np.zeros([crop_size-img.shape[0],img.shape[1],3],dtype=np.uint8)
                img=np.vstack((img,img_mask))
                img_mask = np.zeros([crop_size,crop_size-img.shape[1],3],dtype=np.uint8) 
                img=np.hstack((img,img_mask))

            elif img.shape[0]<crop_size:
                img_mask = np.zeros([crop_size-img.shape[0],crop_size,3],dtype=np.uint8)
                img=np.vstack((img,img_mask))

            elif img.shape[1]<crop_size:
                img_mask = np.zeros([crop_size,crop_size-img.shape[1],3],dtype=np.uint8)
                img=np.hstack((img,img_mask))
             
            plt.imsave("{}/{}_{}_{}_img.png".format(file_path,image_id,i,j),img)


def crop_masks(im,image_id,overlap,crop_size,file_path,ignored_small_obj_size,pad_size):

    for i in range(math.ceil(im.shape[0]/crop_size)):
        for j in range(math.ceil(im.shape[1]/crop_size)):
            
            img= im[(crop_size*i-i*overlap):(crop_size*(i+1)-i*overlap),(crop_size*j-j*overlap):(crop_size*(j+1)-j*overlap),:]
            
            if img.shape[0]<crop_size and img.shape[1]<crop_size: 
                img_mask = np.zeros([crop_size-img.shape[0],img.shape[1],3],dtype=np.uint8)
                img=np.vstack((img,img_mask))
                img_mask = np.zeros([crop_size,crop_size-img.shape[1],3],dtype=np.uint8) 
                img=np.hstack((img,img_mask))
                
            elif img.shape[0]<crop_size and img.shape[1]>crop_size:
                img_mask = np.zeros([crop_size-img.shape[0],crop_size,3],dtype=np.uint8)
                img=np.vstack((img,img_mask))

            elif img.shape[0]>crop_size and img.shape[1]<crop_size:
                img_mask = np.zeros([crop_size,crop_size-img.shape[1],3],dtype=np.uint8)
                img=np.hstack((img,img_mask))

            if 255 in img:
                rows,cols,_ = np.where(img==255)
                x_min=min(cols)
                x_max=max(cols)
                y_min=min(rows)
                y_max=max(rows)
                w = x_max-x_min
                h = y_max-y_min
                
                if w > ignored_small_obj_size and h>ignored_small_obj_size and x_min>pad_size and y_min >pad_size  and x_max<crop_size-pad_size and y_max<crop_size-pad_size:
                    line= str(x_min)+ " " + str(x_max)+ " " +  str(y_min)+ " " + str(y_max)
                    with open("{}/{}_{}_{}_mask.txt".format(file_path,image_id,i,j), 'a+') as f:
                        f.write(line)
                        f.write('\n')
                        
def split_train_val(imgs,anns,train_path,val_path):
    
    X_train, X_test, y_train, y_test = train_test_split( imgs, anns, test_size=0.2, random_state=42)
   
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
        
    train_datas = X_train + y_train
    val_datas = X_test+y_test

    for train_data in train_datas:
        shutil.move(train_data,train_path)
        
    for val_data in val_datas:
        shutil.move(val_data,val_path)

def create_jsons(file_path,file_type):
    
    coco_data = {}
    coco_data['categories'] = [{'id': 0, 'name': 'ship', 'supercategory': 'ship'}]
    coco_images = []
    annotations = []
    files = sorted(glob.glob(file_path + "/" + file_type + "/*.txt"))
    image_id=0
    id_=0
    for file in files:
        file_name=file.split("/")[-1].rsplit("_",1)[0]+ "_img.png"
        #print(file_name)

        img= Image.open(file.split("/")[0] + "/"+file_type+"/" + file_name)
        im = np.array(img)
        coco_images.append({'height': im.shape[0], 'width': im.shape[1], 'id': image_id, 'file_name': file_name})


        ann = open(file, 'r')
        content = ann.read()
        im_ann=content.split('\n')
        im_ann=im_ann[:-1]

        for i in range(len(im_ann)):

                xmin,xmax,ymin,ymax=im_ann[i].split(' ')

                dict_={}

                dict_['iscrowd']=0
                dict_['image_id'] = image_id
                dict_['bbox'] = [int(xmin),int(ymin),int(xmax)-int(xmin),int(ymax)-int(ymin)]
                dict_['segmentation'] = []
                dict_['category_id'] = 0
                dict_['id'] = id_
                dict_['area'] = (int(xmax)-int(xmin)) * (int(ymax)-int(ymin)) 
                annotations.append(dict_)
                id_+=1
        image_id+=1

    coco_data['images'] = coco_images
    coco_data['annotations'] = annotations
    json_object = json.dumps(coco_data, indent=4)

    with open(file_path + "/" + file_type + ".json", "w") as outfile:
        outfile.write(json_object)

def clean_files(imgs,anns):
    
    ann_ids=[]
    for ann in anns:
        ann_ids.append(ann.split("/")[-1].rsplit("_",1)[0])
    for img in imgs:
        img_id= img.split("/")[-1].rsplit("_",1)[0]
        if img_id not in ann_ids:
            os.remove(img) 
        
def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--overlap', type=int, default=100, help='overlap help')
    parser.add_argument('--crop_size', type=int, default=640, help='crop_size help')
    parser.add_argument('--file_path', default="cropped", help='file_path help')
    parser.add_argument('--train_path', default="cropped/train/", help='train_path help')
    parser.add_argument('--val_path', default="cropped/val/", help='val_path help')
    parser.add_argument('--ignored_small_obj_size', type=int, default=5, help='ignore small object if weight or height smalller than this value help')
    parser.add_argument('--max_size', type=int, default=100, help='give orginal image and bboxes if bboxes size larger than this value help')
    parser.add_argument('--pad_size',type=int, default=1, help='pad_size')
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        os.makedirs(args.file_path)
                    
    Image.MAX_IMAGE_PIXELS=None
    images=glob.glob("train/*.png")
    
    print("Cropped images and annotation files will be saved to \033[1m{}\033[0m directory".format(args.file_path))
    print("Number of images : ",len(images))
    print("overlap size: ",args.overlap)
    print("crop size:",args.crop_size)
    print("ignored small object size :", args.ignored_small_obj_size)
    print("max object size for cropping:",args.max_size)
    print("padding size :",args.pad_size)
    ann = open('annot_train.txt', 'r')
    
    content = ann.read()
    for image in tqdm.tqdm(images):
        
        img= Image.open(image)
        if len(img.size)==2:
            img=img.convert('RGB')
        im = np.array(img)

        image_id=image.split("/")[1].split('.')[0]
        crop_images(im,image_id,args.overlap,args.crop_size,args.file_path)    
        im_ann = content.split("#")[int(image_id)].split('\n')
        bboxes = im_ann[4:][:-1]
        
        for i in range(len(bboxes)):
            mask=np.zeros([im.shape[0],im.shape[1],3],dtype=np.uint8)
            xmin,ymin,xmax,ymax,_=bboxes[i].split(' ')
            xmin,ymin,xmax,ymax = int(xmin),int(ymin),int(xmax),int(ymax)
            w = xmax-xmin
            h = ymax -ymin
            if w<args.ignored_small_obj_size or h<args.ignored_small_obj_size:
                continue
            if w<args.max_size or h<args.max_size:
                mask[ymin:ymax,xmin:xmax].fill(255)
                crop_masks(mask,image_id,args.overlap,args.crop_size,args.file_path,args.ignored_small_obj_size,args.pad_size)
            else:
                mask[ymin:ymax,xmin:xmax].fill(255)
                crop_masks(mask,image_id,args.overlap,args.crop_size,args.file_path,args.ignored_small_obj_size,args.pad_size)
                im_w,im_h = img.size

                if im_w > im_h:
                    fark = im_w-im_h
                    im_mask = np.zeros([fark,im_w,3],dtype=np.uint8)
                    im=np.vstack((im,im_mask))
                    mask=np.zeros([im_w,im_w,3],dtype=np.uint8)
                if im_h > im_w:
                    fark = im_h-im_w
                    im_mask = np.zeros([im_h,fark,3],dtype=np.uint8)
                    im=np.hstack((im,im_mask))
                    mask=np.zeros([im_h,im_h,3],dtype=np.uint8)
                
                mask[ymin:ymax,xmin:xmax].fill(255) 
                mask = cv2.resize(mask,(args.crop_size,args.crop_size))
                
        
                rows,cols,_ = np.where(mask==255)
                x_min=min(cols)
                x_max=max(cols)
                y_min=min(rows)
                y_max=max(rows)
                w = xmax-xmin
                h = ymax -ymin
                
                if w > args.ignored_small_obj_size and h>args.ignored_small_obj_size and x_min>args.pad_size and y_min >args.pad_size  and x_max<args.crop_size-args.pad_size and y_max<args.crop_size-args.pad_size:
                    img = Image.fromarray(im)
                    img = img.resize((args.crop_size,args.crop_size))
                    img.save(args.file_path + "/" + image.split('/')[-1].replace(".png","_img.png"))
                    line= str(x_min)+ " " + str(x_max)+ " " +  str(y_min)+ " " + str(y_max)
                    with open("{}/{}_mask.txt".format(args.file_path,image_id), 'a+') as f:
                        f.write(line)
                        f.write('\n')
                        
    anns= glob.glob(args.file_path + "/*.txt")
    imgs= glob.glob(args.file_path + "/*.png")
    
    print("annotation counts after crop process :",len(anns))
    print("image counts after crop process :",len(imgs))
    print("Images without annotations are deleting ...")
    
    clean_files(imgs,anns)
    clean_files(anns,imgs)
    
    control image and annotations counts
    anns= sorted(glob.glob(args.file_path + "/*.txt"))
    imgs= sorted(glob.glob(args.file_path + "/*.png"))     
    print("annotation counts after cleaning :", len(anns))
    print("image counts after cleaning :",len(imgs))
    
    print("images are moving to train and val directories ...")
    split_train_val(imgs,anns,args.train_path,args.val_path)
    
    print("coco format json files are creating ...")
    json_files=["train","val"]
    for file in json_files:
        create_jsons(args.file_path,file)
        print(file+".json file created in " + args.file_path +" directory")
        
    print("done!!!")
                    
if __name__ == "__main__":
    main()

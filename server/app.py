from flask import Flask, render_template, redirect, url_for, send_from_directory, request
from flask_bootstrap import Bootstrap
from PIL import Image
from werkzeug.utils import secure_filename

import os
import numpy as np
import sys
sys.path.append("..")
import tensorflow as tf
print(sys.path)
from settings import *

# from model import SSDModel
# from model import ModelHelper
from model2 import SSDModel
from model2 import ModelHelper
from model import nms
#import matplotlib.pyplot as plt
from scipy.misc import imsave
import cv2

app = Flask(__name__)
Bootstrap(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
images_directory = os.path.join(APP_ROOT, 'images')
thumbnails_directory = os.path.join(APP_ROOT, 'thumbnails')

if not os.path.isdir(images_directory):
    os.mkdir(images_directory)
if not os.path.isdir(thumbnails_directory):
    os.mkdir(thumbnails_directory)

sess = None

# def run_inference(image, model, sess, mode, sign_map):

#     image = np.asarray(image)
#     image_orig = np.copy(image) 

#     x = model['x']
#     is_training = model['is_training']
#     preds_conf = model['preds_conf']
#     preds_loc = model['preds_loc']
#     probs = model['probs']

#     print(image.shape)
#     image = Image.fromarray(image) 
#     orig_w, orig_h = image.size

#     if NUM_CHANNELS == 1:
#         image = image.convert('L')  # 8-bit 

#     image = image.resize((IMG_W, IMG_H), Image.LANCZOS)  # high-quality downsampling filter
#     image = np.asarray(image)               # 将图像转为矩阵

#     images = np.array([image])  # create a "batch" of 1 image   #创建包含一个图像的批次矩阵
    
#     if NUM_CHANNELS == 1:
#         images = np.expand_dims(images, axis=-1)  # need extra dimension of size 1 for grayscale        # 增加一个灰度的额外维度？？？

#     # Perform object detection # keep track of duration of object detection + NMS        # 开始处理前的时间
#     preds_conf_val, preds_loc_val, probs_val = sess.run([preds_conf, preds_loc, probs], feed_dict={x: images, is_training: False})
#     # Gather class predictions and confidence values
#     y_pred_conf = preds_conf_val[0]  # batch size of 1, so just take [0]
#     y_pred_conf = y_pred_conf.astype('float32')
#     prob = probs_val[0]

#     # Gather localization predictions
#     y_pred_loc = preds_loc_val[0]

#     # Perform NMS
#     boxes = nms(y_pred_conf, y_pred_loc, prob)
#     scale = np.array([orig_w / IMG_W, orig_h / IMG_H, orig_w / IMG_W, orig_h / IMG_H])
#     if len(boxes) > 0:
#         boxes[:, :4] = boxes[:, :4] * scale

#     image = image_orig

#     for box in boxes:
#         # Get box parameters
#         box_coords = [int(round(x)) for x in box[:4]]
#         print(box_coords)
#         cls = int(box[4])
#         cls_prob = box[5]
#         image = cv2.rectangle(image, tuple(box_coords[:2]), tuple(box_coords[2:]), (0, 255, 0))
#         label_str = '%s %.3f' % (sign_map[cls], cls_prob)
#         image = cv2.putText(image, label_str, (box_coords[0], box_coords[1]), 0, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

#     return image


def run_inference(image, model, sess, mode, sign_map):
    """
    Run inference on a given image

    Arguments:
        * image: Numpy array representing a single RGB image
        * model: Dict of tensor references returned by SSDModel()
        * sess: TensorFlow session reference
        * mode: String of either "image", "video", or "demo"

    Returns:
        * Numpy array representing annotated image
    """
    # Save original image in memory

    IMG_W = 400
    IMG_H = 260
    image = np.asarray(image)
    image_orig = np.copy(image)     # 原始图像

    # Get relevant tensors
    # 获取相关Tensor
    x = model['x']
    is_training = model['is_training']
    preds_conf = model['preds_conf']
    preds_loc = model['preds_loc']
    probs = model['probs']

    # Convert image to PIL Image, resize it, convert to grayscale (if necessary), convert back to numpy array
    image = Image.fromarray(image)          # 将图像矩阵，转为图像对象
    orig_w, orig_h = image.size
    # 灰度图
    if NUM_CHANNELS == 1:
        image = image.convert('L')  # 8-bit grayscale
    
    image = image.resize((IMG_W, IMG_H), Image.LANCZOS)  # high-quality downsampling filter
    image = np.asarray(image)               # 将图像转为矩阵

    images = np.array([image])  # create a "batch" of 1 image   #创建包含一个图像的批次矩阵
    if NUM_CHANNELS == 1:
        images = np.expand_dims(images, axis=-1)  # need extra dimension of size 1 for grayscale        # 增加一个灰度的额外维度？？？

    # Perform object detection
    preds_conf_val, preds_loc_val, probs_val = sess.run([preds_conf, preds_loc, probs], feed_dict={x: images, is_training: False})
    # Gather class predictions and confidence values
    y_pred_conf = preds_conf_val[0]  # batch size of 1, so just take [0]
    y_pred_conf = y_pred_conf.astype('float32')
    prob = probs_val[0]

    # Gather localization predictions
    y_pred_loc = preds_loc_val[0]

    # Perform NMS
    boxes = nms(y_pred_conf, y_pred_loc, prob)
    # logging.info("标注矩形：{}".format(boxes))

    # Rescale boxes' coordinates back to original image's dimensions
    # Recall boxes = [[x1, y1, x2, y2, cls, cls_prob], [...], ...]
    # 将标注矩形框位置按缩放比例恢复



    scale = np.array([orig_w / IMG_W, orig_h / IMG_H, orig_w / IMG_W, orig_h / IMG_H])
    if len(boxes) > 0:
        boxes[:, :4] = boxes[:, :4] * scale
    # logging.info("标注矩形2：{}".format(boxes))

    # Draw and annotate boxes over original image, and return annotated image
    image = image_orig
    box_coords = []
    B_Coor = []
    Lable = []
    for box in boxes:
        # Get box parameters
        box_coords = [int(round(x)) for x in box[:4]]
        cls = int(box[4])
        cls_prob = box[5]

        # Annotate image
        image = cv2.rectangle(image, tuple(box_coords[:2]), tuple(box_coords[2:]), (0, 255, 0))
        label_str = '%s %.3f' % (sign_map[cls], cls_prob)
        image = cv2.putText(image, label_str, (box_coords[0], box_coords[1]), 0, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        print(tuple(box_coords[:2]))
        print(tuple (box_coords[2:]))
        print("-----------------------------------")
        #for box in box_coords:
                 #print(box)    
        B_Coor.append(tuple(box_coords[:4]))
        Lable.append(str(cls))

    return image , B_Coor, Lable


def generate_output(input_files, mode = 'demo'):
     
    # print('Running inference on %s' % input_files)
    # image_orig = np.asarray(Image.open(input_files).convert("RGB"))

    image_orig = Image.open(input_files).convert("RGB")

    image, coord, label= run_inference(image_orig, model, sess, mode, sign_map)

    head, tail = os.path.split(input_files)
    path = './images/result_%s' % tail
    #plt.imsave(path, image)
    imsave(path, image)
    return path


@app.route('/')
def index():

    path = os.path.join(os.path.abspath(__file__), 'thumbnails')
    print(path)
    thumbnail_names = os.listdir(thumbnails_directory)
    #return render_template('gallery.html', )
    return render_template('index.html',thumbnail_names=thumbnail_names)

@app.route('/result/<filename>')
def result(filename):

    path = os.path.join(os.path.abspath(__file__), 'thumbnails')
    print(path)
    thumbnail_names = os.listdir(thumbnails_directory)
    #return render_template('gallery.html', )
    return render_template('result.html',thumbnail_names=thumbnail_names, image_result= filename)

@app.route('/thumbnails/<filename>')
def thumbnails(filename):
    return send_from_directory('thumbnails', filename)

@app.route('/images/<filename>')
def images(filename):
    return send_from_directory('images', filename)

@app.route('/public/<path:filename>')
def static_files(filename):
    return send_from_directory('./public', filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        for upload in request.files.getlist('images'):
            filename = upload.filename
            # Always a good idea to secure a filename before storing it
            filename = secure_filename(filename)
            # This is to verify files are supported
            ext = os.path.splitext(filename)[1][1:].strip().lower()
            if ext in set(['jpg', 'jpeg', 'png']):
                print('File supported moving on...')
            else:
                return render_template('error.html', message='Uploaded files are not supported...')
            destination = '/'.join([images_directory, filename])
            # Save original image

            upload.save(destination)

            basewidth = 600
            
            img = Image.open(destination)
            if img.size[0] > 600:
                wpercent = (basewidth/float(img.size[0]))
                hsize = int((float(img.size[1])*float(wpercent)))
                img = img.resize((basewidth,hsize), Image.ANTIALIAS)
                img.save(destination) 
            
            print('start detection')
            output = generate_output(destination)
            print('done detection')
            img_path = output
            # Save a copy of the thumbnail image

            print(img_path)
            #image = Image.open(destination)
            image = Image.open(img_path)

            if image.mode != "RGB":
                image = image.convert("RGB")
            
            print('image_opened')
            image.thumbnail((300, 170))

            result_filename = img_path.split('/')[-1]
            image.save('/'.join([thumbnails_directory, result_filename]))
        #return redirect(url_for('index'))
        return redirect(url_for('result', filename = result_filename))
    return render_template('index.html')

if __name__ == '__main__':



    sign_map = {}
    with open('../signnames.csv', 'r') as f:
        for line in f:
            line = line[:-1]  # strip newline at the end
            sign_id, sign_name = line.split(',')
            sign_map[int(sign_id)] = sign_name
            
    sign_map[0] = 'background'  # class ID 0 reserved for background class

    sess = tf.Session()
    
    model = SSDModel()
    saver = tf.train.Saver()
    print(MODEL_SAVE_PATH)
    saver.restore(sess, "/tmp3/chunting/final/model2.ckpt100")
    

    app.run(host='0.0.0.0', threaded=True, port=os.environ.get('PORT', 8787))

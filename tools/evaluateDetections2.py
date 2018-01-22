import os
import argparse
import shutil
import matplotlib.pyplot as plt
import Image
import sys
import re
from eval_voc import voc_eval
import numpy as np

def cal_IOU(boxA, boxB):
	
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 	
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
	if(interArea > 0):
		print(interArea)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou




parser = argparse.ArgumentParser(description='Print stats on detection, given a detection file and a ground truth file.')
parser.add_argument('detectionPath', metavar='Detections.csv', type=str, help='The path to the csv-file containing the detections.')
parser.add_argument('truthPath', metavar='MyAnnotations.csv', type=str, help='The path to the csv-file containing the annotations that are the ground truth.')
parser.add_argument('-c', '--copyFP', action='store_true', help='Copy images with false positives to the falsePositive/ directory.')
parser.add_argument('-f', '--saveFP', metavar='dirname/', type=str, help='Path to the original test-images to extract the false positive patches to. If not given, they are not extracted.')
parser.add_argument('-w', '--widthHistogram', action='store_true', help='Show a histogram of the widths of true positives.')
parser.add_argument('-fn', '--falseNegativesOnly', action='store_true', help='Print only the false negatives.')
parser.add_argument('-s', '--sizeMinimum', metavar='0x0', type=str, help='Disregard any annotation smaller than the specified size. First number is width.')

args = parser.parse_args()

if not os.path.isfile(args.detectionPath):
	print("Error: The given detection file does not exist.")
	exit()
detectionFile = open(os.path.abspath(args.detectionPath), 'r')

if not os.path.isfile(args.truthPath):
	print("Error: The given annotation file does not exist.")
	exit()
annotationFile = open(os.path.abspath(args.truthPath), 'r')

if args.sizeMinimum != None and not re.match('[0-9]+x[0-9]+', args.sizeMinimum):
	print("Error: The size must be in the format 20x20, where the numbers can be any integer (regex match [0-9]+x[0-9]+. First number is width.")
	exit()
elif args.sizeMinimum != None:
	minWidth = int(args.sizeMinimum.partition('x')[0])
	minHeight = int(args.sizeMinimum.partition('x')[2])

detectionFile.readline() # Discard the header-line.
header = annotationFile.readline() # Discard the header-line.

detections = detectionFile.readlines()
annotations = annotationFile.readlines()
numAnnotations = len(annotations)

fpPath = os.path.join(os.path.dirname(args.detectionPath), 'falsePositives')
fpCropPath = os.path.join(os.path.dirname(args.detectionPath), 'fpCrops')
if args.copyFP and not os.path.exists(fpPath):
	os.mkdir(fpPath)
	
if args.saveFP and not os.path.exists(fpCropPath):
	os.mkdir(fpCropPath)
	
# Remove any annotations that are too small:
fpCount = 0
tpCount = 0
id = 0
widthsFound = []

aps = []
for cls in range(10):

 	if cls == 0:
 		continue

	rec, prec, ap = voc_eval(args.detectionPath, args.truthPath, cls, ovthresh=0.5)
	#print(rec)
	#print (prec)
	aps += [ap]
	print('AP for {} = {:.4f}'.format(cls, ap))

print('Mean AP = {:.4f}'.format(np.mean(aps)))




# for cls in range(9):

# 	for detection in detections:

# 		fields = detection.split(';')
# 		potentialAnnotations = [line for line in annotations if fields[0].strip() in line]

# 		print(potentialAnnotations)

# 		annoField = potentialAnnotations[0].split(';')

# 		gt_box = []
# 		det_box = []

# 		assert (len(annoField) - 1) % 5 ==0
# 		assert (len(fields) - 1) % 5 ==0

# 		for i in range(len(annoField)/5):
# 			box = annoField[1+ i * 5 : 1+ i * 5 +4]
# 			if(box[0]==cls)
# 				gt_box.append()

# 		for i in range(len(fields)/5):
# 			box = fields[1+ i * 5 : 1+ i * 5 +4]
# 			if(box[0]==cls)
# 				det_box.append()
# 		#true or false positives
# 		ap = cal_ap

# 		for annotation in gt_box:

# 			gt_box = [float(p) for p in annotation[1:5]]
# 			d_box = [float(p) for p in annotation[1:5]]
# 			#diffs = [abs(int(float(annFields[i+2]))-int(float(fields[i+1]))) for i in range(4)]
# 			#match = (len([x for x in diffs if x > 10]) == 0)

# 			iou = cal_IOU(gt_box, d_box)
# 			print(iou)
# 			if iou > 0.5:
# 				match = True

# 			if match:
# 				annotations.remove(annotation)
# 				break	
				
# 		if match:
# 			tpCount += 1
# 			widthsFound.append(int(fields[3])-int(fields[1]))
# 		else:
# 			fpCount += 1
# 			if not args.falseNegativesOnly:
# 				print("False positive in: %s" % fields[0])
# 			if args.copyFP:
# 				shutil.copy('../'+os.path.join(os.path.dirname(args.detectionPath) + 'resized_test_images_400x260', '%s' % fields[0].strip()), fpPath)
# 			if args.saveFP != None:
# 				im = Image.open(os.path.join(args.saveFP, fields[0]))
# 				patch = im.crop([int(x) for x in fields[1:5]])
# 				patch.save(os.path.join(fpCropPath, 'fp%d.png' % id))
# 		id += 1

# 	if len(annotations) > 0:
# 		if not args.falseNegativesOnly:	
# 			print('\n------\n\nFalse negatives:\n')
# 		sys.stdout.write(header)
# 		sys.stdout.write('\n'.join([x.strip() for x in annotations]))

# 	if not args.falseNegativesOnly:			
# 		print('\n\n------')
# 		print('Number of annotations: %d' % numAnnotations)
# 		print('------')
# 		print("True positives: %d" % tpCount)
# 		print("False positives: %d" % fpCount)
# 		print("False negatives: %d" % len(annotations))
# 		print('------')

# 	if args.widthHistogram and tpCount > 0:
# 		fig = plt.figure()
# 		ax = fig.add_subplot(111)
# 		n, bins, patches = ax.hist(widthsFound, range=(10,110), bins=20)
# 		ax.set_title('Histogram over detected sign widths in %s' % os.path.dirname(args.detectionPath))
# 		plt.gca().set_xlabel("Sign widths in pixels")
# 		plt.gca().set_ylabel("Number of detections")
# 		plt.show()


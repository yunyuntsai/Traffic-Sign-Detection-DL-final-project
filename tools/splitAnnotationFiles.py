import os
import argparse
import random

parser = argparse.ArgumentParser(description='Split an annotation file into two.')
parser.add_argument('percentage', metavar='splitPercentage', type=int, help='The percentage that will be put in split1.csv. The rest of the lines will be put in split2.csv.')
parser.add_argument('path', metavar='filename.csv', type=str, help='The path to the csv-file containing the annotations.')
parser.add_argument('-f', '--filter', metavar='acceptedTag', type=str, help='If given, only annotations with this tag will be processed.')
parser.add_argument('-c', '--category', metavar='category', type=str, help='If given, the file categories.txt will be loaded and only signs with a tag that belongs to the given category are processed. categories.txt should be formatted with one category on each line in the format categoryName: tag1[, tag2, ... tagN]. It must be placed in the working directory.')
parser.add_argument('-p', '--prefix', metavar='fileNamePrefix', type=str, help='If given, this string will be put in front of the output filenames: fileNamePrefix-split1.csv and fileNamePrefix-split2.csv')


args = parser.parse_args()

if not os.path.isfile(args.path):
	print("Error: The given annotation file does not exist.\nSee annotateVisually.py -h for more info.")
	exit()
	
if args.category != None and not os.path.isfile('categories.txt'):
	print("Error: A category was given, but categories.txt does not exist in the working directory.\nTo use this functionality, create the file with a line for each category in the format\ncategoryName: tag1[, tag2, ... tagN]")
	exit()
	
if not 0 < args.percentage < 100:
	print("Error: The split percentage must be more than 0 and less than 100.")
	exit()

categories = {}
if args.category != None:
	categories = {k.split(':')[0] : [tag.strip() for tag in k.split(':')[1].split(',')] for k in open('categories.txt', 'r').readlines()}
	if args.category not in categories:
		print("Error: The category '%s' does not exist in categories.txt." % args.category)
		exit()

csv = open(os.path.abspath(args.path), 'r')
header = csv.readline()
allAnnotations = []

for line in csv:
	fields = line.split(";")
	
	if args.filter != None and args.filter != fields[1]:
		continue
		
	if args.category != None and fields[1] not in categories[args.category]:
		continue
		
	allAnnotations.append(line)

random.shuffle(allAnnotations)
splitPosition = int(round(args.percentage/100.0*len(allAnnotations)))
split = [allAnnotations[0:splitPosition], allAnnotations[splitPosition:]]

basePath = os.path.dirname(args.path)
outNames = ['split1.csv', 'split2.csv']
if args.prefix != None:
	outNames = ['%s-%s' % (args.prefix, name) for name in outNames]
out = [open(os.path.join(basePath, name), 'w') for name in outNames]

out[0].write(header)
out[0].writelines(split[0])
out[1].write(header)
out[1].writelines(split[1])

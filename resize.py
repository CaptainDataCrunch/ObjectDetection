from PIL import Image
import argparse
import scipy.misc as sm
import glob

def parseArgument():
	"""
	Code for parsing arguments
	"""
	parser = argparse.ArgumentParser(description='Parsing a file.')
	parser.add_argument('--image_directory', nargs=1, required=True)
	parser.add_argument('--class', nargs=1, required=True)
	parser.add_argument('--size', nargs=1, required=True)
	args = vars(parser.parse_args())
	return args

def img_to_matrix(filename, STANDARD_SIZE, verbose=False):
	"""
	takes a filename and turns it into a numpy array of RGB pixels
	"""
	img = Image.open(filename)
	if verbose==True:
		print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
	img = img.resize(STANDARD_SIZE)
	# img = list(img.getdata())
	# img = map(list, img)
	# img = np.array(img)
	return img

def get_all_images(directory):
	files = glob.glob(directory+"*.jpg")
	return files

def main():
	args = parseArgument()
	directory = args['image_directory'][0]
	classification = args['class'][0]
	size = args['size'][0].split(",")
	STANDARD_SIZE = (int(size[0]), int(size[1]))
	files = get_all_images(directory)
	for i in range(len(files)):
		output = img_to_matrix(files[i], STANDARD_SIZE)
		sm.imsave(classification + '_' + str(i) + '.jpg', output)

if __name__ == '__main__':
	main()

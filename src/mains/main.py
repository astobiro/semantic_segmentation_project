from generators.dataset_prep import Dataprep
from generators.dataset import Dataset
from generators.data_loader import Dataloder
from utils.utils import Params
from models.segmentation_model import SegmentationModel
import sys
SRC_ABSOLUTE_PATH = "."
sys.path.append(SRC_ABSOLUTE_PATH)

def main():
	args = sys.argv
	print(args[1])
	
	try:
		# Capture the command line arguments from the interface script.
		args = sys.argv
		print(args)
		# Parse the configuration parameters for the ConvNet Model.
		config = args[1]

	except:
		print( 'Missing or invalid arguments !' )
		exit(0)

	# Load the dataset from the library, process and print its details.
	dataset = Dataprep(config)
	dataset.showFirstImage()
	# Construct, compile, train and evaluate the ConvNet Model.
	model = SegmentationModel(dataset.config, dataset)
	#print(model.savepath)

	model.define_model()

	model.fit_model()

	model.evaluate_model()

	model.iou_calc_save()
if __name__ == '__main__':
	main()

import numpy as np
SMOOTH = 1e-5

def round_if_needed_np(x, threshold):
	"""round mask using a threshold value

	Args:
		x (numpy array): mask
		threshold (fload): threshold value for pixel selection

	Returns:
		float: rounded mask array
	"""
	threshold = np.ones_like(x)*threshold
	if threshold is not None:
		x = np.greater(x, threshold)
		x = x.astype(float)
		#x = ..cast(x, backend.floatx())
	return x

def compute_iou_per_class(gt, pr, class_weights=1., threshold=0.5, smooth=SMOOTH, per_image=True):
	'''Method to compute IoU metric evaluation per class'''
	numClasses = gt.shape[3]
	pr = round_if_needed_np(pr,threshold)
	a = gt*pr
	b = gt+pr
	ious = np.zeros((1,numClasses))
	for i in range(0,numClasses):
		intersection = np.sum(a[...,i])
		union = np.sum(b[...,i]) - intersection
		iou = (intersection + smooth) / (union + smooth)
		ious[0,i] = iou
	return ious

def compute_iou_per_structure(model,test_generator,params=0):
	"""compute the IoU values per structure
	
	Args:
		model (model): semantic segmentation model
		test_generator (DataGenerator): data generator
		params (params): params configuration file

	Returns:
		numpy array: (NxS), N = number of images, S = number of structures
	"""
	ious = np.zeros((len(test_generator),model.output_shape[-1]))

	for i in range(len(test_generator)):
		image,gt_mask = test_generator.__getitem__(i)
		pr_mask = model.predict(image)
		iou = compute_iou_per_class(gt_mask, pr_mask, threshold=0.5, class_weights=1.)
		ious[i,:] = iou

	return ious
import segmentation_models as sm
import sys
SRC_ABSOLUTE_PATH = "."
sys.path.append(SRC_ABSOLUTE_PATH)
from generators.data_loader import Dataloder
from generators.dataset import Dataset
from utils.iou_calc import compute_iou_per_structure
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas  as pd
import numpy   as np
import nibabel as nib
import matplotlib.pyplot as plt
import time

class SegmentationModel:
	def __init__(self, config, dataset):
		self.config = config
		self.resultpath = "data/results/" + self.config.TESTNO + "/"
		self.dataset = dataset
		self.metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
		self.optim = keras.optimizers.Adam(self.config.LR)
		self.preprocess_input = self.preprocess_inputInit()
		self.model = self.modelInit()
		self.dice_loss = sm.losses.DiceLoss()
		self.focal_loss = self.focal_lossInit()
		self.total_loss = self.dice_loss + (1*self.focal_loss)
		self.callbacks = self.callbacksInit()
		return

	def modelInit(self):
		#Initializes model variable
		n_classes = 1 if len(self.config.CLASSES) == 1 else (len(self.config.CLASSES) + 1)
		#Loads Unet model using BACKBONE in config.json and number of classes based on CLASSES, with activation type in config.json
		model = sm.Unet(self.config.BACKBONE, classes=n_classes, activation=self.config.ACTIVATION)
		return model

	def preprocess_inputInit(self):
		#Initializes preprocess_input variable
		#Loads preprocessing based on BACKBONE
		preprocess_input = sm.get_preprocessing(self.config.BACKBONE)
		return preprocess_input

	def focal_lossInit(self):
		#Initializes focal_loss variable
		n_classes = 1 if len(self.config.CLASSES) == 1 else (len(self.config.CLASSES) + 1)
		#Sets type to binary for 1 class only and to categorical when there is more than one class
		focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
		return focal_loss

	def callbacksInit(self):
		#Initializes callbacks variable
		#sets history file name saved on resultpath
		csv_history_file = self.resultpath + "training_history_log.csv"
		#Callbacks for training, early stopping, checkpoints saving the best model on best_model.h5
		callbacks = [
			keras.callbacks.EarlyStopping(monitor='val_loss',patience=14),
			keras.callbacks.ModelCheckpoint(self.resultpath + 'best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
			keras.callbacks.ReduceLROnPlateau(patience=7),
			keras.callbacks.CSVLogger(csv_history_file)
		]
		return callbacks

	def define_model(self):
		#Funcion to define or initialize the model with the dataset and metrics
		n_classes = 1 if len(self.config.CLASSES) == 1 else (len(self.config.CLASSES) + 1)
		#Dataset for training images
		self.train_dataset = Dataset(
		    self.dataset.images_loc, 
		    self.dataset.mask_loc,
		    self.dataset.train_dataframe, 
		    classes=self.config.CLASSES, 
		    #augmentation=get_training_augmentation(),
		    #preprocessing=get_preprocessing(preprocess_input),
		)

		# Dataset for validation images
		self.valid_dataset = Dataset(
		    self.dataset.images_loc, 
		    self.dataset.mask_loc, 
		    self.dataset.valid_dataframe,
		    classes=self.config.CLASSES, 
		    #augmentation=get_validation_augmentation(),
		    #preprocessing=get_preprocessing(preprocess_input),
		)
		#Dataloader for training images
		self.train_dataloader = Dataloder(self.train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
		#Dataloader for validation images
		self.valid_dataloader = Dataloder(self.valid_dataset, batch_size=1, shuffle=False)

		print(self.train_dataloader[0][0].shape)
		print(self.train_dataloader[0][1].shape)
		#Checks if the dataloader is all right
		assert self.train_dataloader[0][0].shape == (self.config.BATCH_SIZE, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3)
		assert self.train_dataloader[0][1].shape == (self.config.BATCH_SIZE, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, n_classes)

		self.model.compile(self.optim, self.total_loss, self.metrics)

		self.test_dataset = Dataset(
			self.dataset.images_loc, 
			self.dataset.mask_loc,
			self.dataset.test_dataframe, 
			classes=self.config.CLASSES, 
			#augmentation=get_validation_augmentation(),
			#preprocessing=get_preprocessing(self.preprocess_input),
		)

		self.test_dataloader = Dataloder(self.test_dataset, batch_size=1, shuffle=False)
		return

	def fit_model(self):
		start_time = time.time()
		self.history = self.model.fit_generator(
			self.train_dataloader, 
		    steps_per_epoch=len(self.train_dataloader), 
		    epochs=self.config.EPOCHS, 
		    callbacks=self.callbacks, 
		    validation_data=self.valid_dataloader, 
		    validation_steps=len(self.valid_dataloader),
		)
		end_time = time.time()
		self.train_time = end_time - start_time
		print( "The model took %0.3f seconds to train.\n"%self.train_time )

		# Plot training & validation iou_score values
		plt.figure(figsize=(30, 5))
		plt.subplot(121)
		plt.plot(self.history.history['iou_score'])
		plt.plot(self.history.history['val_iou_score'])
		plt.title('Model iou_score')
		plt.ylabel('iou_score')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')

		# Plot training & validation loss values
		plt.subplot(122)
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(self.resultpath + "model_training_matrix.png")

		return

	def evaluate_model(self):
		self.scores = self.model.evaluate_generator(self.test_dataloader)

		print("Loss: {:.5}".format(self.scores[0]))
		loss = self.scores[0]
		means = []
		for metric, value in zip(self.metrics, self.scores[1:]):
		    print("mean {}: {:.5}".format(metric.__name__, value))
		    means.append([metric.__name__, value])
		score = pd.DataFrame(data=[loss, means[0][1], means[1][1]], columns=["Loss", means[0][0], means[1][0]])
		score.to_csv(self.resultpath + "loss-mean_values.csv")

	def iou_calc_save(self):
		ious = compute_iou_per_structure(self.model, self.test_dataloader)
		df = pd.DataFrame(data=ious)
		df.to_csv(self.resultpath + "iou_log.csv")

	def save_image_results(self):
		ids = np.arange(len(test_dataset))
		for i in ids:
			image, gt_mask = test_dataset[i]
			image = np.expand_dims(image, axis=0)
			pr_mask = model.predict(image).round()

			cv2.imwrite(self.resultpath + "raw_" + self.test_dataset.ids[i], image)
			cv2.imwrite(self.resultpath + "gt_mask_" + self.test_dataset.ids[i], gt_mask)
			cv2.imwrite(self.resultpath + "pr_mask_" + self.test_dataset.ids[i], pr_mask)
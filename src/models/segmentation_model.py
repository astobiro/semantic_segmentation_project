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
		self.resultpath = self.check_results_path()
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

	def check_results_path(self):
		resultpath = "data/results/" + self.config.TESTNO + "/"
		if not os.path.exists(resultpath):
			os.mkdir(resultpath)
		return resultpath

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
		#Compiles model with chosen optimizer, loss and metrics
		self.model.compile(self.optim, self.total_loss, self.metrics)
		#Dataset for test images
		self.test_dataset = Dataset(
			self.dataset.images_loc, 
			self.dataset.mask_loc,
			self.dataset.test_dataframe, 
			classes=self.config.CLASSES, 
			#augmentation=get_validation_augmentation(),
			#preprocessing=get_preprocessing(self.preprocess_input),
		)
		#Dataloader for test images
		self.test_dataloader = Dataloder(self.test_dataset, batch_size=1, shuffle=False)
		return

	def fit_model(self):
		#Function to train model
		#Start time to get time to train at the end of training
		start_time = time.time()
		#Fits model and saves history
		self.history = self.model.fit_generator(
			self.train_dataloader, 
		    steps_per_epoch=len(self.train_dataloader), 
		    epochs=self.config.EPOCHS, 
		    callbacks=self.callbacks, 
		    validation_data=self.valid_dataloader, 
		    validation_steps=len(self.valid_dataloader),
		)
		#End time to get time to train at the end of training
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
		#Function to calculate scores on the test dataset
		self.scores = self.model.evaluate_generator(self.test_dataloader)
		#Total loss
		print("Loss: {:.5}".format(self.scores[0]))
		loss = self.scores[0]
		means = []
		#Gets mean metric values
		for metric, value in zip(self.metrics, self.scores[1:]):
		    print("mean {}: {:.5}".format(metric.__name__, value))
		    means.append([metric.__name__, value])
		#Saves values on a csv

		score = pd.DataFrame([[loss, means[0][1], means[1][1]]], columns=["Loss", means[0][0], means[1][0]])
		score.to_csv(self.resultpath + "loss-mean_values.csv")

	def iou_calc_save(self):
		#Function to calculate IOU for every image and save the results on a csv
		ious = compute_iou_per_structure(self.model, self.test_dataloader)
		df = pd.DataFrame(data=ious)
		df.to_csv(self.resultpath + "iou_log.csv")

	def save_image_results(self):
		#Function to save every image from test dataset and its predicted mask
		ids = np.arange(len(self.test_dataset))
		SAVE_DIR = self.resultpath + "test_results/"
		if not os.path.exists(SAVE_DIR):
			os.mkdir(SAVE_DIR)
		input()
		for i in ids:
			image, gt_mask = self.test_dataset[i]
			image = np.expand_dims(image, axis=0)
			pr_mask = model.predict(image).round()
			plt.figure()
			plt.subplot(1,2,1)
			plt.imshow(image, cmap='bone')
			plt.subplot(1,2,2)
			plt.imshow(image, cmap='bone')
			plt.imshow(pr_mask, alpha=0.5, cmap='nipy_spectral')
			plt.savefig(SAVE_DIR + "result_" + self.test_dataset.ids[i])

	def load_best_results(self):
		self.model.load_weights(self.resultpath + "best_model.h5")
		self.model.compile(self.optim, self.total_loss, self.metrics)

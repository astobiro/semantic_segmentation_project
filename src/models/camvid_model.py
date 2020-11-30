import segmentation_models as sm
import sys
SRC_ABSOLUTE_PATH = "."
sys.path.append(SRC_ABSOLUTE_PATH)
from generators.data_loader import Dataloder
from generators.dataset import Dataset
from utils import utils
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

class CamVidModel:
	def __init__(self, config, dataset):
		self.config = config
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
		n_classes = 1 if len(self.config.CLASSES) == 1 else (len(self.config.CLASSES) + 1)
		model = sm.Unet(self.config.BACKBONE, classes=n_classes, activation=self.config.ACTIVATION)
		return model
	def preprocess_inputInit(self):
		preprocess_input = sm.get_preprocessing(self.config.BACKBONE)
		return preprocess_input
	def focal_lossInit(self):
		n_classes = 1 if len(self.config.CLASSES) == 1 else (len(self.config.CLASSES) + 1)
		focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
		return focal_loss
	def callbacksInit(self):
		# Callbacks for training
	    callbacks = [
	        keras.callbacks.EarlyStopping(monitor='val_loss',patience=14),
	        keras.callbacks.ModelCheckpoint('best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
	        keras.callbacks.ReduceLROnPlateau(patience=7)
	    ]
	    return callbacks

	def define_model(self):
		n_classes = 1 if len(self.config.CLASSES) == 1 else (len(self.config.CLASSES) + 1)
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
		self.train_dataloader = Dataloder(self.train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
		self.valid_dataloader = Dataloder(self.valid_dataset, batch_size=1, shuffle=False)

		print(self.train_dataloader[0][0].shape)
		print(self.train_dataloader[0][1].shape)

		assert self.train_dataloader[0][0].shape == (self.config.BATCH_SIZE, 512, 512, 3)
		assert self.train_dataloader[0][1].shape == (self.config.BATCH_SIZE, 512, 512, n_classes)

		self.metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
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
		plt.legend(['Train', 'Test'], loc='upper left')

		# Plot training & validation loss values
		plt.subplot(122)
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.savefig("model_training_matrix.png")

		return

	def evaluate_model(self):
		self.scores = self.model.evaluate_generator(self.test_dataloader)

		print("Loss: {:.5}".format(self.scores[0]))
		for metric, value in zip(self.metrics, self.scores[1:]):
		    print("mean {}: {:.5}".format(metric.__name__, value))
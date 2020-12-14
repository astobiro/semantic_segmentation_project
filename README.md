# semantic_segmentation_project / Sistema para segmentação de estruturas de interesse em imagens médicas baseado em deep learning
Semantic segmentation of covid 19 images found in https://www.kaggle.com/andrewmvd/covid19-ct-scans

# Dependencies:
To install necessary library run:
```python
  pip install -U albumentations>=0.3.0
  pip install -U --pre segmentation-models
  pip install keras==2.3.1
  pip install tensorflow==2.1.0
  pip install keras_applications==1.0.8
  pip install image-classifiers==1.0.0
  pip install efficientnet==1.0.0
```

After cloning the repo, just run a command line from the root directory (folder semantic_segmentation_project). Run the command: python src/main.py config.json the code that will execute will train the net with parameters of the file config.json. The config.json file on the repo is this one:
```json
{
    "BACKBONE" : "efficientnetb3",
    "BATCH_SIZE" : 4,
    "CLASSES" : ["infection"],
    "LR" :  0.0001,
    "EPOCHS" : 40,
    "ACTIVATION" : "sigmoid",
    "imgfolder": "data/datasets/dataset2D-384x384/lungs",
    "maskfolder" : "data/datasets/dataset2D-384x384/masks",
    "train_dataframepath" :"data/datasets/dataset2D-384x384/n1_train_dataframe.csv",
    "test_dataframepath" :"data/datasets/dataset2D-384x384/n1_test_dataframe.csv",
    "valid_dataframepath" :"data/datasets/dataset2D-384x384/n1_valid_dataframe.csv",
    "TESTNO" : "n1",
    "IMAGE_SIZE" : 384,
    "load" : "n"
}
```

To create a custom config file just modify this file or create another file with all of these parameters. The parameters are self explanatory except the load parameter. This parameter has 2 states n and y, when it is n the program will perform training from 0. When it is y the program will read the weight file best_weights.h5 found in the folder data/results/TESTNO (TESTNO being the parameter within config.json). That file is the best weights from the previous training. The program will then load these weights into the model.

To use a different dataset, it is necessary to put the images without masks in a folder called lungs, the respective masks in another folder called masks. The corresponding image and mask files must have the same filenames.

To create the training, test and validation dataframes the csv files must have the first cell of the first column empty and then every other cell in this column counting from 0 to the max number of images. The first cell of the second column has "filename" and the other cells the image filenames. The same must be done for training, test and validation images. The csv file must be named TESTNO_train_dataframe.csv for the training file switching train for test and valid for test images and validation images respectively. The parameter IMAGE_SIZE must have the size of the image dimensions but you have to modify the segmentation_model.py if you are going to use images with different height and width. The parameter classes must also be modified for your case.

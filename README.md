# TCC_deepvideos


### How to Train:
- Set up hyperparameters inside **"run_train.py"** file.
- Run **```python run_train.py```**
- Results and training checkpoint will be saved at **"results/prints_unet3d"**.
- In order to retrain, you must check the sample number the last checkpoint was at (check at **"results/prints_unet3d/weights"**), and type it when asked on terminal.

### How to Evaluate:
- Set up hyperparameters inside **"run_evaluation.py"** file.
-- You must specify the correct name of the weight file (.pth) and its model state number.
- Run ```**python run_evaluation.py**```
- A **.csv** file will be generated inside the folder specified at variable RUN_PATH, containing the calculation of the PSNR and SSIM metrics of each sample.

### How to Test:
- Set up hyperparameters inside **"val_model.py"** file.
-- You must specify the correct name of the weight file (.pth) and its model state number.
- Run **```python val_model.py```**
- The output images will be saved at SAVE_IMAGES_PATH hyperparameter.
- Each image will contain 5 images: 

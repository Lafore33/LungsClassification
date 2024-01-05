# Lungs Classification
The task is to classify X-ray images of lungs. There are 3 classes: 0 - normal, 1 - non-covid, 2 - covid 19.
For the task I used private datasets. Train dataset contains X-ray images of lungs and binary masks for this images. Test Dataset contains only X-ray images of lungs.
Train Dataset was split into train and validaation datasets for training the model. Test Dataset was used to see how model works with new data. 
datasets.py is a file with custom dataset classes; 
model.py contains the model itself; 
graphs.py shows losses on test and train; 
metrics.py transoforms predictictions and true values, so f1 can be calculated; 
run.py shows one epoch; 
predictictions.py is where model gives actual predictions; 
training.py is a file with training/test cycle;

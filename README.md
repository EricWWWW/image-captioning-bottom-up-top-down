#  Image captioning using Bottom-up, Top-down Attention

This is a PyTorch implementation of [ Bottom-up and Top-down Attention for Image Captioning](http://www.panderson.me/up-down-attention). Training and evaluation is done on the MSCOCO Image captioning challenge dataset. Bottom up features for MSCOCO dataset are extracted using Faster R-CNN object detection model trained on Visual Genome dataset. Pretrained bottom-up features are downloaded from [here](https://github.com/peteanderson80/bottom-up-attention). Modifications made to the original model:

*   ReLU activation instead of Tanh gate in Attention model
*   Discriminative supervision in addition to cross-entropy loss
## Difference between the original repo
- add Chinese annotations
- Update pytorch version to 1.4,so you and run it in new version
- add new metric: SPCIE and WMD

##  Results obtained 

<table class="tg">
  <tr>
    <th>Model</th>
    <th>BLEU-4</th>
    <th>METEOR</th>
    <th>ROUGE-L</th>
    <th>CIDEr</th>
  </tr>
  <tr>
    <td>[This implementation](https://drive.google.com/file/d/10atC8rY7PdhnKW08INO33mEXYUyQ6G0N/view?usp=sharing)</td>
    <td>35.9</td>
    <td>26.9</td>
    <td>56.2</td>
    <td>111.5</td>
  </tr>
  <tr>
    <td>Original paper implementation</td>
    <td>36.2</td>
    <td>27.0</td>
    <td>56.4</td>
    <td>113.5</td>
    </tr>
</table>

Results reported on Karpathy test split. Pretrained model can be downloaded by clicking on the link above.

##  Requirements 

python 3.6

pytorch 1.0+

h5py 2.8

tqdm 4.26

nltk 3.3

##  Data preparation 

Create a folder called 'data'

Create a folder called 'final_dataset'

Download the MSCOCO [Training](http://images.cocodataset.org/zips/train2014.zip) (13GB)  and [Validation](http://images.cocodataset.org/zips/val2014.zip) (6GB)  images. 

Also download Andrej Karpathy's [training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). This zip file contains the captions.

Unzip all files and place the folders in 'data' folder.

Next, download the [bottom up image features](https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip).

Unzip the folder and place unzipped folder in 'bottom-up_features' folder.  

Next type this command in a python 2 environment: 
```bash
python bottom-up_features/tsv.py
```

This command will create the following files - 

*   An HDF5 file containing the bottom up image features for train and val splits, 36 per image for each split, in an I, 36, 2048 tensor where I is the number of images in the split.
*   PKL files that contain training and validation image IDs mapping to index in HDF5 dataset created above.

Move these files to the folder 'final_dataset'.

Next, type this command: 
```bash
python create_input_files.py
```

This command will create the following files - 

*   A JSON file for each split containing the order in which to load the bottom up image features so that they are in lockstep with the captions loaded by the dataloader.
*   A JSON file for each split with a list of N_c * I encoded captions, where N_c is the number of captions sampled per image. These captions are in the same order as the images in the HDF5 file. Therefore, the ith caption will correspond to the i // N_cth image.
*   A JSON file for each split with a list of N_c * I caption lengths. The ith value is the length of the ith caption, which corresponds to the i // N_cth image.
*   A JSON file which contains the word_map, the word-to-index dictionary.

Next, go to nlg_eval_master folder and type the following two commands:
```bash
pip install -e .
nlg-eval --setup
```
This will install all the files needed for evaluation.

##  Training 

To train the bottom-up top down model from scratch, type:
```bash
python train.py
```

The dataset used for learning and evaluation is the MSCOCO Image captioning challenge dataset. It is split into training, validation and test sets using the popular Karpathy splits. This split contains 113,287 training images with five captions each, and 5K images respectively for validation and testing. Teacher forcing is used to aid convergence during training. Teacher forcing is a method of training sequence based tasks on recurrent neural networks by using the actual or expected output from the training dataset at the current time step y(t) as input in the next time step X(t+1), rather than the output generated by the network. Teacher forcing addresses slow convergence and instability when training recurrent networks that use model output from a prior time step as an input.

Weight normalization was found to prevent the model from overfitting and is used liberally for all fully connected layers.

Gradients are clipped during training to prevent gradient explosion that is not uncommon with LSTMs. The attention dimensions, word embedding dimension and hidden dimensions of the LSTMs are set to 1024.

Dropout is set to 0.5. Batch size is set to 100. 36 pretrained bottom-up feature maps per image are used as input to the Top-down Attention model. The Adamax optimizer is used with a learning rate of 2e-3. Early stopping is employed if the BLEU-4 score of the validation set shows no improvement over 20 epochs.

##  Evaluation 

To evaluate the model on the karpathy test split, download this [repo](https://github.com/EricWWWW/image-caption-metrics) and config it follow the `README.md`.When finishing,copy the folder `pycocoevalcap` to the root dir of this repo.Then edit the eval.py file to include the model checkpoint location and the `eval.py`

Beam search is used to generate captions during evaluation. Beam search iteratively considers the set of the k best sentences up to time t as candidates to generate sentences of size t + 1, and keeps only the resulting best k of them. A beam search of five is used for inference.

The metrics reported are ones used most often in relation to image captioning and include BLEU-4, CIDEr, METEOR, ROUGE-L, SPICE and WMD. Official MSCOCO evaluation scripts are used for measuring these scores.

## References

Code adapted with thanks from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

Evaluation code adapted from https://github.com/Maluuba/nlg-eval/tree/master/nlgeval

Tips for improving model performance and code for converting bottom-up features tsv file to hdf5 files sourced from https://github.com/hengyuan-hu/bottom-up-attention-vqa

https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/

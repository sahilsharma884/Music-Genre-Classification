# Music-Genre-Classification (GTZAN Dataset)

# Prerequisite
- Python 3.6.8
- CUDA 9.0 (Follow these: https://blog.quantinsti.com/install-tensorflow-gpu/ )
- Sublime Text (Optional)

# Package Required
- **numpy** (For Numerical Computation)
- **librosa** (For dealing with Audios)
- **tqdm** (For showing loading/progrss bar)
- **sklearn** (For Splitting the data into train, valid and test, and Confusion Matrix)
- **keras** (For Deep learning: CNN and VGG16)
- **matplotlib** (For showing the graph of train and valid with loss and accuracy)
- **collections** (For storing the genres corresponding names to showing in confusion matrix)
- **itertools** (For iterating the elements)
- **pickle** (For storing the data into hard drive, so that we don't need to compute again and again. Just call the pickle file and load)

# System Requirement
- 16GB RAM
- Nvidia 4GB RAM

# Dataset
GTZAN Genre Collection (Download Link: http://marsyas.info/downloads/datasets.html )

# Steps of Execution:
1) **Read_File.py** : This file perform reading the audio files with labels corresponding to number of genres.

2) **Audio_Segment.py** : This file perform clipping of audio into small segment/clip of duration depending on window size and overlap.

3) **Feature_Extraction.py** : This file contain feature extraction techniques (STFT, Melspectrogram and MFCC). Whichever you want to extract, just change the name of the function 'to_stft' instead.

4) **Split_Data.py** : This file perform splitting of data into train, valid and test data.

5) **CNN_Model.py** : This file contain CNN model. (Also include CNN + RNN in comment section) So that you can train with RNN if required.

6) **CNN_BiDirectional.py** : This file is for CNN+BiRNN Model.

7) **VGG16_Model.py** : This file contain VGG16 model. (Also include VGG16 + RNN in comment section) So that you can also train with RNN also if required.

8) **VGG16_BiDirectional.py** : This file is for VGG16+BiRNN Model.

***Other files are just for EDA:***

**Waveform.py** : To show the plot of wave for the audio

**Plot_Audio.py** : To plot the spectogram of segments of the audio. (You can uncomment the code if you want to see STFT, Melspectrogram or MFCC)

**Plot_CM.py** : Create a module to print the Confusion Matrix

# Result
**See the result with every model with different feature extraction in 'Result and Output' Folder**
  - train, valid and test loss and accuracy 
  - test confusion matrix
  
# Description
You can study the description/thesis about this project in 'Major Thesis_(Final).pdf'

# Publication
Faiyaz Ahmad, Sahil,'Music Genre Classification using Spectral Analysis Techniques With Hybrid Convolution-Recurrent Neural Network',International Journal of Innovative Technology and Exploring Engineering (IJITEE), Volume-9, Issue-1, Novemeber, 2019
Link: https://www.ijitee.org/wp-content/uploads/papers/v9i1/A3956119119.pdf

import os
import librosa
import tqdm
import numpy

genre_directory = 'genres'
genre_label = {}
label_value = 0

'''
@description: Assigning the folder string as label value
'''
for folder in os.listdir(genre_directory):
	genre_label[folder] = label_value
	label_value += 1

'''
genre_label = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
'''

X = []
y = []
'''
@description: Read the discrete value from songs with assigning labels
'''
n_rates = 660000
for folder in os.listdir(genre_directory):
	print('Processing folder {0}'.format(folder))
	for files in tqdm.tqdm(os.listdir(genre_directory+'/'+folder)):
		data, sr = librosa.load(genre_directory+'/'+folder+'/'+files)
		data = data[:n_rates]
		X.append(data)
		y.append(genre_label[folder])
print('Done!')

print('Saving')
numpy.save('X_Data.npy',X)
numpy.save('y_Label.npy',y)
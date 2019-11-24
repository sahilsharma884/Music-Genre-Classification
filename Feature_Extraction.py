import numpy
import librosa
import tqdm

def to_stft(X):	# -1,513,129,1
	D = lambda x: librosa.stft(x, n_fft=1024, hop_length=512)[:,:,numpy.newaxis]
	f = map(D,X)
	return numpy.array(numpy.abs(list(f)))

def to_melspectrogram(X):	# -1,128,129,1
	D = lambda x: librosa.feature.melspectrogram(x, n_fft=1024, hop_length=512)[:,:,numpy.newaxis]
	f = map(D,X)
	return numpy.array(list(f))

def to_mfcc(X):
	D = lambda x: librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(x, n_fft=1024, hop_length=512)), n_mfcc=13)[:,:,numpy.newaxis]
	f = map(D,X)
	return numpy.array(list(f))

print('Data Features')
X = numpy.load('X_SegData.npy')
y = numpy.load('y_SegLabel.npy')
print(X.shape)

X_feat = []
y_feat = []
for i in tqdm.tqdm(range(0,X.shape[0])):
	X_feat.extend(to_stft(X[i]))
	y_feat.extend(y[i])

print(numpy.array(X_feat).shape)
print(numpy.array(y_feat).shape)

print("Saving")
# numpy.save('X_MelSpec.npy',X_feat)
# numpy.save('y_MelSpec.npy',y_feat)
numpy.save('X_Feat.npy',X_feat)
numpy.save('y_Feat.npy',y_feat)
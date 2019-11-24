import numpy
import librosa.display
import matplotlib.pyplot as plt
import Feature_Extraction

X = numpy.load('X_SegData.npy')
y = numpy.load('y_Label.npy')

print(X.shape)
for i in range(0,X.shape[0],100):
	# STFT Power Spectrum
	# D = librosa.amplitude_to_db(numpy.abs(librosa.stft(X[i])), ref=numpy.max)
	# librosa.display.specshow(D, y_axis='log')
	# plt.show()

	# Waveplot Of Signal
	# librosa.display.waveplot(X[i])
	# plt.show()

	# STFT Plot
	Stft_dis = []
	D = Feature_Extraction.to_stft(X[i])
	for j in range(D.shape[0]):
		Stft_dis.extend(numpy.reshape(D[j],[-1,513,129]))
	for k in range(len(Stft_dis)):
		librosa.display.specshow(librosa.amplitude_to_db(Stft_dis[k]), y_axis='log')
		plt.show()
	break

	# MelSpectrogram
	# Mel_dis = []
	# D = Feature_Extraction.to_melspectrogram(X[i])
	# for j in range(D.shape[0]):
	# 	Mel_dis.extend(numpy.reshape(D[j],[-1,128,129]))
	# for k in range(len(Mel_dis)):
	# 	librosa.display.specshow(Mel_dis[k], y_axis='log')
	# 	plt.show()
	# break

	# Chromatogram
	# Chroma_dis = []
	# D = Feature_Extraction.to_chromagram(X[i])
	# print(D.shape)
	# for j in range(D.shape[0]):
	# 	Chroma_dis.extend(numpy.reshape(D[j],[-1,12,129]))
	# for k in range(len(Chroma_dis)):
	# 	librosa.display.specshow(Chroma_dis[k], y_axis='chroma')
	# 	plt.show()
	# break	
	
print(y.shape)
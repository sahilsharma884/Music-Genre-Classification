import numpy
import tqdm

def audio_segment(X, y, window_size=0.1, overlap=0.5):
	S_X = []
	S_y = []
	total_sample = X.shape[0]
	# print('Total samples:{0}'.format(total_sample))

	number_windows = int(total_sample * window_size)
	# print('Total windows:{0}'.format(number_windows))

	number_overlap = int(number_windows * overlap)
	# print('Total overlap:{0}'.format(number_overlap))

	for i in range(0,total_sample-number_windows+number_overlap,number_overlap):
		S_X.append(X[i:i+number_windows])
		S_y.append(y)

	return numpy.array(S_X), numpy.array(S_y)

print('Data Segment..')
X = numpy.load('X_Data.npy')
y = numpy.load('y_Label.npy')

Seg_X = []
Seg_y = []

'''
@description: Each songs is segmented with respect to windows size and overlap
'''
for i in tqdm.tqdm(range(X.shape[0])):
	data_X, data_y = audio_segment(X[i], y[i])
	Seg_X.append(data_X)
	Seg_y.append(data_y)

print('Number of Segment in each song:{0}, and with labelsL:{1}'.format(data_X.shape, data_y.shape))

numpy.save('X_SegData.npy',Seg_X)
numpy.save('y_SegLabel.npy',Seg_y)
import numpy
import librosa.display
import librosa
import matplotlib.pyplot as plt

audio_file = './genres/blues/blues.00000.au'

read_file = librosa.load(audio_file)
print(read_file[0])
librosa.display.waveplot(read_file[0])
plt.show()
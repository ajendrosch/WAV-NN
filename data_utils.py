import os
import soundfile as sf
import scipy.io.wavfile as wav
import numpy as np
from pipes import quote
#from config import nn_config


def read_wav_as_np(filename):
	data , sample_rate = sf.read(filename)
	return data, sample_rate

def write_np_as_16bit_wav(X, sample_rate, filename):
	Xnew = X * 1 #32767.0
	Xnew = Xnew.astype('int16')
	wav.write(filename, sample_rate, Xnew)
	return

def convert_wav_to_mp3_to_wav(filename, sample_frequency):
	ext = filename[-4:]
	if(ext != '.wav'):
		return
	files = filename.split('/')
	orig_filename = files[-1][0:-4]
	orig_path = filename[0:-len(files[-1])-len(files[-2])-1] # Change to new file structure
	mp3_path = ''
	mp3_path = orig_path + 'MP3'
	if not os.path.exists(mp3_path):
		os.makedirs(mp3_path)
	lqwav_path = ''
	lqwav_path = orig_path + 'LQ-WAV'
	if not os.path.exists(lqwav_path):
		os.makedirs(lqwav_path)
	mp3_name = mp3_path + '/' + orig_filename + '.mp3'
	lqwav_name = lqwav_path + '/' + orig_filename + '.wav'
	sample_freq_str = "{0:.1f}".format(float(sample_frequency)/1000.0)
	bitrate = 8
	cmd = 'lame -b {0} --resample {1} {2} {3}'.format(int(bitrate), sample_freq_str,quote(filename), quote(mp3_name))
	os.system(cmd)
	cmd = 'lame --decode {0} {1} --resample {2}'.format(quote(mp3_name), quote(lqwav_name), sample_freq_str)
	os.system(cmd)
	return

def convert_folder_to_wav(directory, sample_rate=44100):
	for file in os.listdir(directory):
		fullfilename = directory+file
		if file.endswith('.wav'):
			convert_wav_to_mp3_to_wav(filename=fullfilename, sample_frequency=sample_rate)				
	return
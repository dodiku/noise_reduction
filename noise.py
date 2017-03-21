import librosa
from pysndfx import AudioEffectsChain
import numpy as np


'''------------------------------------
FILE READER:
receives filename,
returns audio time series (y) and sampling rate of y (sr)
------------------------------------'''
def read_file(file_name):
    sample_file = file_name
    sample_directory = '00_samples/'
    sample_path = sample_directory + sample_file

    # generating audio time series and a sampling rate (int)
    y, sr = librosa.load(sample_path)

    return y, sr

'''------------------------------------
NOISE REDUCTION USING POWER:
receives an audio matrix,
returns the matrix after gain reduction on noise
------------------------------------'''
def reduce_noise_power(y):

    # using power
    # S = np.abs(librosa.stft(y))
    # S_power = librosa.power_to_db(S**2)
    # print (S_power.shape)
    # print (S_power)
    # print (S_power.max())
    # print (S_power/S_power.max())


    # using the centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    # print (cent.shape)
    # print (cent)
    # print (np.mean(cent))
    # print (np.median(cent))

    threshold_h = round(np.median(cent))*1.5
    threshold_l = round(np.median(cent))*0.2
    less_noise = AudioEffectsChain().lowshelf(gain=-12.0, frequency=threshold_l, slope=0.5).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)
    y_clean = less_noise(y)
    # print (y_clean)

    return y_clean


'''------------------------------------
SILENCE TRIMMER:
receives an audio matrix,
returns an audio matrix with almost no silence
------------------------------------'''
def trim_silence(y):
    y_trimmed, index = librosa.effects.trim(y, top_db=12, frame_length=2)

    return y_trimmed


'''------------------------------------
AUDIO ENHANCER:
receives an audio matrix,
returns the same matrix after audio manipulation
------------------------------------'''
def enhance(y):
    apply_audio_effects = AudioEffectsChain().lowshelf(gain=12.0, frequency=260, slope=0.1).reverb(reverberance=25, hf_damping=5, room_scale=5, stereo_depth=50, pre_delay=20, wet_gain=0, wet_only=False)#.normalize()
    y_enhanced = apply_audio_effects(y)

    return y_enhanced

'''------------------------------------
OUTPUT GENERATOR:
receives a destination path, file name, audio matrix, and sample rate,
generates a wav file based on input
------------------------------------'''
def output_file(destination ,filename, y, sr):
    destination = destination + filename + '.wav'
    librosa.output.write_wav(destination, y, sr)


'''------------------------------------
LOGIC

files:
01_counting.m4a
02_wind_and_cars.m4a
03_truck.m4a
04_voices.m4a
05_ambeint.m4a
06_office.m4a
------------------------------------'''
samples = ['01_counting.m4a','02_wind_and_cars.m4a','03_truck.m4a','04_voices.m4a','05_ambeint.m4a','06_office.m4a']

# destination paths
trimmed_destination = 'samples_trimmed/'
noise_destination = 'samples_noise_reduced/'

for s in samples:
    # reading a file
    filename = s
    y, sr = read_file(filename)

    # reducing noise using db power
    y_reduced_power = reduce_noise_power(y)

    # # generating output file [1]
    output_file('01_samples_noise_reduced/' ,filename, y_reduced_power, sr)

    # trimming silences
    y_trimmed_power = trim_silence(y_reduced_power)

    # generating output file [2]
    output_file('02_samples_trimmed/' ,filename, y_trimmed_power, sr)

    # enhancing sound style
    y_enhanced = enhance(y_trimmed_power)

    # generating output file [3]
    output_file('03_samples_enhanced/' ,filename, y_enhanced, sr)

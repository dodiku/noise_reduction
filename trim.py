import librosa

'''
01_counting.m4a
02_wind_and_cars.m4a
03_truck.m4a
04_voices.m4a
05_ambeint.m4a
06_office.m4a
'''

sample_file = '06_office.m4a'
sample_directory = 'samples/'
sample_path = sample_directory + sample_file
trimmed_destination = 'samples_trimmed/'
silenced_destination = 'samples_silence_reduced/'

y, sr = librosa.load(sample_path)
y_trimmed, index = librosa.effects.trim(y, top_db=12, frame_length=2)
print(librosa.get_duration(y), librosa.get_duration(y_trimmed))

destination = trimmed_destination + sample_file[:-4] + '.wav'
librosa.output.write_wav(destination, y_trimmed, sr)

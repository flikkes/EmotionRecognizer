import struct
import os
import librosa
import math
import json


class WavFileHelper():

    def read_file_properties(self, filename):

        wave_file = open(filename, "rb")

        riff = wave_file.read(12)
        fmt = wave_file.read(36)

        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I", sample_rate_string)[0]

        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H", bit_depth_string)[0]

        return (num_channels, sample_rate, bit_depth)

    def save_mfcc(self, dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, sample_rate=44100, segment_duration=2):
        data = {
            "mapping": [],
            "mfcc": [],
            "labels": []
        }

        num_samples_per_segment = int()

        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
            if dirpath is not dataset_path:

                dirpath_components = dirpath.split("/")
                semantic_label = dirpath_components[-1]
                data["mapping"].append(semantic_label)
                print("\nProcessing {}".format(semantic_label))

                for f in filenames:
                    file_path = os.path.join(dirpath, f)
                    signal, sr = librosa.load(file_path, sr=sample_rate)

                    duration = librosa.core.get_duration(
                        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
                    samples_per_track = sample_rate * duration
                    num_segments = int(duration / segment_duration)
                    num_samples_per_segment = int(
                        samples_per_track / num_segments)
                    expected_num_mfcc_vectors_per_segment = math.ceil(
                        num_samples_per_segment / hop_length)

                    for s in range(num_segments):
                        start_sample = num_samples_per_segment * s
                        finish_sample = start_sample + num_samples_per_segment
                        mfcc = librosa.feature.mfcc(
                            signal[start_sample:finish_sample], sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                        mfcc = mfcc.T
                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            print("{}, segment: {}".format(file_path, s))
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)

    def extract_mfcc(self, file_name, n_mfcc=13, n_fft=2048, hop_length=512, sample_rate=44100, segment_duration=2):
        signal, sr = librosa.load(file_name, sr=sample_rate)
        duration = librosa.core.get_duration(
            y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
        samples_per_track = sample_rate * duration
        num_segments = int(duration / segment_duration)
        num_samples_per_segment = int(samples_per_track / num_segments)
        expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
        return_data = []
        for s in range(num_segments):
            start_sample = num_samples_per_segment * s
            finish_sample = start_sample + num_samples_per_segment
            mfcc = librosa.feature.mfcc(
                signal[start_sample:finish_sample], sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
            mfcc = mfcc.T
            if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                return_data.append(mfcc)
        return return_data

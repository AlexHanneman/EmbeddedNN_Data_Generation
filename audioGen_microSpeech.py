import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from argparse import ArgumentParser
from typing import List


# We add this path so we can import the speech processing modules.//
# sys.path.append("/content/tensorflow/tensorflow/examples/speech_commands/")
import input_data
import models

WANTED_WORDS = "yes,no"

number_of_labels = WANTED_WORDS.count(',') + 1
number_of_total_labels = number_of_labels + \
    2  # for 'silence' and 'unknown' label

PREPROCESS = 'micro'
WINDOW_STRIDE = 20
MODEL_ARCHITECTURE = 'tiny_conv'

SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
    SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
    WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)

audio_processor = input_data.AudioProcessor(
    None, None,
    0, 0,
    WANTED_WORDS.split(','), 0,
    0, model_settings, None)


def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


def getWavData(wav_path):
    with tf.compat.v1.Session() as sess:
        data = audio_processor.get_features_for_wav(
            wav_path, model_settings, sess)
        return data


def run_tflite_inference(tflite_model_path, wav_path, data):
    # Load Data
    # with tf.compat.v1.Session() as sess:

    # data = audio_processor.get_features_for_wav(
    #   wav_path, model_settings, sess)
    # data_label = 3

    # data = np.expand_dims(data, axis=).astype(np.float32)
    data = np.array(data)

    # Append extra dimension
    data = data[:, :, :, np.newaxis]

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()

    # Get the input and output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Quantise the data using input settings
    input_scale, input_zero_point = input_details["quantization"]
    data = data / input_scale + input_zero_point
    data = data.astype(input_details["dtype"])

    # Run Inferance
    interpreter.set_tensor(input_details["index"], data)
    interpreter.invoke()

    # Get the output results and sort
    output = interpreter.get_tensor(output_details["index"])[0]
    label_id = output.argmax()
    prob = output[label_id]

    return [label_id, prob]


def wavToC(features, outputPath, labelID, labelProb):
    variable_base = "soundData"
    with gfile.GFile(outputPath, 'w') as f:
        features_min, features_max = input_data.get_features_range(
            model_settings)
        f.write('const unsigned char g_%s[] = {' % variable_base)
        i = 0
        for value in features.flatten():
            quantized_value = int(
                round((255 * (value - features_min)) / (features_max - features_min)))
            if quantized_value < 0:
                quantized_value = 0
            if quantized_value > 255:
                quantized_value = 255
            if i == 0:
                f.write('\n  ')
            f.write('%d, ' % (quantized_value))
            i = (i + 1) % 10
        f.write('\n};\n')


def wavToBin(features, outputPath):
    audio = []
    features_min, features_max = input_data.get_features_range(
        model_settings)
    for value in features.flatten():
        quantized_value = int(
            round((255 * (value - features_min)) / (features_max - features_min)))
        if quantized_value < 0:
            quantized_value = 0
        if quantized_value > 255:
            quantized_value = 255
        audio.append(quantized_value)
    with open(outputPath, "wb") as f:
        f.write(bytearray(audio))


if __name__ == '__main__':
    parser = ArgumentParser("Generate audio binary formatted for microspeech")
    parser.add_argument(
        "--input", "-i", help="Input file/folder path", type=str, required=True)
    parser.add_argument(
        "--output", "-o", help="Output file/folder path", type=str, required=True)

    arguments = parser.parse_args()

    modelPath: str = os.path.join(os.getcwd(), 'microSpeech.tflite')
    sourceFiles: List[str] = []

    if not os.path.exists(arguments.input):
        raise ValueError("Input path does not exist")
    if not os.path.exists(arguments.output):
        raise ValueError("Output path does not exist")

    if os.path.isdir(arguments.input):
        sourceFiles = os.listdir(arguments.input)
    elif os.path.isfile(arguments.input):
        sourceFiles = [arguments.input]
    else:
        raise ValueError("Input path not directory or file")

    for file in sourceFiles:
        data = getWavData(os.path.join(os.getcwd(), arguments.input, file))
        features = data[0]

        # result = run_tflite_inference(modelPath, filePath, data)
        # label_id, prob = result
        # print("Sound detected is: ", label_id,
        # "(", labels[label_id], ")", " With Probability: ", (prob/255)*100, "% ")

        fileName: str = os.path.join(
            arguments.output, os.path.basename(file)) + ".bin"

        wavToBin(features, fileName)

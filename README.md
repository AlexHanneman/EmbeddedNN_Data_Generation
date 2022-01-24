# Embedded NN Binary Data Generators

Inputs can be provided either as a path to a file or as a path to a directory containing data files

Output path should be to a directory

All paths provided should be relative to the current working directory

## Image Gen
Supports binary image generation for MobileNet (TFL and CMix) and PersonDetect

MobileNet image size  = 160, 160
Alpha channel needed for CMix implementation

PersonDetect image size = 238, 208 (W,H) no alpha channel needed

```pwsh
python .\imageGen_MobileNet_PersonDetect.py --width 160 --height 160 --input .\input --output output
```

## Audio Gen
Supports generation of .bin files from .wav files for processing using microspeech network

Currently tensorflow must be installed to generate the binary files. The .tflite file is also needed as TensorFlow takes the output parameters from this file.

Removing the dependency on tensorflow is desired.

Example batch generation of audio files
```pwsh
python .\audioGen_microSpeech.py --input input --output output
```

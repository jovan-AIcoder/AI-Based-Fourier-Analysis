import os
import pandas as pd
import librosa
import numpy as np
from tensorflow import keras
from tensorflow import math
def analyze(audio_path,max_modes=10000,epochs=256,use_phase_shift=True,learning_rate=0.00001,save_model=None,verbose=2,positive_freqs_only=True,abs_amplitudes=True):
    # error handling
    if not isinstance(audio_path, str):
        raise ValueError('Audio path must be a string.')
    if not isinstance(max_modes, int) or max_modes <= 0:
        raise ValueError('Max modes must be a positive integer.')
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError('Epochs must be a positive integer.')
    if not isinstance(use_phase_shift,bool):
        raise ValueError('use_phase_shift must be a boolean.')
    if not (isinstance(learning_rate,(int,float))) or learning_rate <= 0:
        raise ValueError('learning_rate must be a positive float.')
    if(save_model != None) and (type(save_model) != str):
        raise ValueError('save_model must be a string or None.')
    if (not isinstance(verbose,int)) or (verbose not in [0,1,2]):
        raise ValueError('Available values for verbose is 0,1,2.')
    if not isinstance(positive_freqs_only,bool):
        raise ValueError('positive_freqs_only must be a boolean.')
    if not isinstance(abs_amplitudes,bool):
        raise ValueError('abs_amplitudes must be a boolean')
    try:
        if not os.path.isfile(audio_path):
            raise ValueError("Audio file not found.")
    except Exception as e:
        raise ValueError(f'Error accessing audio file: {e}')
    if not audio_path.endswith(('.wav', '.mp3', '.flac', '.ogg')):
        raise ValueError('Unsupported audio format. Supported formats are: .wav, .mp3, .flac, .ogg')
    
    # audio sampling
    if(verbose > 0):
        print('Sampling the audio signal...')

    a, b = librosa.load(audio_path, mono=True)
    duration = librosa.get_duration(y=a, sr=b)
    time = np.linspace(0, duration, len(a))
    df = pd.DataFrame({'time': time,'amplitude': a})
    t = df['time'] * 1e6 # convert to microseconds
    y = df['amplitude']
    if len(a) < 10:
        raise ValueError('Audio file is too short.')
    if(verbose > 0):
        print('Audio signal sampled successfully.')

    # analyze the audio signal
    model = keras.Sequential([
        keras.layers.Dense(max_modes, activation=math.sin, input_shape=(1,),use_bias=use_phase_shift),
        keras.layers.Dense(1,use_bias=False)
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    if(verbose == 1):
        print('Model summary: \n')
        model.summary()
    if(verbose > 0):
        print('Analyzing the audio signal...')

    model.fit(t.values.reshape(-1, 1), y.values, epochs=epochs, verbose=verbose)
    if use_phase_shift:
        weights, biases = model.layers[0].get_weights()
        amplitudes = model.layers[1].get_weights()[0]
        weights_flat = ((weights[0])*1e6)/(2*np.pi) # convert to Hz
        df_freq = pd.DataFrame({'Frequencies': weights_flat,'Phase shift': biases,'Amplitudes': amplitudes.flatten()})
        df_freq.index = [f'Frequency_{i}' for i in range(len(biases))]
    else:
        weights = model.layers[0].get_weights()[0]
        amplitudes = model.layers[1].get_weights()[0]

        weights_flat = ((weights.flatten())*1e6)/(2*np.pi) # convert to Hz

        df_freq = pd.DataFrame({
            'Frequencies': weights_flat,
            'Amplitudes': amplitudes.flatten()
        })

        df_freq.index = [
            f'Frequency_{i}'
            for i in range(len(weights_flat))
        ]
    
    if positive_freqs_only:
        if(verbose > 0):
            print('Keeping the positive frequencies...')
        df_freq = (df_freq[df_freq['Frequencies'] >= 0].reset_index(drop=True))  
    
    if abs_amplitudes:
        if(verbose > 0):
            print('Taking the absolute value of all amplitudes...')
        df_freq['Amplitudes'] = (df_freq['Amplitudes'].abs())
    
    if(verbose > 0):
        print('Audio signal analyzed successfully.')
    
    if(save_model != None):
        model.save(save_model)
        print(f'Model is saved successfully to path: {save_model}.')
    
    return df_freq
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow import math
import os
def sin(x):
    return math.sin(x)
def analyzer(audio_path,max_freq=2048):
    # audio sampling
    print('Sampling the audio signal...')
    a, b = librosa.load(audio_path, mono=True)
    duration = librosa.get_duration(y=a, sr=b)
    time = np.linspace(0, duration, len(a))
    df = pd.DataFrame({'time': time,'amplitude': a})
    t = df['time']
    y = df['amplitude']
    # analyze the audio signal
    model = keras.Sequential([
        keras.layers.Dense(max_freq, activation=sin, input_shape=(1,)),
        keras.layers.Dense(1,use_bias=False)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print('Analyzing the audio signal...')
    model.fit(t, y, epochs=30)
    y_pred = model.predict(t)
    # plot the audio signal
    plt.plot(t, y)
    plt.plot(t, y_pred, '--')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Signal')
    plt.legend(['Original Signal', 'Learned Signal'])
    plt.savefig('audio_signal.png')
    plt.show()
    weights, biases = model.layers[0].get_weights()
    amplitudes = model.layers[1].get_weights()[0]
    weights_flat = weights[0]
    df_freq = pd.DataFrame({'Frequencies': weights_flat,'Phase shift': biases,'Amplitudes': amplitudes.flatten()})
    df_freq.index = [f'Neuron_{i}' for i in range(len(biases))]
    return df_freq


df_freq = analyzer('bird.mp3')
df_freq.to_excel('frequencies.xlsx', index_label='frequency_index')
os.startfile('frequencies.xlsx')
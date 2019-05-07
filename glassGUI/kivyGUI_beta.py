from kivy.lang import Builder
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.garden.graph import MeshLinePlot
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.properties import StringProperty
from kivy.properties import ListProperty

from threading import Thread
import audioop
import pyaudio
import librosa

from keras.models import load_model
import tensorflow as tf

import numpy as np
import wave

import timeit
LABELS = ['Glass breaking!',' '] # Others, Siren, Honks

def get_microphone_level():
    """
    source: http://stackoverflow.com/questions/26478315/getting-volume-levels-from-pyaudio-for-use-in-arduino
    audioop.max alternative to audioop.rms
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    DEVICE_IDX = 6
    p = pyaudio.PyAudio()

    s = p.open(format=FORMAT,
               channels=CHANNELS,
               rate=RATE,
               input=True,
               input_device_index = DEVICE_IDX,
               frames_per_buffer=CHUNK)

    global levels
    global levels_2
    global four_sec
    global temp_levels_2

    while True:
        data = s.read(CHUNK,exception_on_overflow=False)

        mx = audioop.rms(data, 2)/255

        if len(levels_2) >= int(RATE*0.5/CHUNK):

            temp_levels_2.extend(levels_2)

            four_sec = temp_levels_2

            temp_levels_2 = levels_2         

            levels_2 = []

        if len(levels) >= 100:
            levels = []

        levels.append(mx)
        levels_2.append(data)

	# pass to model
	# get results
	# if result: change label
	# else: continue

class Logic(BoxLayout):
    pred_label = StringProperty()
    bg_color = ListProperty()
    font_color = ListProperty()

    def __init__(self,):
        super(Logic, self).__init__()
        self.plot = MeshLinePlot(color=[1, 1, 1, 1])
        self.pred_label = ' '
        self.bg_color = [0,0,0,1]
        self.font_color = [1,1,1,1]

    def start(self):
        self.ids.graph.add_plot(self.plot)
        Clock.schedule_interval(self.get_value, 0.1)

        Clock.schedule_interval(self._listen, 1)

    def stop(self):
        Clock.unschedule(self.get_value)
        Clock.unschedule(self._listen)

    def get_value(self, dt):
        self.plot.points = [(i, j/5) for i, j in enumerate(levels)]
	
    def _listen(self,dt):

        X = np.empty(shape=(1,128,63,1,1))
	    
        input_length = 16000 # Sampling rate * Audio duration = 44100 * 1 = 44100 
		
        global four_sec
        #tic = timeit.default_timer()
        wf = wave.open('output.wav', 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(four_sec))
        wf.close()
        #fname = 'models/clips/7389-1-0-2.wav'

        #four_sec, _ = librosa.core.load(fname, sr=44100, res_type="kaiser_fast")
        #data = np.transpose(four_sec)		

        data, _ = librosa.core.load('output.wav', sr=16000, res_type="kaiser_fast")
		# preprocessing
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
               max_offset = input_length - len(data)
               offset = np.random.randint(max_offset)
            else:
               offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        #y = librosa.util.normalize(data)
        
        #y = np.array(data)
        #y = y/255
        y = librosa.util.normalize(data)
        librosa.output.write_wav('output2.wav', y, sr=16000, norm=False)           
        #y = librosa.util.buf_to_float(y,n_bytes=8)
		# convert to melspec
        melspec = librosa.feature.melspectrogram(y = y, sr = 16000, n_fft=512, hop_length=256, n_mels=128)
	   
        log_melspec = np.log(melspec.T + 1e-10)
	    
        log_melspec = np.array([log_melspec])
        log_melspec = np.transpose(log_melspec)
        y = log_melspec
	    
        y = np.expand_dims(y, axis=-1)

        X[0,] = y
        
        X = np.squeeze(X, axis =-1)
        #toc = timeit.default_timer()
        #print(toc-tic)
        with graph.as_default(): # For circumventing keras's problem
             predictions = model.predict(X,batch_size=1,verbose=1)

        top_1 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :1]]
        the_prediction = ([' '.join(list(x)) for x in top_1])[0]
        if the_prediction == 'Glass breaking!':
           self.pred_label = the_prediction
           self.bg_color = [1,0,0,1]
           self.font_color = [0,0,0,1]
        else:
        	self.pred_label = the_prediction
        	self.bg_color = [0,0,0,1]
        	self.font_color = [1,1,1,1]

        print(self.pred_label)

class RealTimeMicrophone(App):
    def build(self):
        return Builder.load_file("MyWidget.kv")


if __name__ == "__main__":
    
    # Loading stuff
    model = load_model('best_glass_16k.h5')
    model.load_weights('best_glass_16k.h5')
    graph = tf.get_default_graph()

    temp_levels = []
    temp_levels_2 = []
    four_sec = []
    levels = []  # store levels of microphone
    levels_2 = []

    get_level_thread = Thread(target = get_microphone_level)
    get_level_thread.daemon = True
    get_level_thread.start()

    RealTimeMicrophone().run()
import sounddevice as sd
import sys

devices = sd.query_devices()
indevice = -1
outdevice = -1
for i in range(len(devices)) :
    if devices[i]['name'] == 'Soundflower (2ch)' :
        indevice = i
    if devices[i]['name'] == 'Built-in Output' :
            outdevice = i

if indevice == -1 or outdevice == -1 :
    print("couldn't find the right devices")
    sys.exit

instream = sd.InputStream(samplerate = 44100, device = indevice)
outstream = sd.OutputStream(samplerate = 44100, device = outdevice)
instream.start()
outstream.start()
while True :
    outstream.write(instream.read(1024)[0])

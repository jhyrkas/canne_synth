import sounddevice as sd

devices = sd.query_devices()
indevice = 0
outdevice = 0
for i in range(len(devices)) :
    if devices[i]['name'] == 'Soundflower (2ch)' :
        indevice = i
    if devices[i]['name'] == 'Built-in Output' :
        outdevice = i

instream = sd.InputStream(samplerate = 44100, device = indevice)
outstream = sd.OutputStream(samplerate = 44100, device = outdevice)
instream.start()
outstream.start()
while True :
    outstream.write(instream.read(1024)[0])

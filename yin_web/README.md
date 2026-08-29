# YIN Pitch Detector -- Django (simple)

A minimal Django app around your YIN notebook code.

- `detector/yin.py` -- your notebook algorithm, unchanged.
- `detector/views.py` -- one view: read the WAV, run `pitchDetect`, draw the
  waveform + pitch plot, list the notes.
- `detector/templates/detector/index.html` -- one page: a file-upload form and
  a Record button, with the results shown below.

## How input works
- **Upload:** pick a `.wav` file and submit the form.
- **Live mic:** the browser records, converts the audio to WAV itself, and
  submits it through the same form. Because the browser sends WAV, the server
  needs no ffmpeg -- just `soundfile`.

## Run
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```
Open http://127.0.0.1:8000/ . For mic access use localhost or HTTPS.

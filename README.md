# NGX
Social Distancing System using NAO and AI

## Getting Started
On Mac OS X run:
```
pip install pipenv
pipenv install -r requirements.txt
pipenv shell
```


On Windows run:
```
pip install pipenv
pipenv install -r requirements.txt
pipenv shell
```

In order to start NGX code clone [this repository](https://github.com/moj124/NGX) and run two commands: 
```
pipenv run python3 manage.py runserver
pipenv run python3 manage.py runscript run_video
```



## Youtube Video Showing Kinba

[Click here](https://www.youtube.com/watch?v=FGXuvlBr0H0)


### How to run tests:

 Run `test.sh`
 ```bash
 ./test.sh
 ```
## Optional Dependencies

- Any pyttsx3 text-to-speech engine (``sapi5, nsss or espeak``) for Kinba to talk out loud (e.g. Ubuntu do ``sudo apt install espeak``)
- Portaudio + python-devel packages for voice control
- ``notify-send`` on Linux if you want to receive *nice* and desktop-notification instead of *ugly* pop up windows (e.g. Ubuntu do ``sudo apt install libnotify-bin``)
- ``ffmpeg`` if you want ``music`` to download songs as .mp3 instead of .webm

## Docker

Run with docker (docker needs to be installed and running):

```
[sudo] make build_docker
[sudo] make run_docker
```
## 1. How to install Kinba:

firstly install jarvis here: https://github.com/delandcaglar/kinbapy 
(the )Please use an pyhthon 3.6+ environment(sidenote: webbrowser like libraries might cause problems on 3.7 if anaconda python get used it can significatly ease the installing process especially on kivy part on linux)

To install the necessary kinba libraries for mac OS X:

Download espeak on mac: http://macappstore.org/espeak/

Download pyaudio:
```
xcode-select --install
brew remove portaudio
brew install portaudio
pip3 install pyaudio
```
Download ffmpeg:

`brew install ffmpeg`

Download wkhtmltopdf: https://wkhtmltopdf.org/downloads.html

Then run
`cd Jarvis-master`
then run
`./test.sh`

if it does not work run __main__.py in installer file

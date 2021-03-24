# NGX
Social Distancing System using NAO and AI, uses Django technology to create a website for the GUI and the use of YOLOv5 models to perfrom people and mask detection and tracking.

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

In order to start NGX code clone [this repository](https://github.com/moj124/NGX) and run the two commands in parallel using two terminal windows: 
```
pipenv run python3 manage.py runserver
pipenv run python3 manage.py runscript run_video
```

## Troubleshooting

Its common over multiple executions of the ''pipenv run python3 manage.py runserver'' will produce redundant TCP connections after the attempts of closing down via 'CTRL + C'. In order to counteract this problem, run the follow command:

'''
sudo lsof -t -i tcp:8000 | xargs kill -9 
'''
This should kill any remaining TCP connection, allowing the established connection to the local webpage being served via the runserver command highlighted above.

## NAO programming interface
In order to develop the system with usage of the NAOqi software, the Python SDK of Naoqi must be downloaded from [here](https://www.softbankrobotics.com/emea/en/support/nao-6/downloads-softwares), you may find this installation guide helpful [here](https://developer.softbankrobotics.com/nao6/naoqi-developer-guide/sdks/python-sdk/python-sdk-installation-guide#python-install-guide).

Also the python path must be assigned to the respective Naoqi python directory:
```
 PYTHONPATH = 'path/to/Naoqi/python/'
 export PYTHONPATH
```

### Interfacing with NAO
Implementation with the NAO robot can be achieved to produce the desired feedback functionality design within ['pages/detect_person_yolov5.py'](https://github.com/moj124/NGX/blob/main/pages/detect_person_yolov5.py) that is being called within ['scripts/run_video'](https://github.com/moj124/NGX/tree/main/scripts).



## Youtube Video Showing Kinba

[Click here](https://www.youtube.com/watch?v=FGXuvlBr0H0)


### How to run tests:

 Run `test.sh`
 ```bash
 ./test.sh
 ```

# NGX
Social Distancing System using NAO and AI, uses Django technology to create a website for the GUI and the use of YOLOv5 models to perfrom people and mask detection and tracking.

## Getting Started with Masked-Wearer Detection
Show processing of images and videos for mask detection
```
python3 detect_mask_yolov5.py --view-img
```
## Getting Started with People Counter and GUI
A Windows based installation and function guide is provided, [Click here](https://youtu.be/RUkjI9g5vaw).



On Mac OS X run:
```
pip install pipenv
pipenv install -r requirements.txt
pipenv shell
pipenv run python3 manage.py makemigrations pages
pipenv run python3 manage.py migrate
```


On Windows run:
```
pip install pipenv
pipenv install -r requirements.txt
pipenv shell
pipenv run python manage.py makemigrations pages
pipenv run python manage.py migrate
```

In order to start NGX code clone [this repository](https://github.com/moj124/NGX) and run the two commands in parallel using two terminal windows: 
```
pipenv run python3 manage.py runserver
pipenv run python3 manage.py runscript run_video
```

### Creating a Superuser
To start off the webpage, its good to create a superuser that has full access to the website and is able to manage the database of the GUI. 
```
pipenv run python3 manage.py createsuperuser
```

Enter details:
```
Username: *****
Email address: admin@example.com
Password: **********
Password (again): *********
Superuser created successfully.
```
### Troubleshooting

Its common over multiple executions of the `pipenv run python3 manage.py runserver` will produce redundant TCP connections after the attempts of closing down via `CTRL + C`. In order to counteract this problem, run the follow command:

```
sudo lsof -t -i tcp:8000 | xargs kill -9 
```
This should kill any remaining TCP connection, allowing the established connection to the local webpage being served via the runserver command highlighted above.

### NAO programming interface
In order to develop the system with usage of the NAOqi software, the Python SDK of Naoqi must be downloaded from [here](https://www.softbankrobotics.com/emea/en/support/nao-6/downloads-softwares), you may find this installation guide helpful [here](https://developer.softbankrobotics.com/nao6/naoqi-developer-guide/sdks/python-sdk/python-sdk-installation-guide#python-install-guide).

Also the python path must be assigned to the respective Naoqi python directory:
```
 PYTHONPATH = 'path/to/Naoqi/python/'
 export PYTHONPATH
```

### Interfacing with NAO
Implementation with the NAO robot can be achieved to produce the desired feedback functionality design within ['pages/detect_person_yolov5.py'](https://github.com/moj124/NGX/blob/main/pages/detect_person_yolov5.py) that is being called within ['scripts/run_video'](https://github.com/moj124/NGX/tree/main/scripts).



### Youtube Video Showing GUI with Detection/Tracking Aspects

[Click here](https://youtu.be/RUkjI9g5vaw)


### How to run the Test Videos:
Parameters for different videos can be set in ['scripts/run_video'](https://github.com/moj124/NGX/tree/main/scripts), Such as running these two parameters seperately, will reproduce the results in ['runs/tests'](https://github.com/moj124/NGX/tree/main/runs/tests):
```
def run():
    """
    run script for detection and tracking algoirthm
    """
    # retrieve inputs for line trigger and data source

    opt = {}

    # Test with street video
    # opt['start'] = (0, 500)
    # opt['end'] = (1800, 950)
    # opt['line-side'] = 'left'
    # opt['source'] = 'data/images/street.mp4'
    # opt['axis'] = 'horizontal'

    # Test with mass_walking video
    opt['start'] = (0, 500)
    opt['end'] = (1344, 500)
    opt['source'] = 'data/images/mass_walking.mp4'
    opt['axis'] = 'horizontal'
    opt['line-side'] = 'right'
    
    # Test with walkingby video
    #opt['start'] = (400, 0)
    #opt['end'] = (400, 1000)
    #opt['source'] = 'data/images/walkingby.mp4'
    #opt['axis'] = 'vertical'
    #opt['line-side'] = 'right'
```
By setting these parameters, we can then run the script with the GUI using these commands in parallel:
```
pipenv run python3 manage.py runserver
pipenv run python3 manage.py runscript run_video
```

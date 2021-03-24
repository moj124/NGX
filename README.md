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



## Youtube Video Showing Kinba

[Click here](https://www.youtube.com/watch?v=FGXuvlBr0H0)


### How to run tests:

 Run `test.sh`
 ```bash
 ./test.sh
 ```

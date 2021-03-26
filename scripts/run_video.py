# Usage
# pipenv run python3 manage.py runscript run_video

from pages.detect_person_yolov5 import start, warning, p
from django.db.models import Sum
from django.db.models.functions import Coalesce
from pages.models import Group
import datetime
import argparse
import random


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
    # opt['start'] = (0, 500)
    # opt['end'] = (1344, 500)
    # opt['source'] = 'data/images/mass_walking.mp4'
    # opt['axis'] = 'horizontal'
    # opt['line-side'] = 'right'

    # Test with walkingby video
    opt['start'] = (400, 0)
    opt['end'] = (400, 1000)
    opt['source'] = 'data/images/walkingby.mp4'
    opt['axis'] = 'vertical'
    opt['line-side'] = 'right'

    # retrieve active group total

    def get_current():
        """
        retrieve total of active group counts
        """
        queryset = get_valid_groups()
        queryset = queryset.aggregate(sum_=Coalesce(Sum('people'), 0))

        return queryset

    # retrieve all active group records
    def get_valid_groups():
        """
        retrieve all active group records
        """
        queryset = Group.objects.all()

        # get time delta  of 15 minutes
        start_date = datetime.datetime.now() - datetime.timedelta(minutes=15)

        # filter by time delta of 15 minutes
        queryset = queryset.filter(
            date__range=[start_date, datetime.datetime.now()])

        return queryset

    # run detection and and tracking algorithms
    start(opt['start'], opt['end'], opt['line-side'],
          opt['axis'], opt['source'])

    # For the NAO robot code
    ########################################################################################
    # set the capacity limits
    # lower_capacity = 20
    # max_capacity = 30

    # select random number between 0 and 1 uniformally
    # prob = random.uniform(0, 1)

    # get current total of activate groups
    # x = get_current()

    # perform congestion control
    # if (x >= lower_capacity and x <= max_capacity and p(x) > prob) or x > max_capacity:
    #     warning('Room 41 is currently, full please wait or check another room.')
    ########################################################################################

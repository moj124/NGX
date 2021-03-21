# Usage
# pipenv run python3 manage.py runscript run_video

from pages.detect_person_yolov5 import start, warning, p
from django.db.models import Sum
from django.db.models.functions import Coalesce
from pages.models import Group
import datetime
import random


def run():
    """
    run script for detection and tracking algoirthm
    """

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
    start()

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

from django.urls import reverse_lazy
from django.shortcuts import render, redirect
from django.views.generic import ListView
from django.db.models import Sum
from django.db.models.functions import Coalesce
from django.contrib.auth import authenticate, login, logout
from pages.forms import SignUpForm
from django.contrib.auth.forms import AuthenticationForm

from .models import Group
import csv
import datetime


class EventListView(ListView):
    login_url = '/login/'
    redirect_field_name = '/accounts/login/'
    model = Group
    template_name = 'dashboard.html'
    success_url = reverse_lazy('/dashboard/')

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get the context
        context = super(EventListView, self).get_context_data(**kwargs)
        # Create any data and add it to the context
        context['count'] = self.get_current()
        context['total'] = self.get_total_count()
        context['groups'] = self.get_valid_groups()

        return context

    def get_current(self):
        # get sum of people from active groups
        queryset = self.get_valid_groups()
        queryset = queryset.aggregate(sum_=Coalesce(Sum('people'), 0))

        return queryset

    def get_valid_groups(self):
        queryset = Group.objects.all()

        # get time delta  of 15 minutes
        start_date = datetime.datetime.now() - datetime.timedelta(minutes=15)

        # filter by time delta of 15 minutes
        queryset = queryset.filter(
            date__range=[start_date, datetime.datetime.now()])
        return queryset

    def get_total_count(self):
        # retrieve total by the data file
        with open("data/data.csv", 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            total = sum(float(row["count"]) for row in reader)

        return int(total)


def admin_view(request, *args, **kwargs):
    return redirect('/admin/')


def logout_view(request, *args, **kwargs):
    # log out user
    logout(request)
    return redirect('/home/')


def home_view(request, *args, **kwargs):
    return render(request, "home.html")


def login_view(request, *args, **kwargs):

    if request.method == 'POST':  # request is has a form

        # retrieve username and password
        username = request.POST['username']
        password = request.POST['password']

        # authenticate the details of the user
        user = authenticate(username=username, password=password)

        # if authentication fails to confirm
        if user is not None:
            # login the user
            login(request, user)
            return redirect('/home/')
        else:
            # send login form
            form = AuthenticationForm()
            return render(request, 'registration/login.html', {'form': form})
    else:
        # send login form
        form = AuthenticationForm()
        return render(request, 'registration/login.html', {'form': form})


def register_view(request, *args, **kwargs):

    if request.method == 'POST':  # request is has a form

        form = SignUpForm(request.POST)

        # validates form sent
        if form.is_valid():
            # save details
            form.save()

            # get username and password details from form
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')

            # authenticate user
            user = authenticate(username=username, password=raw_password)

            # log user in
            login(request, user)

            # enter user to homepage
            return redirect('/home/')
        # error occurs reload registration page
        return render(request, "register.html")
    else:
        # retrieve form
        form = SignUpForm()
        return render(request, 'register.html', {'form': form})


def display_view(request, *args, **kwargs):

    # retrieve active group total
    def get_total():
        queryset = get_valid_groups()
        queryset = queryset.aggregate(sum_=Coalesce(Sum('people'), 0))
        return queryset

    # retrieve all active group records
    def get_valid_groups():
        queryset = Group.objects.all()

        # get time delta  of 15 minutes
        start_date = datetime.datetime.now() - datetime.timedelta(minutes=15)

        # filter by time delta of 15 minutes
        queryset = queryset.filter(
            date__range=[start_date, datetime.datetime.now()])
        return queryset

    return render(request, "display.html", {'count': get_total()})


def data_view(request, *args, **kwargs):
    return render(request, "data.html")

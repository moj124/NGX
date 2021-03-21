from django.conf.urls import include, url
from django.urls import path
from django.contrib.auth.decorators import login_required
from pages.views import EventListView, display_view, data_view, register_view, login_view, home_view, logout_view
urlpatterns = [
    path(r"dashboard/", login_required(EventListView.as_view()), name='dashboard'),
    path('', home_view, name='home'),
    path('home/', home_view, name='home'),
    path('display/', display_view, name='display'),
    path('data/', login_required(data_view), name='data'),
    path('accounts/login/', login_view, name='login'),
    path('register/', register_view, name='register'),
    path('logout/', logout_view, name='logout'),
    url(r"^accounts/", include("django.contrib.auth.urls"), name='accounts')
]

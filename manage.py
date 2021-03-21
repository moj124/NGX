#!/usr/bin/env python
# sudo lsof -t -i tcp:8000 | xargs kill -9
# pipenv run python3 manage.py startapp pages
# pipenv run python3 manage.py runserver
# pipenv run python3 manage.py collectstatic --noinput --clear
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trydjango.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

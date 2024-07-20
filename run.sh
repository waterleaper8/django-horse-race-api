#!/bin/bash
while true
do
        source /home/ubuntu/.profile
        cd /home/ubuntu/dev/django-horse-race-api
        source .venv/bin/activate
        python manage.py runserver
done

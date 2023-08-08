while true
do
        source /home/waterleaper/.profile
        cd /home/waterleaper/horse-race-api
        source .venv/bin/activate
        python manage.py runserver
done

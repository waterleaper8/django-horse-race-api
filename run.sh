while true
do
        source /home/waterleaper/.profile
        source /home/waterleaper/.bashrc
        cd /home/waterleaper/horse-race-api
        source .venv/bin/activate
        python manage.py runserver
done

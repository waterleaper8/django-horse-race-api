from django.shortcuts import render
from django.http import HttpResponse
import lightgbm as lgb
from . import odds
import pandas as pd

# Create your views here.
def oddsComparison(request):
    if 'race_url' in request.GET:
        race_url = request.GET['race_url']
        race_id = race_url.split('race_id=')[1][:12]
        print('race_id', race_id)
        try:
            odds_json = odds.detect(race_id)
            return HttpResponse(odds_json)
        except Exception as e:
            return HttpResponse(e)

from django.shortcuts import render
from django.http import HttpResponse
import lightgbm as lgb
from . import odds
import pandas as pd
import traceback

# Create your views here.


def oddsComparison(request):
    if 'race_id' in request.GET:
        race_id = request.GET['race_id']
        print('race_id', race_id)
        try:
            odds_json = odds.detect(race_id)
            return HttpResponse(odds_json)
        except Exception as e:
            # return HttpResponse(e)
            return HttpResponse(traceback.format_exc())

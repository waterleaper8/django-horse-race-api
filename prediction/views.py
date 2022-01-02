from django.shortcuts import render
from django.http import HttpResponse
import decimal
import lightgbm as lgb
from . import process
import pandas as pd

# Create your views here.
def init_read():
    # Resultsクラスのオブジェクトを作成
    r = process.Results.read_pickle([
        'prediction/pickle/results_21.pkl',
    ])
    # 前処理
    r.preprocessing()

    # 馬の過去成績データ追加
    hr = process.HorseResults.read_pickle([
        'prediction/pickle/horse_results_21.pkl',
        'prediction/pickle/hr_add.pkl',
    ])
    r.merge_horse_results(hr)
    # 5世代分の血統データの追加
    p = process.Peds.read_pickle([
        'prediction/pickle/peds_21.pkl',
        'prediction/pickle/peds_add.pkl',
    ])
    p.encode()
    r.merge_peds(p.peds_e)
    # カテゴリ変数の処理
    r.process_categorical()

    X = r.data_c.drop(['rank', 'date', '単勝'], axis=1)
    y = r.data_c['rank']

    # LightGBMのハイパーパラメータ
    params = {
        'n_estimators': 400,  # 100
        'learning_rate': 0.06,  # 0.1
        'max_depth': -1,  # -1
        'max_bin': 700,  # 255
        'min_child_weight': 0.001,  # 0.001
        'feature_pre_filter': False,

        'lambda_l1': 1.5067064463131568e-05,
        'lambda_l2': 8.382977103422135,
        'num_leaves': 31,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.8897348492363203,
        'bagging_freq': 2,
        'min_child_samples': 100,
    }

    # 予測モデル作成&学習
    lgb_clf = lgb.LGBMClassifier(**params)
    lgb_clf.fit(X.values, y.values)

    me = process.ModelEvaluator(lgb_clf, [
        'prediction/pickle/return_tables_21.pkl',
    ])

    return r, hr, p, lgb_clf, me


r, hr, p, lgb_clf, me = init_read()


def prediction(request):
    if 'race_url' in request.GET:
        race_url = request.GET['race_url']
        race_id = race_url.split('race_id=')[1][:12]
        try:
            stu = process.ShutubaTable.scrape([race_id])

            # 前処理
            stu.preprocessing()
            # 馬の過去成績データの追加。新馬はNaNが追加される。
            stu.merge_horse_results(hr)
            # 5世代分の血統データの追加
            stu.merge_peds(p.peds_e)
            # すべてのカテゴリをcategory型に登録し、ダミー変数化
            stu.process_categorical(r.le_horse, r.le_jockey, r.data_h)

            # 馬が勝つ確率を予測
            pred = me.predict_proba(stu.data_c.drop(
                ['date'], axis=1), train=False, std=False)
            # 予測結果を表に結合
            pred_table = stu.data[~(
                stu.data['馬体重(増減)'] == '--')][['馬番', '馬名']].copy()
            pred_table = pred_table.rename(
                columns={'馬番': 'number', '馬名': 'name'})
            pred_table['percentage'] = pred*100
            pred_table_json = pred_table.sort_values(
                'percentage', ascending=False).to_json(orient='records', force_ascii=False)

            return HttpResponse(pred_table_json)
        except Exception as e:
            return HttpResponse(e)

def coupon(request):
    if 'coupon_code' in request.GET:
        coupon_code = request.GET['coupon_code']
        if coupon_code == '0001':
            result = '1000円引きクーポン！'
        elif coupon_code == '0002':
            result = '10%引きクーポン！'
        else:
            result = 'Error:Not found coupon code!'
        return HttpResponse(result)

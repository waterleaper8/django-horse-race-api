import urllib.request
import urllib.parse
import time
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.support.select import Select
import chardet
from selenium.webdriver.common.by import By


def detect(race_id):
    progressBar = 0

    options = webdriver.ChromeOptions()
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("-headless")
    options.page_load_strategy = 'eager'
    driver = webdriver.Chrome(options=options)

    url = f'https://race.netkeiba.com/odds/index.html?race_id={race_id}'

    shutuba_url = f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
    shutuba_df = pd.read_html(shutuba_url, encoding='EUC-JP')[0]
    shutuba_df = shutuba_df.T.reset_index(level=0, drop=True).T
    print(shutuba_df.columns)
    shutuba_df = shutuba_df[['馬 番', '馬名']]
    shutuba_df = shutuba_df.set_axis(['umaban', 'umamei'], axis=1)

    driver.get(url)
    time.sleep(3)
    title = driver.title[:-26]
    t = driver.find_element("class name", 'RaceData01').text
    title = title + ' / ' + t

    # 単勝・複勝
    tanfuku = driver.find_element(By.CSS_SELECTOR, '#Ninki')
    umabans = tanfuku.find_elements(By.CSS_SELECTOR, 'tr')[1:]
    tansho = []
    umaban_list = []
    for umaban in umabans:
        if '取消' in umaban.text or '除外' in umaban.text:
            umaban_list.append(int(umaban.find_elements(By.CSS_SELECTOR, 'td')[2].text))
            continue
        try:
            _umaban = int(umaban.find_element(By.CSS_SELECTOR, 'span').text)
        except:
            continue
        umaban_list.append(int(umaban.find_element(By.CSS_SELECTOR, 'span').text))
    print("len(umaban_list)", len(umaban_list))
    tansho = pd.DataFrame(data=range(1, len(umaban_list)+1, 1), index=umaban_list, columns=['tansho'])
    tansho = tansho.sort_index()
    progressBar += 1

    ngscore = 0
    base_score = 1000000000
    score = base_score

    # 馬連
    url = f'https://race.netkeiba.com/odds/index.html?type=b4&race_id={race_id}&housiki=c99'
    print(url)
    driver.get(url)
    time.sleep(0.1)
    score_arr = [0] * len(umaban_list)

    # 馬券とそのオッズを取得し、スコアを計算
    combis = driver.find_elements(By.CSS_SELECTOR, '.Combi01')
    odds = driver.find_elements(By.CSS_SELECTOR, '.Name_Odds')
    for combi, odd in zip(combis, odds):
        combi = combi.text.split('\n')
        odd = float(odd.text)
        score_arr[int(combi[0]) - 1] += (1 / odd)
        score_arr[int(combi[1]) - 1] += (1 / odd)

    # ドロップダウンを取得
    dropdown = driver.find_element(By.CSS_SELECTOR, '#ninki_select')
    n_dropdown = len(dropdown.find_elements(By.CSS_SELECTOR, 'option'))
    select = Select(dropdown)

    if n_dropdown > 3:
        for i in range(2, n_dropdown-1, 1):
            # 3ページ目で中断
            if i == 3:
                break
            # ドロップダウンを取得し、次のオプションへ遷移
            dropdown = driver.find_element(By.CSS_SELECTOR, '#ninki_select')
            select = Select(dropdown)
            print(url)
            select.select_by_index(i)
            time.sleep(0.1)

            # 馬券とそのオッズを取得し、スコアを計算
            combis = driver.find_elements(By.CSS_SELECTOR, '.Combi01')
            odds = driver.find_elements(By.CSS_SELECTOR, '.Name_Odds')
            for combi, odd in zip(combis, odds):
                combi = combi.text.split('\n')
                odd = float(odd.text)
                score_arr[int(combi[0]) - 1] += (1 / odd)
                score_arr[int(combi[1]) - 1] += (1 / odd)

    score_arr = [ngscore if i == 0 else i for i in score_arr]
    umaren = pd.DataFrame(data=score_arr, columns=['umaren'])
    umaren.index = range(1, len(umaren)+1, 1)
    umaren['umaren'] = umaren.rank(ascending=False)['umaren'].astype(int)
    df = pd.concat([tansho, umaren], axis=1)
    progressBar += 1

    # ワイド
    url = f'https://race.netkeiba.com/odds/index.html?type=b5&race_id={race_id}&housiki=c99'
    print(url)
    driver.get(url)
    time.sleep(0.1)
    score_arr = [0] * len(umaban_list)

    # 馬券とそのオッズを取得し、スコアを計算
    combis = driver.find_elements(By.CSS_SELECTOR, '.Combi01')
    odds = driver.find_elements(By.CSS_SELECTOR, '.Name_Odds')
    for combi, odd in zip(combis, odds):
        combi = combi.text.split('\n')
        odd_sp = odd.text.split(' ')
        _odd = (float(odd_sp[0]) + float(odd_sp[1])) / 2
        score_arr[int(combi[0]) - 1] += (1 / _odd)
        score_arr[int(combi[1]) - 1] += (1 / _odd)

    # ドロップダウンを取得
    dropdown = driver.find_element(By.CSS_SELECTOR, '#ninki_select')
    n_dropdown = len(dropdown.find_elements(By.CSS_SELECTOR, 'option'))
    select = Select(dropdown)

    if n_dropdown > 3:
        for i in range(2, n_dropdown-1, 1):
            # 3ページ目で中断
            if i == 3:
                break
            # ドロップダウンを取得し、次のオプションへ遷移
            dropdown = driver.find_element(By.CSS_SELECTOR, '#ninki_select')
            select = Select(dropdown)
            print(url)
            select.select_by_index(i)
            time.sleep(0.1)

            # 馬券とそのオッズを取得し、スコアを計算
            combis = driver.find_elements(By.CSS_SELECTOR, '.Combi01')
            odds = driver.find_elements(By.CSS_SELECTOR, '.Name_Odds')
            for combi, odd in zip(combis, odds):
                combi = combi.text.split('\n')
                odd_sp = odd.text.split(' ')
                _odd = (float(odd_sp[0]) + float(odd_sp[1])) / 2
                score_arr[int(combi[0]) - 1] += (1 / _odd)
                score_arr[int(combi[1]) - 1] += (1 / _odd)

    score_arr = [ngscore if i == 0 else i for i in score_arr]
    wide = pd.DataFrame(data=score_arr, columns=['wide'])
    wide.index = range(1, len(wide)+1, 1)
    wide['wide'] = wide.rank(ascending=False)['wide'].astype(int)
    df = pd.concat([df, wide], axis=1)
    progressBar += 1

    # 馬単
    url = f'https://race.netkeiba.com/odds/index.html?type=b6&race_id={race_id}&housiki=c99'
    print(url)
    driver.get(url)
    time.sleep(0.1)
    score = base_score
    score_arr = [0] * len(umaban_list)

    # 馬券とそのオッズを取得し、スコアを計算
    combis = driver.find_elements(By.CSS_SELECTOR, '.Combi01')
    odds = driver.find_elements(By.CSS_SELECTOR, '.Name_Odds')
    for combi, odd in zip(combis, odds):
        combi = combi.text.split('\n')
        odd = float(odd.text)
        score_arr[int(combi[0]) - 1] += score * (1 / odd)
        score -= 1
        score_arr[int(combi[1]) - 1] += score * (1 / odd)
        score -= 1

    # ドロップダウンを取得
    dropdown = driver.find_element(By.CSS_SELECTOR, '#ninki_select')
    n_dropdown = len(dropdown.find_elements(By.CSS_SELECTOR, 'option'))
    select = Select(dropdown)

    if n_dropdown > 3:
        for i in range(2, n_dropdown-1, 1):
            # 2ページ目で中断
            if i == 2:
                break
            # ドロップダウンを取得し、次のオプションへ遷移
            dropdown = driver.find_element(By.CSS_SELECTOR, '#ninki_select')
            select = Select(dropdown)
            print(url)
            select.select_by_index(i)
            time.sleep(0.1)

            # 馬券とそのオッズを取得し、スコアを計算
            combis = driver.find_elements(By.CSS_SELECTOR, '.Combi01')
            odds = driver.find_elements(By.CSS_SELECTOR, '.Name_Odds')
            for combi, odd in zip(combis, odds):
                combi = combi.text.split('\n')
                odd = float(odd.text)
                score_arr[int(combi[0]) - 1] += score * (1 / odd)
                score -= 1
                score_arr[int(combi[1]) - 1] += score * (1 / odd)
                score -= 1

    score_arr = [ngscore if i == 0 else i for i in score_arr]
    umatan = pd.DataFrame(data=score_arr, columns=['umatan'])
    umatan.index = range(1, len(umatan)+1, 1)
    umatan['umatan'] = umatan.rank(ascending=False)['umatan'].astype(int)
    df = pd.concat([df, umatan], axis=1)
    progressBar += 1

    # 三連単
    url = f'https://race.netkeiba.com/odds/index.html?type=b8&race_id={race_id}&housiki=c99'
    driver.get(url)
    time.sleep(0.1)
    # ドロップダウンを取得
    dropdown = driver.find_element(By.CSS_SELECTOR, '#ninki_select')
    select = Select(dropdown)
    n_dropdown = len(dropdown.find_elements(By.CSS_SELECTOR, 'option'))
    print(f"三連単: 1 / {n_dropdown-2}")
    score = base_score
    score_arr = [0] * len(umaban_list)

    # 馬券とそのオッズを取得し、スコアを計算
    combis = driver.find_elements(By.CSS_SELECTOR, '.Combi01')
    odds = driver.find_elements(By.CSS_SELECTOR, '.Name_Odds')
    for combi, odd in zip(combis, odds):
        combi = combi.text.split('\n')
        try:
            odd = float(odd.text)
            score_arr[int(combi[0]) - 1] += score * (1 / odd) * len(umaban_list)
            score -= 1
            score_arr[int(combi[1]) - 1] += score * (1 / odd) * (len(umaban_list) - 1)
            score -= 1
            score_arr[int(combi[2]) - 1] += score * (1 / odd) * (len(umaban_list) - 2)
            score -= 1
        except:
            continue

    if n_dropdown > 3:
        for i in range(2, n_dropdown-1):
            # Xページ目で中断
            # if i == 2:
            #     break
            # ドロップダウンを取得し、次のオプションへ遷移
            dropdown = driver.find_element(By.CSS_SELECTOR, '#ninki_select')
            select = Select(dropdown)
            print(f"三連単: {i} / {n_dropdown-2}")
            select.select_by_index(i)
            time.sleep(0.5)

            # 馬券とそのオッズを取得し、スコアを計算
            combis = driver.find_elements(By.CSS_SELECTOR, '.Combi01')
            odds = driver.find_elements(By.CSS_SELECTOR, '.Name_Odds')
            for combi, odd in zip(combis, odds):
                combi = combi.text.split('\n')
                try:
                    odd = float(odd.text)
                    score_arr[int(combi[0]) - 1] += score * (1 / odd)
                    score -= 1
                    score_arr[int(combi[1]) - 1] += score * (1 / odd)
                    score -= 1
                    score_arr[int(combi[2]) - 1] += score * (1 / odd)
                    score -= 1
                except:
                    continue

    score_arr = [ngscore if i == 0 else i for i in score_arr]
    stan = pd.DataFrame(data=score_arr, columns=['santan'])
    stan.index = range(1, len(stan)+1, 1)
    stan['santan'] = stan.rank(ascending=False)['santan'].astype(int)
    df = pd.concat([df, stan], axis=1)
    progressBar += 1

    # 三連複
    url = f'https://race.netkeiba.com/odds/index.html?type=b7&race_id={race_id}&housiki=c99'
    driver.get(url)
    time.sleep(0.1)
    # ドロップダウンを取得
    dropdown = driver.find_element(By.CSS_SELECTOR, '#ninki_select')
    select = Select(dropdown)
    n_dropdown = len(dropdown.find_elements(By.CSS_SELECTOR, 'option'))
    print(f"三連複: 1 / {n_dropdown-2}")
    score_arr = [0] * len(umaban_list)

    # 馬券とそのオッズを取得し、スコアを計算
    combis = driver.find_elements(By.CSS_SELECTOR, '.Combi01')
    odds = driver.find_elements(By.CSS_SELECTOR, '.Name_Odds')
    for combi, odd in zip(combis, odds):
        combi = combi.text.split('\n')
        odd = float(odd.text)
        score_arr[int(combi[0]) - 1] += (1 / odd)
        score_arr[int(combi[1]) - 1] += (1 / odd)
        score_arr[int(combi[2]) - 1] += (1 / odd)

    if n_dropdown > 3:
        for i in range(2, n_dropdown-1):
            # 3ページ目で中断
            # if i == 3:
            #     break
            # ドロップダウンを取得し、次のオプションへ遷移
            dropdown = driver.find_element(By.CSS_SELECTOR, '#ninki_select')
            select = Select(dropdown)
            print(f"三連複: {i} / {n_dropdown-2}")
            select.select_by_index(i)
            time.sleep(0.5)

            # 馬券とそのオッズを取得し、スコアを計算
            combis = driver.find_elements(By.CSS_SELECTOR, '.Combi01')
            odds = driver.find_elements(By.CSS_SELECTOR, '.Name_Odds')
            for combi, odd in zip(combis, odds):
                combi = combi.text.split('\n')
                try:
                    odd = float(odd.text)
                except:
                    score_arr[int(combi[0]) - 1] += (1 / odd)
                    score_arr[int(combi[1]) - 1] += (1 / odd)
                    score_arr[int(combi[2]) - 1] += (1 / odd)

    score_arr = [ngscore if i == 0 else i for i in score_arr]
    spuku = pd.DataFrame(data=score_arr, columns=['sanpuku'])
    spuku.index = range(1, len(spuku)+1, 1)
    spuku['sanpuku'] = spuku.rank(ascending=False)['sanpuku'].astype(int)
    df = pd.concat([df, spuku], axis=1)
    df.reset_index(drop=True, inplace=True)
    progressBar += 1

    driver.quit()

    concat_df = pd.concat([shutuba_df, df], axis=1)
    odds_json = concat_df.to_json(orient='records', force_ascii=False)

    return odds_json

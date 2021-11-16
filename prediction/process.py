import re
import time
import pandas as pd
import numpy as np
import datetime
import requests
import platform
from scipy.special import comb
from tqdm.notebook import tqdm
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import chromedriver_binary
from selenium import webdriver
from selenium.webdriver import Chrome, ChromeOptions
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import optuna.integration.lightgbm as lgb_o
from itertools import permutations
from itertools import combinations
import IPython.core.display as display
import IPython.display
import pickle

# 訓練データと出馬表データを加工する抽象クラス


class DataProcessor:
    def __init__(self):
        self.data = pd.DataFrame()  # raw data
        self.data_p = pd.DataFrame()  # after preprocessing
        self.data_h = pd.DataFrame()  # after merging horse_results
        self.data_pe = pd.DataFrame()  # after merging peds
        self.data_c = pd.DataFrame()  # after processing categorical features

    # 馬の過去成績データの追加
    def merge_horse_results(self, hr, n_samples_list=[5, 9, 'all']):
        self.data_h = self.data_p.copy()

        for n_samples in n_samples_list:
            self.data_h = hr.merge_all(self.data_h, n_samples=n_samples)

        # 6/6追加： 馬の出走間隔追加
        self.data_h['interval'] = (
            self.data_h['date'] - self.data_h['latest']).dt.days
        self.data_h.drop(['開催', 'latest'], axis=1, inplace=True)

    def merge_peds(self, peds):
        self.data_pe = self.data_h.merge(
            peds, left_on='horse_id', right_index=True, how='left')

        self.no_peds = self.data_pe[self.data_pe['peds_0'].isnull(
        )]['horse_id'].unique()
        if len(self.no_peds) > 0:
            print('scrape peds at horse_id_list "no_peds"')

    # カテゴリ変数の処理
    def process_categorical(self, le_horse, le_jockey, results_m):
        df = self.data_pe.copy()

        # ラベルエンコーディング, horse_id, jockey_idを0始まりの整数に変換
        mask_horse = df['horse_id'].isin(le_horse.classes_)
        new_horse_id = df['horse_id'].mask(mask_horse).dropna().unique()
        le_horse.classes_ = np.concatenate([le_horse.classes_, new_horse_id])
        df['horse_id'] = le_horse.transform(df['horse_id'])
        mask_jockey = df['jockey_id'].isin(le_jockey.classes_)
        new_jockey_id = df['jockey_id'].mask(mask_jockey).dropna().unique()
        le_jockey.classes_ = np.concatenate(
            [le_jockey.classes_, new_jockey_id])
        df['jockey_id'] = le_jockey.transform(df['jockey_id'])

        # horse_id, jockey_idをpandasのcategory型に変換
        df['horse_id'] = df['horse_id'].astype('category')
        df['jockey_id'] = df['jockey_id'].astype('category')

        # その他のカテゴリ変数をpandasのcategory型に変換してからダミー変数化
        # 例を一定にするため
        weathers = results_m['weather'].unique()
        race_types = results_m['race_type'].unique()
        ground_states = results_m['ground_state'].unique()
        sexes = results_m['性'].unique()
        df['weather'] = pd.Categorical(df['weather'], weathers)
        df['race_type'] = pd.Categorical(df['race_type'], race_types)
        df['ground_state'] = pd.Categorical(df['ground_state'], ground_states)
        df['性'] = pd.Categorical(df['性'], sexes)
        df = pd.get_dummies(
            df, columns=['weather', 'race_type', 'ground_state', '性'])

        self.data_c = df


class Results(DataProcessor):
    def __init__(self, results):
        super(Results, self).__init__()
        self.data = results

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    @staticmethod
    def scrape(race_id_list):
        """
        レース結果データをスクレイピングする関数
        Parameters:
        ----------
        race_id_list : list
            レースIDのリスト
        Returns:
        ----------
        race_results_df : pandas.DataFrame
            全レース結果データをまとめてDataFrame型にしたもの
        """

        # race_idをkeyにしてDataFrame型を格納
        race_results = {}
        for race_id in tqdm(race_id_list):
            try:
                url = "https://db.netkeiba.com/race/" + race_id
                # メインとなるテーブルデータを取得
                df = pd.read_html(url)[0]

                html = requests.get(url)
                html.encoding = "EUC-JP"
                soup = bs(html.text, "html.parser")

                # 天候、レースの種類、コースの長さ、馬場の状態、日付をスクレイピング
                texts = (
                    soup.find("div", attrs={"class": "data_intro"}).find_all(
                        "p")[0].text
                    + soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text
                )
                info = re.findall(r'\w+', texts)
                for text in info:
                    if text in ["芝", "ダート"]:
                        df["race_type"] = [text] * len(df)
                    if "障" in text:
                        df["race_type"] = ["障害"] * len(df)
                    if "m" in text:
                        df["course_len"] = [
                            int(re.findall(r"\d+", text)[0])] * len(df)
                    if text in ["良", "稍重", "重", "不良"]:
                        df["ground_state"] = [text] * len(df)
                    if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                        df["weather"] = [text] * len(df)
                    if "年" in text:
                        df["date"] = [text] * len(df)

                # 馬ID、騎手IDをスクレイピング
                horse_id_list = []
                horse_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/horse")}
                )
                for a in horse_a_list:
                    horse_id = re.findall(r"\d+", a["href"])
                    horse_id_list.append(horse_id[0])
                jockey_id_list = []
                jockey_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/jockey")}
                )
                for a in jockey_a_list:
                    jockey_id = re.findall(r"\d+", a["href"])
                    jockey_id_list.append(jockey_id[0])
                df["horse_id"] = horse_id_list
                df["jockey_id"] = jockey_id_list

                # インデックスをrace_idにする
                df.index = [race_id] * len(df)

                race_results[race_id] = df
                time.sleep(1)
            # 存在しないrace_idを飛ばす
            except IndexError:
                continue
            # wifiの接続が切れた時などでも途中までのデータを返せるようにする
            except Exception as e:
                print(e)
                break
            # Jupyterで停止ボタンを押した時の対処
            except:
                break

        # pd.DataFrame型にして一つのデータにまとめる
        race_results_df = pd.concat([race_results[key]
                                    for key in race_results])

        return race_results_df

    # 前処理
    def preprocessing(self):
        df = self.data.copy()

        # 着順に数字以外の文字列が含まれているものを取り除く
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df['着順'] = df['着順'].astype(int)
        df['rank'] = df['着順'].map(lambda x: 1 if x < 4 else 0)

        # 性齢を生と年齢に分ける
        df['性'] = df['性齢'].map(lambda x: str(x)[0])
        df['年齢'] = df['性齢'].map(lambda x: str(x)[1:]).astype(int)

        # 馬体重を体重と体重変化に分ける
        df["体重"] = df["馬体重"].str.split("(", expand=True)[0]
        df["体重変化"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1]

        # errors='coerce'で、"計不"など変換できない時に欠損値にする
        df['体重'] = pd.to_numeric(df['体重'], errors='coerce')
        df['体重変化'] = pd.to_numeric(df['体重変化'], errors='coerce')

        # 単勝をfloatに変換
        df["単勝"] = df["単勝"].astype(float)
        # 距離は10の位を切り捨てる
        df["course_len"] = df["course_len"].astype(float) // 100

        # 不要な列の削除
        df.drop(['タイム', '着差', '調教師', '性齢', '馬体重', '馬名', '騎手', '人気', '着順'],
                axis=1, inplace=True)

        df['date'] = pd.to_datetime(df['date'], format='%Y年%m月%d日')

        # 開催場所
        df['開催'] = df.index.map(lambda x: str(x)[4:6])

        # 6/6出走数追加
        df['n_horses'] = df.index.map(df.index.value_counts())

        self.data_p = df

    # カテゴリ変数の処理
    def process_categorical(self):
        self.le_horse = LabelEncoder().fit(self.data_pe['horse_id'])
        self.le_jockey = LabelEncoder().fit(self.data_pe['jockey_id'])
        super().process_categorical(self.le_horse, self.le_jockey, self.data_pe)


class ShutubaTable(DataProcessor):
    def __init__(self, shutuba_tables):
        super(ShutubaTable, self).__init__()
        self.data = shutuba_tables

    @classmethod
    def scrape(cls, race_id_list):
        data = pd.DataFrame()
        for race_id in tqdm(race_id_list):
            time.sleep(0.5)
            url = 'https://race.netkeiba.com/race/shutuba.html?race_id=' + race_id
            df = pd.read_html(url)[0]
            df = df.T.reset_index(level=0, drop=True).T

            html = requests.get(url)
            html.encoding = 'EUC-JP'
            soup = bs(html.text, 'html.parser')

            texts = soup.find('div', attrs={'class': 'RaceData01'}).text
            texts = re.findall(r'\w+', texts)
            for text in texts:
                if 'm' in text:
                    df['course_len'] = [
                        int(re.findall(r'\d+', text)[0])] * len(df)
                if text in ['曇', '晴', '雨', '小雨', '小雪', '雪']:
                    df['weather'] = [text] * len(df)
                if text in ['良', '重']:
                    df['ground_state'] = [text] * len(df)
                if '不' in text:
                    df['ground_state'] = ['不良'] * len(df)
                if '稍' in text:
                    df['ground_state'] = ['稍重'] * len(df)
                if '芝' in text:
                    df['race_type'] = ['芝'] * len(df)
                if '障' in text:
                    df['race_type'] = ['障害'] * len(df)
                if 'ダ' in text:
                    df['race_type'] = ['ダート'] * len(df)

            mmdds = soup.find('dl', attrs={'id': 'RaceList_DateList'})
            mmdd = mmdds.find('dd', attrs={'class': 'Active'}).text[:5]
            mm = str(mmdd.strip('月')[0]).zfill(2)
            dd = str(mmdd.strip('月')[2]).rstrip('日').zfill(2)
            date = race_id[:4] + '/' + mm + '/' + dd

            df['date'] = [date] * len(df)
            df['start'] = texts[0]+':'+texts[1]

            # horse_id
            horse_id_list = []
            horse_td_list = soup.find_all('td', attrs={'class': 'HorseInfo'})
            for td in horse_td_list:
                horse_id = re.findall(r'\d+', td.find('a')['href'])[0]
                horse_id_list.append(horse_id)
            # jockey_id
            jockey_id_list = []
            jockey_td_list = soup.find_all('td', attrs={'class': 'Jockey'})
            for td in jockey_td_list:
                jockey_id = re.findall(r'\d+', td.find('a')['href'])[0]
                jockey_id_list.append(jockey_id)
            df['horse_id'] = horse_id_list
            df['jockey_id'] = jockey_id_list

            df.index = [race_id] * len(df)
            data = data.append(df)
        return cls(data)

    @classmethod
    def scrape2(cls, race_id_list, date):
        data = pd.DataFrame()
        options = webdriver.ChromeOptions()
        options.add_argument("headless")
        options.add_argument("window-size=1200x600")
        wd = webdriver.Chrome(options=options)

        for race_id in tqdm(race_id_list):
            url = 'https://race.netkeiba.com/race/shutuba.html?race_id=' + race_id
            wd.get(url)
            df = pd.read_html(wd.page_source)[
                0].T.reset_index(level=0, drop=True).T

            texts = wd.find_element_by_class_name('RaceData01').text
            texts = re.findall(r'\w+', texts)
            for text in texts:
                if 'm' in text:
                    df['course_len'] = [
                        int(re.findall(r'\d+', text)[0])] * len(df)
                if text in ['曇', '晴', '雨', '小雨', '小雪', '雪']:
                    df['weather'] = [text] * len(df)
                if text in ['良', '重']:
                    df['ground_state'] = [text] * len(df)
                if '不' in text:
                    df['ground_state'] = ['不良'] * len(df)
                if '稍' in text:
                    df['ground_state'] = ['稍重'] * len(df)
                if '芝' in text:
                    df['race_type'] = ['芝'] * len(df)
                if '障' in text:
                    df['race_type'] = ['障害'] * len(df)
                if 'ダ' in text:
                    df['race_type'] = ['ダート'] * len(df)
            df['date'] = [date] * len(df)
            df['start'] = texts[0]+':'+texts[1]

            # horse_id
            horse_id_list = []
            horse_td_list = wd.find_elements_by_css_selector('td.HorseInfo')
            for td in horse_td_list:
                href = td.find_element_by_tag_name('a').get_attribute('href')
                horse_id = re.findall(r'\d+', href)[0]
                horse_id_list.append(horse_id)
            # jockey_id
            jockey_id_list = []
            jockey_td_list = wd.find_elements_by_css_selector('td.Jockey')
            for td in jockey_td_list:
                href = td.find_element_by_tag_name('a').get_attribute('href')
                jockey_id = re.findall(r'\d+', href)[0]
                jockey_id_list.append(jockey_id)
            df['horse_id'] = horse_id_list
            df['jockey_id'] = jockey_id_list

            df.index = [race_id] * len(df)
            data = data.append(df)

        return cls(data)

    def preprocessing(self):
        df = self.data.copy()

        # 性齢を生と年齢に分ける
        df['性'] = df['性齢'].map(lambda x: str(x)[0])
        df['年齢'] = df['性齢'].map(lambda x: str(x)[1:]).astype(int)

        # 馬体重を体重と体重変化に分ける
        df = df[df["馬体重(増減)"] != '--']
        df["体重"] = df["馬体重(増減)"].str.split("(", expand=True)[0].astype(int)
        df["体重変化"] = df["馬体重(増減)"].str.split("(", expand=True)[1].str[:-1]
        # 2020/12/13追加：増減が「前計不」などのとき欠損値にする
        df['体重変化'] = pd.to_numeric(df['体重変化'], errors='coerce')

        df['date'] = pd.to_datetime(df['date'])

        df['枠'] = df['枠'].astype(int)
        df['馬番'] = df['馬番'].astype(int)
        df['斤量'] = df['斤量'].astype(int)

        df['開催'] = df.index.map(lambda x: str(x)[4:6])

        # 6/6出走数追加
        df['n_horses'] = df.index.map(df.index.value_counts())

        # 使用する列を選択
        df = df[['枠', '馬番', '斤量', 'course_len', 'weather', 'race_type',
                'ground_state', 'date', 'horse_id', 'jockey_id', '性', '年齢',
                 '体重', '体重変化', '開催', 'n_horses']]

        self.data_p = df.rename(columns={'枠': '枠番'})


class HorseResults:
    def __init__(self, horse_results):
        self.horse_results = horse_results[[
            '日付', '着順', '賞金', '着差', '通過', '開催', '距離', '上り']]
        self.preprocessing()

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    @staticmethod
    def scrape(horse_id_list):
        """
        馬の過去成績データをスクレイピングする関数
        Parameters:
        ----------
        horse_id_list : list
            馬IDのリスト
        Returns:
        ----------
        horse_results_df : pandas.DataFrame
            全馬の過去成績データをまとめてDataFrame型にしたもの
        """

        # horse_idをkeyにしてDataFrame型を格納
        horse_results = {}
        for horse_id in tqdm(horse_id_list):
            try:
                time.sleep(0.5)
                url = 'https://db.netkeiba.com/horse/' + horse_id
                df = pd.read_html(url)[3]
                # 受賞歴がある馬の場合、3番目に受賞歴テーブルが来るため、4番目のデータを取得する
                if df.columns[0] == '受賞歴':
                    df = pd.read_html(url)[4]
                df.index = [horse_id] * len(df)
                horse_results[horse_id] = df
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        # pd.DataFrame型にして一つのデータにまとめる
        horse_results_df = pd.concat(
            [horse_results[key] for key in horse_results])

        return horse_results_df

    def preprocessing(self):
        df = self.horse_results.copy()

        # 着順に数字以外の文字列が含まれているものを取り除く
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df['着順'] = df['着順'].astype(int)

        df['date'] = pd.to_datetime(df['日付'])
        df.drop(['日付'], axis=1, inplace=True)

        # 賞金のNaNを0で埋める
        df['賞金'].fillna(0, inplace=True)

        # 1着の着差を0にする
        df['着差'] = df['着差'].map(lambda x: 0 if x < 0 else x)

        # レース展開データ
        # n=1: 最初のコーナー位置 n=4: 最終コーナー位置
        def corner(x, n):
            if type(x) != str:
                return x
            elif n == 4:
                return int(re.findall(r'\d+', x)[-1])
            elif n == 1:
                return int(re.findall(r'\d+', x)[0])
        df['first_corner'] = df['通過'].map(lambda x: corner(x, 1))
        df['final_corner'] = df['通過'].map(lambda x: corner(x, 4))

        df['final_to_rank'] = df['final_corner'] - df['着順']
        df['first_to_rank'] = df['first_corner'] - df['着順']
        df['first_to_final'] = df['first_corner'] - df['final_corner']

        # 開催場所
        df['開催'] = df['開催'].str.extract(
            r'(\D+)')[0].map(place_dict).fillna('11')
        # race_type
        df['race_type'] = df['距離'].str.extract(r'(\D+)')[0].map(race_type_dict)
        # 距離は10の位を切り捨てる
        df['course_len'] = df['距離'].str.extract(r'(\d+)').astype(int) // 100
        df.drop(['距離'], axis=1, inplace=True)
        # インデックス名を与える
        df.index.name = 'horse_id'

        self.horse_results = df
        self.target_list = ['着順', '賞金', '着差', 'first_corner',
                            'first_to_rank', 'first_to_final', 'final_to_rank']

    # n_samplesレース分馬ごとに平均する
    def average(self, horse_id_list, date, n_samples='all'):
        target_df = self.horse_results.query('index in @horse_id_list')

        # 過去何走分取り出すか指定
        if n_samples == 'all':
            filtered_df = target_df[target_df['date'] < date]
        elif n_samples > 0:
            filtered_df = target_df[target_df['date'] < date].\
                sort_values('date', ascending=False).groupby(
                    level=0).head(n_samples)
        else:
            raise Exception('n_samples must be >0')

        # 集計して辞書型に入れる
        self.average_dict = {}
        self.average_dict['non_category'] = filtered_df.groupby(level=0)[self.target_list].mean()\
            .add_suffix(f'_{n_samples}R')
        for column in ['race_type', 'course_len', '開催']:
            self.average_dict[column] = filtered_df.groupby(['horse_id', column])[
                self.target_list].mean().add_suffix(f'_{column}_{n_samples}R')

        # 6/6追加: 馬の出走間隔追加のために、全レースの日付を変数latestに格納
        if n_samples == 5:
            self.latest = filtered_df.groupby(
                'horse_id')['date'].max().rename('latest')

    def merge(self, results, date, n_samples='all'):
        df = results[results['date'] == date]
        horse_id_list = df['horse_id']
        self.average(horse_id_list, date, n_samples)
        merged_df = df.merge(self.average_dict['non_category'], left_on='horse_id',
                             right_index=True, how='left')
        for column in ['race_type', 'course_len', '開催']:
            merged_df = merged_df.merge(self.average_dict[column],
                                        left_on=['horse_id', column],
                                        right_index=True, how='left')

        # 6/6追加：馬の出走間隔追加のために、全レースの日付を変数latestに格納
        if n_samples == 5:
            merged_df = merged_df.merge(self.latest, left_on='horse_id',
                                        right_index=True, how='left')
        return merged_df

    def merge_all(self, results, n_samples='all'):
        date_list = results['date'].unique()
        merged_df = pd.concat([self.merge(results, date, n_samples)
                              for date in stqdm(date_list)])
        return merged_df

    def merge_agari(self, results, date):
        df = results[results['date'] == date]
        horse_id_list = df['horse_id']
        target_df = self.horse_results.query('index in @horse_id_list')
        filtered_df = target_df[target_df['date'] == date]
        agari_s = filtered_df['上り']
        merged_df = df.merge(agari_s, left_on='horse_id',
                             right_index=True, how='left')
        return merged_df

    def merge_all_agari(self, results):
        date_list = results['date'].unique()
        merged_df = pd.concat([self.merge_agari(results, date)
                               for date in tqdm(date_list)])
        return merged_df

    def merge_agari_sum(self, results, horse_id):
        filtered_df = results[results['horse_id'] == horse_id]
        date_list = filtered_df['date'].to_list()

        agari_5R = []
        agari_9R = []
        agari_All = []
        for date in filtered_df['date']:
            target_df = filtered_df[filtered_df['date'] < date].sort_values(
                'date', ascending=False).head(5)
            agari_sum = target_df['上り'].sum()
            agari_5R.append(agari_sum)
            target_df = filtered_df[filtered_df['date'] < date].sort_values(
                'date', ascending=False).head(9)
            agari_sum = target_df['上り'].sum()
            agari_9R.append(agari_sum)
            target_df = filtered_df[filtered_df['date'] < date]
            agari_sum = target_df['上り'].sum()
            agari_All.append(agari_sum)

        df = filtered_df.copy()
        df['上り5R'] = agari_5R
        df['上り9R'] = agari_9R
        df['上りAll'] = agari_All
        return df

    def merge_all_agari_sum(self, results):
        horse_id_list = results['horse_id'].unique()
        merged_df = pd.concat([self.merge_agari_sum(results, horse_id)
                              for horse_id in tqdm(horse_id_list)])
        return merged_df


class Peds:
    def __init__(self, peds):
        self.peds = peds
        self.peds_e = pd.DataFrame()  # after label encoding and transforming into category

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    @staticmethod
    def scrape(horse_id_list):
        """
        血統データをスクレイピングする関数

        Parameters:
        ----------
        horse_id_list : list
            馬IDのリスト

        Returns:
        ----------
        peds_df : pandas.DataFrame
            全血統データをまとめてDataFrame型にしたもの
        """

        peds_dict = {}
        for horse_id in tqdm(horse_id_list):
            try:
                time.sleep(0.5)
                url = "https://db.netkeiba.com/horse/ped/" + horse_id
                df = pd.read_html(url)[0]

                # 重複を削除して1列のSeries型データに直す
                generations = {}
                for i in reversed(range(5)):
                    generations[i] = df[i]
                    df.drop([i], axis=1, inplace=True)
                    df = df.drop_duplicates()
                ped = pd.concat([generations[i]
                                for i in range(5)]).rename(horse_id)

                peds_dict[horse_id] = ped.reset_index(drop=True)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #列名をpeds_0, ..., peds_61にする
        peds_df = pd.concat([peds_dict[key]
                            for key in peds_dict], axis=1).T.add_prefix('peds_')

        return peds_df

    def encode(self):
        df = self.peds.copy()
        for column in df.columns:
            df[column] = LabelEncoder().fit_transform(df[column].fillna('Na'))
        self.peds_e = df.astype('category')


class Return:
    def __init__(self, return_tables):
        self.return_tables = return_tables

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    @staticmethod
    def scrape(race_id_list, pre_return_tables={}):
        """
        払い戻し表データをスクレイピングする関数
        Parameters:
        ----------
        race_id_list : list
            レースIDのリスト
        Returns:
        ----------
        return_tables_df : pandas.DataFrame
            全払い戻し表データをまとめてDataFrame型にしたもの
        """

        return_tables = pre_return_tables
        for race_id in tqdm(race_id_list):
            if race_id in return_tables.keys():
                continue
            try:
                time.sleep(0.5)
                url = "https://db.netkeiba.com/race/" + race_id

                # 普通にスクレイピングすると複勝やワイドなどが区切られないで繋がってしまう。
                # そのため、改行コードを文字列brに変換して後でsplitする
                f = urlopen(url)
                html = f.read()
                html = html.replace(b'<br />', b'br')
                dfs = pd.read_html(html)

                # dfsの1番目に単勝〜馬連、2番目にワイド〜三連単がある
                df = pd.concat([dfs[1], dfs[2]])

                df.index = [race_id] * len(df)
                return_tables[race_id] = df
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        # pd.DataFrame型にして一つのデータにまとめる
        #return_tables_df = pd.concat([return_tables[key] for key in return_tables])
        return return_tables

    @property
    def fukusho(self):
        fukusho = self.return_tables[self.return_tables[0] == '複勝'][[1, 2]]
        wins = fukusho[1].str.split('br', expand=True)[[0, 1, 2]]

        wins.columns = ['win_0', 'win_1', 'win_2']
        returns = fukusho[2].str.split('br', expand=True)[[0, 1, 2]]
        returns.columns = ['return_0', 'return_1', 'return_2']

        df = pd.concat([wins, returns], axis=1)
        for column in df.columns:
            df[column] = df[column].str.replace(',', '')
        return df.fillna(0).astype(int)

    @property
    def tansho(self):
        tansho = self.return_tables[self.return_tables[0] == '単勝'][[1, 2]]
        tansho.columns = ['win', 'return']

        for column in tansho.columns:
            tansho[column] = pd.to_numeric(tansho[column], errors='coerce')

        return tansho

    @property
    def umaren(self):
        umaren = self.return_tables[self.return_tables[0] == '馬連'][[1, 2]]
        wins = umaren[1].str.split('-', expand=True)[[0, 1]].add_prefix('win_')
        return_ = umaren[2].rename('return')
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    @property
    def umatan(self):
        umatan = self.return_tables[self.return_tables[0] == '馬単'][[1, 2]]
        wins = umatan[1].str.split('→', expand=True)[[0, 1]].add_prefix('win_')
        return_ = umatan[2].rename('return')
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    @property
    def wide(self):
        wide = self.return_tables[self.return_tables[0] == 'ワイド'][[1, 2]]
        wins = wide[1].str.split('br', expand=True)[[0, 1, 2]]
        wins = wins.stack().str.split('-', expand=True).add_prefix('win_')
        return_ = wide[2].str.split('br', expand=True)[[0, 1, 2]]
        return_ = return_.stack().rename('return')
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x.str.replace(',', ''), errors='coerce'))

    @property
    def sanrentan(self):
        rentan = self.return_tables[self.return_tables[0] == '三連単'][[1, 2]]
        wins = rentan[1].str.split('→', expand=True)[
            [0, 1, 2]].add_prefix('win_')
        return_ = rentan[2].rename('return')
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    @property
    def sanrenpuku(self):
        renpuku = self.return_tables[self.return_tables[0] == '三連複'][[1, 2]]
        wins = renpuku[1].str.split(
            '-', expand=True)[[0, 1, 2]].add_prefix('win_')
        return_ = renpuku[2].rename('return')
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))


class ModelEvaluator:
    def __init__(self, model, return_tables_path_list):
        self.model = model
        self.rt = Return.read_pickle(return_tables_path_list)
        self.fukusho = self.rt.fukusho
        self.tansho = self.rt.tansho
        self.umaren = self.rt.umaren
        self.umatan = self.rt.umatan
        self.wide = self.rt.wide
        self.sanrentan = self.rt.sanrentan
        self.sanrenpuku = self.rt.sanrenpuku

    # 3着以内に入る確率を予測
    def predict_proba(self, X, train=True, std=True, minmax=False):
        # 実際に出馬表データを入れて予測するときはtrain=False
        if train:
            proba = pd.Series(
                self.model.predict_proba(X.drop(['単勝'], axis=1))[
                    :, 1], index=X.index
            )
        else:
            proba = pd.Series(
                self.model.predict_proba(X)[:, 1], index=X.index
            )
        if std:
            # レース内で標準化して、相対評価する。「レース内偏差値」みたいなもの。
            def standard_scaler(x): return (x - x.mean()) / x.std()
            proba = proba.groupby(level=0).transform(standard_scaler)
        if minmax:
            # データ全体を0~1にする
            proba = (proba - proba.min()) / (proba.max() - proba.min())
        return proba

    # 0か1かを予測
    def predict(self, X, threshold=0.5):
        y_pred = self.predict_proba(X)
        return [0 if p < threshold else 1 for p in y_pred]

    def score(self, y_true, X):
        return roc_auc_score(y_true, self.predict_proba(X))

    def feature_importance(self, X, n_display=20):
        importances = pd.DataFrame({"features": X.columns,
                                    "importance": self.model.feature_importances_})
        return importances.sort_values("importance", ascending=False)[:n_display]

    def pred_table(self, X, threshold=0.5, bet_only=True):
        pred_table = X.copy()[['馬番', '単勝']]
        pred_table['pred'] = self.predict(X, threshold)
        if bet_only:
            return pred_table[pred_table['pred'] == 1][['馬番', '単勝']]
        else:
            return pred_table

    def bet(self, race_id, kind, umaban, amount):
        if kind == 'fukusho':
            rt_1R = self.fukusho.loc[race_id]
            return_ = (rt_1R[['win_0', 'win_1', 'win_2']] == umaban).values * \
                rt_1R[['return_0', 'return_1', 'return_2']].values * amount/100
            return_ = np.sum(return_)
        if kind == 'tansho':
            rt_1R = self.tansho.loc[race_id]
            return_ = (rt_1R['win'] == umaban) * rt_1R['return'] * amount/100
        if kind == 'umaren':
            rt_1R = self.umaren.loc[race_id]
            return_ = (set(rt_1R[['win_0', 'win_1']]) == set(umaban)) \
                * rt_1R['return']/100 * amount
        if kind == 'umatan':
            rt_1R = self.umatan.loc[race_id]
            return_ = (list(rt_1R[['win_0', 'win_1']]) == list(umaban))\
                * rt_1R['return']/100 * amount
        if kind == 'wide':
            rt_1R = self.wide.loc[race_id]
            return_ = (rt_1R[['win_0', 'win_1']].
                       apply(lambda x: set(x) == set(umaban), axis=1)) \
                * rt_1R['return']/100 * amount
            return_ = return_.sum()
        if kind == 'sanrentan':
            rt_1R = self.sanrentan.loc[race_id]
            return_ = (list(rt_1R[['win_0', 'win_1', 'win_2']]) == list(umaban)) * \
                rt_1R['return']/100 * amount
        if kind == 'sanrenpuku':
            rt_1R = self.sanrenpuku.loc[race_id]
            return_ = (set(rt_1R[['win_0', 'win_1', 'win_2']]) == set(umaban)) \
                * rt_1R['return']/100 * amount
        if not (return_ >= 0):
            return_ = amount
        return return_

    def fukusho_return(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = len(pred_table)

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(np.sum([
                self.bet(race_id, 'fukusho', umaban, 1) for umaban in preds['馬番']
            ]))
        return_rate = np.sum(return_list) / n_bets
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x > 0 for x in return_list])
        return n_bets, return_rate, n_hits, std

    def tansho_return(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = len(pred_table)

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(
                np.sum([self.bet(race_id, 'tansho', umaban, 1) for umaban in preds['馬番']]))

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets

        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def tansho_return_odds(self, X, threshold=0.5, odds=15):
        pred_table = self.pred_table(X, threshold)
        pred_table = pred_table[pred_table['単勝'] > odds]
        n_bets = len(pred_table)

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(
                np.sum([self.bet(race_id, 'tansho', umaban, 1) for umaban in preds['馬番']]))

        if len(return_list) > 0:
            std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets

            n_hits = np.sum([x > 0 for x in return_list])
            return_rate = np.sum(return_list) / n_bets
        else:
            std, n_hits, return_rate = 0, 0, 0

        return n_bets, return_rate, n_hits, std

    # 単勝適正回収値：　払い戻し金額が常に一定になるように単勝で賭けた場合の回収率
    def tansho_return_proper(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = len(pred_table)

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(np.sum(preds.apply(lambda x: self.bet(
                race_id, 'tansho', x['馬番'], 1/x['単勝']), axis=1)))

        bet_money = (1 / pred_table['単勝']).sum()

        std = np.std(return_list) * np.sqrt(len(return_list)) / bet_money

        n_hits = np.sum([x > 0 for x in return_list])
        return_rate = np.sum(return_list) / bet_money
        return n_bets, return_rate, n_hits, std

    def umaren_box(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            # 賭けたい馬が1頭しかいない時は単勝で賭ける

            if len(preds) >= 2:
                for umaban in combinations(preds['馬番'], 2):
                    return_ += self.bet(race_id, 'umaren', umaban, 1)
                    n_bets += 1
                return_list.append(return_)
            else:
                continue

        if len(return_list) > 0:
            std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
            n_hits = np.sum([x > 0 for x in return_list])
            return_rate = np.sum(return_list) / n_bets
        else:
            std, n_hits, return_rate = 0, 0, 0

    def umatan_box(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            # 賭けたい馬が1頭しかいない時は単勝で賭ける

            if len(preds) >= 2:
                for umaban in permutations(preds['馬番'], 2):
                    return_ += self.bet(race_id, 'umatan', umaban, 1)
                    n_bets += 1
                return_list.append(return_)
            else:
                continue

        if len(return_list) > 0:
            std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
            n_hits = np.sum([x > 0 for x in return_list])
            return_rate = np.sum(return_list) / n_bets
        else:
            std, n_hits, return_rate = 0, 0, 0

    def wide_box(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            # 賭けたい馬が1頭しかいない時は単勝で賭ける

            if len(preds) >= 2:
                for umaban in combinations(preds['馬番'], 2):
                    return_ += self.bet(race_id, 'wide', umaban, 1)
                    n_bets += 1
                return_list.append(return_)
            else:
                continue

        if len(return_list) > 0:
            std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
            n_hits = np.sum([x > 0 for x in return_list])
            return_rate = np.sum(return_list) / n_bets
        else:
            std, n_hits, return_rate = 0, 0, 0

        return n_bets, return_rate, n_hits, std

    def sanrentan_box(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            # 賭けたい馬が3頭いない時は単勝で賭ける

            if len(preds) >= 3:
                for umaban in permutations(preds['馬番'], 3):
                    return_ += self.bet(race_id, 'sanrentan', umaban, 1)
                    n_bets += 1
                return_list.append(return_)
            else:
                continue

        if len(return_list) > 0:
            std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
            n_hits = np.sum([x > 0 for x in return_list])
            return_rate = np.sum(return_list) / n_bets
        else:
            std, n_hits, return_rate = 0, 0, 0

        return n_bets, return_rate, n_hits, std

    def sanrentan_formation(self, X, threshold=0.5):
        pred = self.predict_proba(X)
        pred_table = X[['馬番', '単勝']].copy()
        pred_table['pred'] = pred
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            umaban = preds.sort_values('pred', ascending=False)['馬番']
            return_ = 0
            # 一番predの高い馬がthreshold以上のときに賭ける
            if preds['pred'].max() > threshold:
                for i in umaban[1:3]:
                    for j in umaban[1:6]:
                        if i == j:
                            continue
                        else:
                            baken = [umaban[0], i, j]
                            return_ += self.bet(race_id, 'sanrentan', baken, 1)
                            n_bets += 1
                return_list.append(return_)
            else:
                continue

        if len(return_list) > 0:
            std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
            n_hits = np.sum([x > 0 for x in return_list])
            return_rate = np.sum(return_list) / n_bets
        else:
            std, n_hits, return_rate = 0, 0, 0

        return n_bets, return_rate, n_hits, std

    def sanrenpuku_box(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            # 賭けたい馬が3頭いない時は単勝で賭ける

            if len(preds) >= 3:
                for umaban in combinations(preds['馬番'], 3):
                    return_ += self.bet(race_id, 'sanrenpuku', umaban, 1)
                    n_bets += 1
                return_list.append(return_)
            else:
                continue

        if len(return_list) > 0:
            std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
            n_hits = np.sum([x > 0 for x in return_list])
            return_rate = np.sum(return_list) / n_bets
        else:
            std, n_hits, return_rate = 0, 0, 0

        return n_bets, return_rate, n_hits, std


def update_data(old, new):
    filtered_old = old[~old.index.isin(new.index)]
    return pd.concat([filtered_old, new])


def concat_pd(data):
    for key in data.keys():
        data[key].index = [key]*len(data[key])
    return pd.concat([data[key] for key in data.keys()], sort=False)


def concat_peds(peds_dict):
    peds = pd.concat([peds_dict[key]
                      for key in peds_dict], axis=1).T.add_prefix('peds_')
    return peds


def split_data(df, test_size=0.3):
    sorted_id_list = df.sort_values('date').index.unique()
    train_id_list = sorted_id_list[:round(len(sorted_id_list) * (1-test_size))]
    test_id_list = sorted_id_list[round(len(sorted_id_list) * (1-test_size)):]
    train = df.loc[train_id_list]  # .drop(['date'], axis=1)
    test = df.loc[test_id_list]  # .drop(['date'], axis=1)
    return train, test


def gain(return_func, X, n_samples=100, t_range=[0.5, 3.5]):
    gain = {}
    for i in tqdm(range(n_samples)):
        # t_rangeの範囲をn_samples等分したものを、thresholdとしてfor分で回す
        threshold = t_range[1] * i / n_samples + t_range[0] * (1-(i/n_samples))
        n_bets, return_rate, n_hits, std = return_func(X, threshold)
        if n_bets > 2:
            gain[threshold] = {
                'return_rate': return_rate,
                'n_hits': n_hits,
                'std': std,
                'n_bets': n_bets
            }
    return pd.DataFrame(gain).T


def plot(df, label=''):
    plt.fill_between(
        df.index, y1=df['return_rate']-df['std'],
        y2=df['return_rate']+df['std'],
        alpha=0.2
    )
    plt.plot(df.index, df['return_rate'], label=label)
    plt.legend()
    plt.grid(True)


place_dict = {
    '札幌': '01', '函館': '02', '福島': '03', '新潟': '04', '東京': '05',
    '中山': '06', '中京': '07', '京都': '08', '阪神': '09', '小倉': '10',
}

race_type_dict = {
    '芝': '芝', 'ダ': 'ダート', '障': '障害'
}


def beep(freq, dur=100):
    """
        ビープ音を鳴らす.
        @param freq 周波数
        @param dur  継続時間（ms）
    """
    if platform.system() == "Windows":
        # Windowsの場合は、winsoundというPython標準ライブラリを使います.
        import winsound
        winsound.Beep(freq, dur)
    else:
        # Macの場合には、Macに標準インストールされたplayコマンドを使います.
        import os
        os.system('play -n synth %s sin %s' % (dur/1000, freq))


def scrape_results(race_id_list):
    tansho_list = []
    fukusho_list = []
    for race_id in tqdm(race_id_list):
        time.sleep(0.5)
        url = 'https://race.netkeiba.com/race/result.html?race_id=' + race_id
        tansho = int(pd.read_html(url)[1].loc[0, 1])
        fukusho = pd.read_html(url)[1].loc[1, 1]
        tansho_list.append(tansho)
        fukusho_list.append(fukusho)
    return tansho_list, fukusho_list

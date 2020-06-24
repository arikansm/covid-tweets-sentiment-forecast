import sys
import logging
logging.disable(sys.maxsize)
import warnings
warnings.filterwarnings('ignore')
import os
import re
import string
from collections import Counter
import datetime as dt
import pandas as pd
import numpy as np
from fbprophet import Prophet
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.util import *
from nltk.corpus import words
from matplotlib import pyplot as plt

class DisableCPlusLogs(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

class CoronaTweetsSentimentAnalysis():
    loaded_tweet_count = 0
    
    def __init__(self):
        self.loaded_tweet_count = 0
        
    def get_sentiment_analysis(self, dir):
        print("> " + dir + " dizini inceleniyor...")
        sentiments = []
        day_number = 1
        for file in sorted(os.listdir(dir)):
            sentiment_result = self.analyze_file(dir, file, day_number)
            sentiments.append(sentiment_result)
            day_number += 1
        
        daily_analysis = pd.DataFrame(sentiments)
        return daily_analysis

    def analyze_file(self, dir, file, day_number):
        print("  > Dosya okunuyor: " + str(file), flush=True)
        tweets = []
        content = self.get_tweets_from_csv(dir + '/' + file)
        tweets.append(content)
        
        tweets_dataframe = pd.concat(tweets)
        texts = tweets_dataframe['text']
        preprocessed = self.preprocess(texts)

        day_title = str(file)[:str(file).index(" Covid")]
        sentiment_result = self.sentiment_analysis(preprocessed, day_number, day_title)
        return sentiment_result

    def preprocess(self, texts):
        print("    > Onislem sureci isletiliyor", flush=True)
        print("      > URL adresleri siliniyor", flush=True)
        texts_without_uri = texts.apply(lambda x: re.sub(r"https\S+", "", str(x)))

        print("      > Tum karakterler kucuk harfe donusturuluyor", flush=True)
        texts_lowercase_no_uri = texts_without_uri.apply(lambda x: x.lower())

        print("      > Noktalama isaretleri siliniyor", flush=True)
        texts_lowercase_without_uri_puncts = texts_lowercase_no_uri.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        texts_lowercase_without_uri_puncts = texts_lowercase_no_uri.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

        print("      > Gereksiz kelimeler (stopwords) siliniyor", flush=True)
        stop_words = set(stopwords.words('english'))
        stop_words.update(['#coronavirus', '#coronavirusoutbreak', '#coronavirusPandemic', '#covid19', '#covid_19', '#epitwitter', '#ihavecorona', 'amp', 'coronavirus', 'covid19'])
        texts_lowercase_without_uri_puncts_stopwords = texts_lowercase_without_uri_puncts.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        return texts_lowercase_without_uri_puncts_stopwords

    def sentiment_analysis(self, texts_preprocessed, day_number, day_title):
        print("    > Duygu analizi yapiliyor", flush=True)
        sentiment_analyzer = SentimentIntensityAnalyzer()
        scores = texts_preprocessed.apply(lambda x: sentiment_analyzer.polarity_scores(x))
        scores_dataframe = pd.DataFrame(list(scores))
        scores_dataframe['val'] = scores_dataframe['compound'].apply(lambda x: 'notr' if x == 0 else ('olumlu' if x > 0 else 'olumsuz'))

        sentiment_result = {}
        sentiment_result["tarih"] = str(day_number) + ". gun (" + day_title + ")"
        sentiment_result["tarih_prophet"] = str(dt.datetime.now() + + dt.timedelta(days=day_number))
        sentiment_result["notr"] = 0
        sentiment_result["olumlu"] = 0
        sentiment_result["olumsuz"] = 0
        
        tag_index = 0
        counts = pd.DataFrame.from_dict(Counter(scores_dataframe['val']), orient = 'index').reset_index()
        print("        > Siniflandirma sonucu:", flush=True)
        for tag in counts["index"]:
            sentiment_result[tag] = counts[0][tag_index]
            print("          > " + tag + ": " + f"{sentiment_result[tag]:,}", flush=True)
            tag_index = tag_index + 1
        return sentiment_result

    def forecast_eighth_day(self, daily_analysis):
        print("> 8. gun paylasilmasi muhtemel tweetlerin duygulari tahmin ediliyor...")
        fc_notr = self.forecast(daily_analysis, "notr")
        fc_olumlu = self.forecast(daily_analysis,  "olumlu")
        fc_olumsuz = self.forecast(daily_analysis, "olumsuz")
        return fc_notr["yhat"], fc_olumlu["yhat"], fc_olumsuz["yhat"]

    def forecast(self, daily_analysis, column):
        da = daily_analysis[["tarih_prophet", column]]
        da.columns = ["ds", "y"]
        prediction_algorithm = Prophet(yearly_seasonality=False, weekly_seasonality = False, daily_seasonality = True, seasonality_mode = 'additive')
        with DisableCPlusLogs():
            prediction_algorithm.fit(da)
        future_dataframe = prediction_algorithm.make_future_dataframe(periods=1)
        forecast_values = prediction_algorithm.predict(future_dataframe)
        forecast_results = forecast_values[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        return forecast_results.tail(1)

    def plot_daily_figure(self, daily_analysis):
        plt.figure(figsize=(16,5), dpi=100)
        plt.plot(daily_analysis.tarih, daily_analysis.notr, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label='notr')
        plt.plot(daily_analysis.tarih, daily_analysis.olumsuz, marker='o', markerfacecolor='red', markersize=12, color='lightsalmon', linewidth=4, label='olumsuz')
        plt.plot(daily_analysis.tarih, daily_analysis.olumlu, marker='o', markerfacecolor='green', markersize=12, color='lightgreen', linewidth=4, label='olumlu')
        plt.gca().set(title="Covid-19 İlişkili Tweetler için Günlük Duygu Analizi Sonuçları", xlabel="(toplam " + f"{self.loaded_tweet_count:,}" + " adet tweet incelenmistir)", ylabel="Tweet Sayisi")
        plt.legend()
        plt.grid()
        plt.show()
        print("  > Analiz sonuclari cizdirildi.", flush=True)

    def plot_forecast_figure(self, notr, olumlu, olumsuz):
        self.plot_bar_chart(notr, olumlu, olumsuz, '8. Gün Duygu Tahmini')
        print("  > Tahmin sonuclari cizdirildi.", flush=True)

    def plot_real_values_figure(self, notr, olumlu, olumsuz):
        self.plot_bar_chart(notr, olumlu, olumsuz, '8. Gün Gercek Duygu Degerleri')
        print("  > 8. gun icin gercek degerler cizdirildi.", flush=True)
    
    def plot_bar_chart(self, notr, olumlu, olumsuz, title):
        fig, ax = plt.subplots()
        index = np.arange(1)
        bar_width = 0.2
        opacity = 0.8
        bar_notr = plt.bar(index, tuple(notr), bar_width, alpha=opacity, color='b', label='notr')
        bar_olumlu = plt.bar(index + bar_width, tuple(olumlu), bar_width, alpha=opacity, color='g', label='olumlu')
        bar_olumsuz = plt.bar(index + bar_width + bar_width, tuple(olumsuz), bar_width, alpha=opacity, color='r', label='olumsuz')
        self.auto_set_bar_label(bar_notr, ax)
        self.auto_set_bar_label(bar_olumlu, ax)
        self.auto_set_bar_label(bar_olumsuz, ax)
        plt.title(title)
        plt.xticks(index + bar_width, (''))
        plt.legend()
        plt.tight_layout()
        plt.show()

    def auto_set_bar_label(self, bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 0.9*height, f"{int(height):,}", ha='center', va='bottom')

    def get_tweets_from_csv(self, file_path):
        csv_file_records = pd.read_csv(file_path, lineterminator = '\n')
        csv_file_records = csv_file_records.loc[csv_file_records.lang=='en', :]
        tweet_count = len(csv_file_records)
        print("    > " + f"{tweet_count:,}" + " adet tweet bulundu.", flush=True)
        self.loaded_tweet_count += tweet_count
        return csv_file_records

if __name__ == "__main__":
    analyzer = CoronaTweetsSentimentAnalysis()

    train_dir = './dataset/egitim_gunleri'
    train_results = analyzer.get_sentiment_analysis(train_dir)
    analyzer.plot_daily_figure(train_results)
    
    notr, olumlu, olumsuz = analyzer.forecast_eighth_day(train_results)
    analyzer.plot_forecast_figure(notr, olumlu, olumsuz)

    test_dir = './dataset/test_gunu'
    test_results = analyzer.get_sentiment_analysis(test_dir)
    analyzer.plot_real_values_figure(test_results["notr"], test_results["olumlu"], test_results["olumsuz"])

import pandas as pd
import numpy as np

covid = pd.read_csv("./data/Covid19SidoInfState.csv")
covid.head()


#서울지역 데이터 분리하기
seoul = covid.loc[covid.gubun == "서울", ["gubun", "stdDay", "incDec"]].copy()
seoul.head()

seoul = seoul.reset_index()
seoul.head()

date = pd.to_datetime(seoul.stdDay[0], format = "%Y년 %m월 %d일 %H시")
getattr(date, "year")


#날짜 데이터 추가하기
year = []
month = []
day = []
for n in seoul.stdDay:
    try:
        date = pd.to_datetime(n, format = "%Y년 %m월 %d일 %H시")
    except:
        date = pd.to_datetime(n, format = "%Y년 %m년 %d일 %H시")
    year.append(getattr(date, "year"))
    month.append(getattr(date, "month"))
    day.append(getattr(date, "day"))

pd.DataFrame(day).value_counts()
seoul["year"] = year
seoul["month"] = month
seoul["day"] = day

seoul = seoul.drop(["index", "stdDay"], axis=1)
seoul.head()

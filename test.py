import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os

os.getcwd()

covid = pd.read_csv("./data/Covid19SidoInfState.csv")
covid.head()

step_data = pd.read_csv("./data/거리두기단계.csv")
step_data
step_date = []

for k in step_data.날짜:
    step_date.append(pd.to_datetime(k, format = "%Y.%m.%d"))

step_data["date"] = step_date
step_data = step_data.drop(["날짜"], axis=1)
            
    
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
stdDay = []
for n in seoul.stdDay:
    try:
        date = pd.to_datetime(n, format = "%Y년 %m월 %d일 %H시")
    except:
        date = pd.to_datetime(n, format = "%Y년 %m년 %d일 %H시")
    stdDay.append(date)
    year.append(getattr(date, "year"))
    month.append(getattr(date, "month"))
    day.append(getattr(date, "day"))

pd.DataFrame(day).value_counts()
seoul.stdDay = stdDay
seoul["year"] = year
seoul["month"] = month
seoul["day"] = day
seoul.head()

#코로나 거리두기 단계 붙이기
step = []
for k in seoul.stdDay:
    for i in range(len(step_data.date)):
        if k <= step_data.date[i]:
            step.append(step_data.단계명[i])
            break
        elif k>=pd.to_datetime("2022-04-18"):
            step.append("거리두기해제")
            break

len(step)
len(seoul)

data = seoul.copy()

data["step"] = step
data = data.drop(["index", "stdDay", "gubun"], axis=1)
data.head()

#더미처리
data["year"] = pd.Categorical(data["year"])
data["month"] = pd.Categorical(data["month"])
data["day"] = pd.Categorical(data["day"])
data["step"] = pd.Categorical(data["step"])
dummies = pd.get_dummies(data[["year", "month", "step"]])
data[dummies.columns] = dummies
data = data.drop(["step"], axis=1)
data = data.drop(["year", "month", "day"], axis=1)

data.keys()




# define the loss function under the generalized extreme value distribution
class GEVLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.FloatTensor([0.0]))
        # self.sigma = torch.FloatTensor([0.5])
        self.xi = nn.Parameter(torch.FloatTensor([0.001]))
        # self.xi = torch.FloatTensor([0.5])

    def forward(self, targets, mu):
        n = targets.numel()
        sigma = torch.exp(self.sigma)
        xi = self.xi
        y = 1 + xi * ((targets - mu) / sigma)
        neg_loglike = n * torch.log(sigma) + (1 + 1 / xi) * torch.sum(torch.log(y)) + torch.sum(y**(-1 / xi))
        return(neg_loglike)

# define the neural network architecture
## feedforward neural network with two hidden layers
class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNN, self).__init__()

        self.h1 = nn.Linear(input_dim, hidden_dim)
        # nn.init.kaiming_uniform_(self.h1.weight)
        nn.init.constant_(self.h1.bias, 0.5)
        self.ReLU = nn.ReLU()
        self.h2 = nn.Linear(hidden_dim, hidden_dim)
        # nn.init.kaiming_uniform_(self.h2.weight)
        nn.init.constant_(self.h2.bias, 0.5)
        self.out = nn.Linear(hidden_dim, output_dim)
        # nn.init.kaiming_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.5)

    def forward(self, x):
        x = self.h1(x)
        x = self.ReLU(x)
        x = self.h2(x)
        x = self.ReLU(x)
        out = self.out(x)
        return(out)
    

learning_rate = 0.0001
model = FNN(data.shape[1] - 1, 10, 1)
myloss = GEVLoss()

optimizer = torch.optim.Adam([{"params" : model.parameters()},
                               {"params" : myloss.parameters()}], lr = learning_rate)

for epoch in range(10000):
    print(epoch, "th epoch")
    y = torch.tensor(data.incDec.values, dtype = torch.float32).reshape(data.shape[0], 1)
    x = data.drop(columns = "incDec").values
    x = torch.tensor(x, dtype = torch.float32)
    optimizer.zero_grad()
    pred = model(x)

    loss = myloss(y, pred)
    loss.backward()
    optimizer.step()
    print(loss.item())


def predict(x, model, sigma, xi, return_level):
    pred_mu = model(x)
    pred_y = pred_mu - torch.exp(sigma) / xi * (1 - (-torch.log(torch.tensor(1 - return_level)))**(-xi))
    return(pred_y)

pred_01 = predict(x, model, sigma = myloss.sigma, xi = myloss.xi, return_level = 1 / 10)
pred_08 = predict(x, model, sigma = myloss.sigma, xi = myloss.xi, return_level = 8 / 10)

import matplotlib.pyplot as plt
plt.plot(data.incDec, label = "True")
plt.plot(pred_01.detach().numpy(),
         label = "prediction with return level 0.1")
plt.plot(pred_08.detach().numpy(),
         label = "prediction with return level 0.8")
plt.legend()




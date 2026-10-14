import numpy as np
import pandas as pd


date = {
    "x": [1,1,0,0],
    "x2": [0,0,0,1],
    "x3": [1,0,0,0],
    "x4": [0,0,1,1],
}
df = pd.DataFrame(date)
print(df)
weight = {
    "w1": [0.2,0.6,0.5,0.9],
    "w2": [0.8,0.4,0.7,0.3],

}
wf = pd.DataFrame(weight)
n=0.6
T=0.0
Net=0
y0=0
a=1
for i in range(4):
    d1=0
    d2=0
    for k in range(4):
        d1=d1+(df.iloc[k, i]-wf.iloc[k, 0])**2
        d2 = d2+(df.iloc[k, i] - wf.iloc[k, 1])**2

    d1=np.sqrt(d1)
    d2=np.sqrt(d2)

    #print(d1,"  ",d2)

    if d1>d2:
        for k in range(4):
            wf.iloc[k, 1]=wf.iloc[k, 1] +n*(df.iloc[k, i] - wf.iloc[k, 1])
    else:
        for k in range(4):
            wf.iloc[k, 0]=wf.iloc[k, 0] +n*(df.iloc[k, i] - wf.iloc[k, 0])

for i in range(4):
    d1 = 0
    d2 = 0
    for k in range(4):
        d1 = d1 + (df.iloc[k, i] - wf.iloc[k, 0]) ** 2
        d2 = d2 + (df.iloc[k, i] - wf.iloc[k, 1]) ** 2

    d1 = np.sqrt(d1)
    d2 = np.sqrt(d2)

    cluster = 0 if d1 < d2 else 1
    print(f"Вектор x{i + 1} принадлежит кластеру {cluster}")




print(wf)

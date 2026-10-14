import pandas as pd

date = {
    "x": [-1,1,-1,1],
    "x2": [-1,-1,1,1],
    "y": [-1,-1,-1,1]
}
df = pd.DataFrame(date)
weight= pd.DataFrame([0,0],columns=["W"])

print(df)
print(weight)
T=0
for k in range(4):
    weight.iloc[0]=weight.iloc[0]+df.iloc[k,0]*df.iloc[k,2]
    weight.iloc[1] = weight.iloc[1] + df.iloc[k, 1] * df.iloc[k, 2]
    T=T-df.iloc[k,2]
    print(weight.iloc[0])
    print(weight.iloc[1])
    print("T:",T)

import pandas as pd

date = {
    "x": [1,1,-1,-1],
    "x2": [1,-1,1,-1],
    "y": [1,-1,-1,-1]
}
df = pd.DataFrame(date)
weight= pd.DataFrame([0,0],columns=["W"])

#print(df)
#print(weight)
T=0
Net=0
y0=0
for k in range(4):
    Net=weight.iloc[0]*df.iloc[k,0]+weight.iloc[1]*df.iloc[k, 1]+(-1)*T
    if Net.item() >0:
        y0=1
    if Net.item()  <=0:
        y0=-1

    #print("NEt:", df.iloc[k, 0].item(), "*", weight.iloc[0].item(), "+", df.iloc[k, 1].item(), "*", weight.iloc[1].item(), "+", (-1), "*", T, "=", Net.item())
    print("NEt:", Net.item())
    if y0!=df.iloc[k,2]:

        #print("d:", d)
        weight.iloc[0]=weight.iloc[0]+1*(df.iloc[k, 2]-y0)*df.iloc[k,0]
        #print("weight11:", weight.iloc[0].item(),"+",1,"*","(",df.iloc[k, 1].item(),"-",y0,")","=",weight.iloc[0].item())
        print("weight11:", weight.iloc[0].item())
        weight.iloc[1] = weight.iloc[1] + 1 * (df.iloc[k, 2] - y0)*df.iloc[k,1]
        #print("weight21:", weight.iloc[1].item(), "+", 1, "*", "(", df.iloc[k, 1].item(), "-", y0, ")", "=", weight.iloc[1].item())
        print("weight21:", weight.iloc[1].item())
        #print("T:", T, "+", 1, "*", "(", df.iloc[k, 1].item(), "-", y0, ")", "*", df.iloc[k, 2].item())
        T=T+1*(df.iloc[k, 2]-y0)*(-1)
        print("T:", T)
        #print("NEt:", Net.item())
        #print("weight11:",weight.iloc[0].item())
        #print("weight21",weight.iloc[1].item())


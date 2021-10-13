import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import time
import pickle

def unix_time(dt):
  timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
  timestamp = int(time.mktime(timeArray))
  return timestamp

#extract data from database
def handle():
  target_db = "sqlite:///SideScan.db"
  print(target_db)
  engine = create_engine(target_db, echo=False)
  Session = sessionmaker(bind=engine)
  session = Session()
  ret = session.execute("select * from Side_Channel_Info;").fetchall()
  csv_header = session.execute("PRAGMA table_info(Side_Channel_Info);").fetchall()
  csv_header = [each[1] for each in csv_header]
  df = pd.DataFrame(ret,columns=csv_header)
  del(df['_id'])
  df.to_csv("raw_data.csv",index=0)

#Let the time interval be 50ms, and get the increments
def trans_mx(xv):
  r = xv[:1].values[:,1:]
  for i in range(1,61):
    start = xv.System_Time[:1].values[0]+i*50
    if len(xv[xv.System_Time==start])>0:
      r = np.concatenate([r,xv[xv.System_Time==start].values[:,1:]])
    else:
      point = xv[xv.System_Time<start].values[-1:,1:] if xv.System_Time[xv.System_Time<start].values[-1]>r[-1,0] else r[-1:,:] #the biggest one that less than
      r = np.concatenate([r,(xv[xv.System_Time>start][:1].values[:,1:]+point)/2])
  r = r[1:]-r[:-1]#increment
  return r[:,1:].reshape(1,60,17)

def get_mx():
  count = 0
  mx = [] 
  label = []
  all_event = pd.read_csv("all_event.csv")#get the ground truth
  time_col = np.array(list(all_event.timestamp))
  class_col = np.array(list(all_event["class"]))
  df = pd.read_csv("raw_data.csv")#get the raw data
  for each,each_class in zip(time_col,class_col):
    count = count+1
    print(count)
    xv = df[(df.System_Time>=each-1550)&(df.System_Time<each+1700)]#get the 3 seconds around the event
    #if there is no data or the duration is less than 3 seconds, skip the event
    if len(xv)==0 or xv.System_Time.iloc[-1]-xv.System_Time.iloc[0]<3050:
        print("Not Valid",each)
        continue
    xv = trans_mx(xv)
    if mx==[]:
      mx = xv
    else:
      mx = np.concatenate([mx,xv],axis=0)
    label.append(each_class)
  #get baseline
  pickle.dump([mx,label],open('multiple_mx.pkl','wb'),protocol=2)

#Get the increments of baseline 
def trans_mx_baseline(xv):#
    r = xv.values[:,1:]
    r = r[1:]-r[:-1]
    return r[:,1:].reshape(1,60,17)

def get_baseline():
    x=pickle.load(open("multiple_mx.pkl",'rb'))
    label = np.array(x[1])
    df = pd.read_csv("raw_data.csv")
    x0 = df[df.System_Time>1591650000000]# make sure there is not event after the time point
    base = []
    for i in range(3000):#number of baseline
        xv=x0[i*61:(i+1)*61]
        if base==[]:
           base = trans_mx_baseline(xv)#get the increment
        else:
           base = np.concatenate([base,trans_mx_baseline(xv)],axis=0)
    x = np.concatenate([x[0],base],axis=0)#concatenate the data
    label = np.concatenate([label,np.zeros(3000)])#concatenate the label
    pickle.dump([x,label],open("multiple_mx.pkl","wb"),protocol=2)

if __name__ == "__main__":
  handle()#extract csv from database
  get_mx()#get the data with label
  get_baseline()#get the baseline(need all_event.csv file)

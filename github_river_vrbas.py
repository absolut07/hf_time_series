"""
This code uses cleaned data. The purpose is predicting one variable
based on historical data and based on values of other variables.

"""
import pandas as pd 
import pickle
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
#%%
"""I chose one of the following models"""

#slm
def run_net(y0,trains):
    u=tf.matmul(y0,trains[0])+trains[1]
    u=tf.transpose(u) #[num_params x 2]
    u=tf.matmul(u,trains[2])
    u=tf.transpose(u)
    return u

#layers and activations
def run_net(y0,trains):
    h=tf.tanh(tf.matmul(y0,trains[0])+trains[1])
    h=tf.tanh(tf.matmul(h, trains[2])+trains[3])
    h=tf.tanh(tf.matmul(h, trains[4])+trains[5])
    h=tf.tanh(tf.matmul(h, trains[6])+trains[7]) #[np x nd]
    h=tf.tanh(tf.matmul(h, trains[8])+trains[9])
    h=tf.tanh(tf.matmul(h, trains[10])+trains[11])
    h=tf.transpose(h) #[num_params x num_days]
    h=tf.sigmoid(tf.matmul(h,trains[12])) #[num_days x 1]
    u=tf.transpose(h)
    return u

#net with layers and sigmoid:
def run_net(y0,trains):
    u=tf.matmul(y0,trains[0])+trains[1]
    u=tf.matmul(u, trains[2])+trains[3]
    u=tf.matmul(u, trains[4])+trains[5]
    #u=tf.matmul(u, trains[6])+trains[7] #[np x nd]
    #u=tf.matmul(u, trains[8])+trains[9]
    #u=tf.matmul(u, trains[10])+trains[11]
    #u=tf.matmul(u, trains[12])+trains[13]
    u=tf.transpose(u) #[num_params x num_days]
    u=tf.sigmoid(tf.matmul(u,trains[6])) #[num_days x 1]
    u=tf.transpose(u)
    return u

#net with layers:
def run_net(y0,trains):
    u=tf.matmul(y0,trains[0])+trains[1]
    u=tf.matmul(u, trains[2])+trains[3]
    u=tf.matmul(u, trains[4])+trains[5]
    u=tf.matmul(u, trains[6])+trains[7] #[np x nd]
    u=tf.matmul(u, trains[8])+trains[9]
    u=tf.matmul(u, trains[10])+trains[11]
    #u=tf.matmul(u, trains[12])+trains[13]
    u=tf.transpose(u) #[num_params x num_days]
    u=(tf.matmul(u,trains[12])) #[num_days x 1]
    u=tf.transpose(u)
    return u
#%%
"""now creating weights and biases and defining the loss function"""

def create_weights(s,m,nd,np,stdev):
    W1shape=[s,m]
    b1shape=[1,m]
    W2shape=[m,m]
    b2shape=[1,m]
    W3shape=[m,m]
    b3shape=[1,m]
    W4shape=[m,m]
    b4shape=[1,m]
    W5shape=[m,m]
    b5shape=[1,m]
    W6shape=[m,m]
    b6shape=[1,m]
    W7shape=[m,nd]
    b7shape=[1,nd]
    W8shape=[np,1]

    W1=tf.Variable(tf.random.truncated_normal(W1shape, stddev=stdev, dtype=tf.float32))
    b1=tf.Variable(tf.random.truncated_normal(b1shape, stddev=stdev, dtype=tf.float32))
    W2=tf.Variable(tf.random.truncated_normal(W2shape, stddev=stdev, dtype=tf.float32))
    b2=tf.Variable(tf.random.truncated_normal(b2shape, stddev=stdev, dtype=tf.float32))
    W3=tf.Variable(tf.random.truncated_normal(W3shape, stddev=stdev, dtype=tf.float32))
    b3=tf.Variable(tf.random.truncated_normal(b3shape, stddev=stdev, dtype=tf.float32))
    W4=tf.Variable(tf.random.truncated_normal(W4shape, stddev=stdev, dtype=tf.float32))
    b4=tf.Variable(tf.random.truncated_normal(b4shape, stddev=stdev, dtype=tf.float32))
    W5=tf.Variable(tf.random.truncated_normal(W5shape, stddev=stdev, dtype=tf.float32))
    b5=tf.Variable(tf.random.truncated_normal(b5shape, stddev=stdev, dtype=tf.float32))
    W6=tf.Variable(tf.random.truncated_normal(W6shape, stddev=stdev, dtype=tf.float32))
    b6=tf.Variable(tf.random.truncated_normal(b6shape, stddev=stdev, dtype=tf.float32))
    W7=tf.Variable(tf.random.truncated_normal(W7shape, stddev=stdev, dtype=tf.float32))
    b7=tf.Variable(tf.random.truncated_normal(b7shape, stddev=stdev, dtype=tf.float32))
    W8=tf.Variable(tf.random.truncated_normal(W8shape, stddev=stdev, dtype=tf.float32))
    return [W1,b1,W2,b2,W7, b7, W8]#W2,b2,W3,b3,W4,b4,W5,b5,W6,b6,


def calc_loss():
  u=run_net(y0, trainable_variables)
  error=tf.reduce_mean(tf.square(u-output_target))
  return error

#%%
#uploading the cleaned data
file_path='C:/Users...'

with open(file_path, 'rb') as f:
    [X]=pickle.load(f) 

#%%
#creating a correlation matrix
fp2='C:/Users/...'
X_df=pd.DataFrame({'precipitation': X[:,0], 'water level': X[:,3],
'water temp.': X[:,4], 'air temperature': X[:,1], 'turbidity': X[:,2]})
corr=X_df.corr()
sns.heatmap(corr, vmin=-1, vmax=1, cmap='viridis')
plt.savefig(fp2, dpi=300, bbox_inches='tight')
#%%
#standardization
x_pad=(X[:,0:1]-tf.reduce_min(X[:,0]))/(tf.reduce_max(X[:,0])-tf.reduce_min(X[:,0])) #padavine
x_tvaz=(X[:,1:2]-tf.reduce_min(X[:,1]))/(tf.reduce_max(X[:,1])-tf.reduce_min(X[:,1])) #temp_vazd
x_mutn=(X[:,2:3]-tf.reduce_min(X[:,2]))/(tf.reduce_max(X[:,2])-tf.reduce_min(X[:,2])) #vodostaj
x_vod=(X[:,3:4]-tf.reduce_min(X[:,3]))/(tf.reduce_max(X[:,3])-tf.reduce_min(X[:,3])) #temp vode
x_tvod=(X[:,4:]-tf.reduce_min(X[:,4]))/(tf.reduce_max(X[:,4])-tf.reduce_min(X[:,4]))

target=x_mutn


X=tf.concat([x_pad, x_tvaz, x_vod, x_tvod, target],1)#x_tvaz,

num_params=len(X[0,:])
redni_br_targeta=num_params-1
#%%
n=len(X)
perc=int(0.8*n)

podela=n-365*2

X_train=X[:(podela+365)]
X_test=X[podela:]


ni=0.001
batch_size=365
num_batches=int(len(X_train)/batch_size)

input_data=tf.transpose(X_train[:batch_size]) #(3,289)
input_data=tf.reshape(input_data, (1,1,num_params,batch_size))
for i in range(1,num_batches):
    input_data=tf.concat([input_data,tf.reshape(tf.transpose(X_train[i*batch_size:(i+1)*batch_size]),(1,1,num_params,batch_size))],1)
    
input_data=tf.reshape(input_data,(num_batches, num_params, batch_size))


output_data=input_data[1:]

input_data=input_data[:num_batches-1]

data_divided_to_batches= tf.data.Dataset.from_tensor_slices((input_data, output_data))#this has to be used for it to work

input_test=tf.transpose(X_test[:batch_size]) #(3,289)
input_test=tf.reshape(input_test, (1,1,num_params,batch_size))
for i in range(1,int(len(X_test)/batch_size)):
    input_test=tf.concat([input_test,tf.reshape(tf.transpose(X_test[i*batch_size:(i+1)*batch_size]),(1,1,num_params,batch_size))],1)

input_test=tf.reshape(input_test,(int(len(X_test)/batch_size), num_params, batch_size))
output_test1=input_test[1:]
input_test=input_test[:len(input_test)-1]

test_data=tf.data.Dataset.from_tensor_slices((input_test, output_test))
#%%

m=300#dimension of the hidden layer, when hidden layers are used
stdev=0.1 #for weight initialization
ni=0.001 #Adam optimizer

num_days=365


y0=input_data[0]
output_target=output_data[0][redni_br_targeta,:num_days]

trainable_variables=create_weights(batch_size, m, num_days, num_params,0.1)
u=run_net(y0, trainable_variables)
#%%
#second variant, when the linear model is used
w=tf.Variable(tf.random.truncated_normal([365,num_days], stddev=0.1, dtype=tf.float32))
b=tf.Variable(tf.random.truncated_normal([1,num_days], stddev=0.1, dtype=tf.float32))
W=tf.Variable(tf.random.truncated_normal([num_params,1], stddev=0.1, dtype=tf.float32))
trainable_variables=[w,b,W]


optimizer = tf.keras.optimizers.Adam(ni)
y0=input_data[0]
#%%
errors_slm=[]
#trains_seq=[]
optimizer = tf.keras.optimizers.Adam(0.0001)
for i in range(600):
    y_whole_train=tf.zeros([1,3])
    y_real=tf.zeros([1,3])
    for data in data_divided_to_batches:
        optimizer.minimize(calc_loss,trainable_variables)
        y0=data[0] #(3,289)
        output_target=data[1][redni_br_targeta,:num_days]
        output_target=tf.reshape(output_target,(1,num_days))
        y_real=tf.concat([y_real,output_target],1)
        y=run_net(y0,trainable_variables)
        y_whole_train=tf.concat([y_whole_train,y],1)
        loss_train=calc_loss()
        poslednja_godina=data[1]
    y_plot_train=y_whole_train[0,3:]
    y_real=y_real[0,3:]
    error=y_real-y_plot_train
    error_sq=tf.reduce_mean(tf.square(error))
    error_sqrt=tf.sqrt(error_sq)
    print('train error total', error_sqrt.numpy())
    print()

    y_whole_test=tf.zeros([1,3])
    target_tt=tf.zeros([1,3])
    for data in test_data:
        yyy=data[0] 
        p=run_net(yyy,trainable_variables)
        y_whole_test=tf.concat([y_whole_test,p],1)
        output_targettt=data[1][redni_br_targeta,:num_days]
        output_targettt=tf.reshape(output_targettt,(1,num_days))
        target_tt=tf.concat([target_tt,output_targettt],1)
        
    target_tt=target_tt[0,3:]
    y_test_plot=y_whole_test[0,3:]
    error=target_tt-y_test_plot
    error_sq=tf.reduce_mean(tf.square(error))
    error_sqrt=tf.sqrt(error_sq)
    error_slm.append(error_sqrt.numpy())
    print('test error total:', error_sqrt.numpy())
    print()

    #trains_seq.append(trainable_variables)

#%%
#when testing different models, collecting different errors
errors=[error_slm, error_6ll, error_6lls, error_6la]
#%%
"""the final model can be an average (by different trained variables) of models"""
nets=tf.zeros([1,365])
for trains in trains_seq:
    p=run_net(input_test[0], trains)
    nets=tf.concat([nets, p], axis=0)
nets=nets[1:,:]
uav=tf.reduce_mean(nets, axis=0)

#%%
"""test plot"""
#uavv=tf.reshape(uav, [365]) this can be instead of y_test_plot
fp3='C:/Users/NN...'
dic={'real values': target_tt.numpy(),'predicted values': y_test_plot.numpy()}
tt=pd.DataFrame(dic)
ax=tt.plot(colormap='viridis')
fig = ax.get_figure()
fig.savefig(fp3, dpi=300)
#%%
"""train plot"""
fp4='C:/Users/NN...'
dic={'target train':y_real.numpy(), 'model train': y_plot_train.numpy()}
tt=pd.DataFrame(dic)
ax=tt.plot(colormap='viridis')
fig = ax.get_figure()
fig.savefig(fp4, dpi=300)    


#%%
#saving trains
fp5='C:/Users/NN...'
with open(fp5, 'wb') as f:
    pickle.dump(trains_seq, f)


#%%
"""errors"""
fp6='C:/Users/NN...'

y6 = errors[0]
x6 = [k for k in range(len(y6))]

y1=errors[1]
x1=[k for k in range(len(y1))]

y2=errors[2]
x2=[k for k in range(len(y2))]

y3=errors[3]
x3=[k for k in range(len(y3))]

#y4=error_6l100m_noact
#x4=[k for k in range(len(y3))]

colors6 = cm.Blues(np.linspace(0.5, 1, len(y6)))
colors1 = cm.Purples(np.linspace(0.5, 1, len(y1)))
colors2 = cm.Greens(np.linspace(0.5, 1, len(y2)))
colors3 = cm.Reds(np.linspace(0.5, 1, len(y3)))
#colors4 = cm.Oranges(np.linspace(0.5, 1, len(y4)))

s_linear6 = [1 for n in range(len(x6))]
s_linear1 = [1 for n in range(len(x1))]
s_linear2 = [1 for n in range(len(x2))]
s_linear3 = [1 for n in range(len(x3))]
#s_linear4 = [1 for n in range(len(x4))]

fig, ax = plt.subplots()

ax.scatter(x6, y6, c=colors6, label='simple linear model', s=s_linear6)
ax.scatter(x1, y1, c=colors1, label='6 layer', s=s_linear1)
ax.scatter(x2, y2, c=colors2, label='6 layer sigmoid', s=s_linear2)
ax.scatter(x3, y3, c=colors3, label='6 layer activations', s=s_linear3)
#ax.scatter(x4, y4, c=colors4, label='6 layer no activations', s=s_linear4)
ax.set(xlabel='epochs', ylabel='test RMSE')
ax.set_ylim([0,0.14])
ax.grid()
legend = ax.legend(loc='upper right', shadow=True, fontsize='medium', markerscale=2.)

fig.savefig(fp6, dpi=300)
#plt.show()

#%%
"""sensitivity analysis;
for this I use the jacobian matrix and analyze derivatives 
"""
ll=[]
for yy in input_data:
    with tf.GradientTape() as g:
        g.watch(yy)
        u=run_net(yy, trainable_variables)
    du=g.jacobian(u,yy)
    ll.append(du)
    
#%%
""" jakobian matrix of an averaged model;
also I make an averaged model and then look at the derivatives """
ll=[]
for yy in input_data:
    with tf.GradientTape() as g:
        g.watch(yy)
        nets=tf.zeros([1,365])
        for trains in trains_seq:
            u=run_net(yy, trains)
            nets=tf.concat([nets, u], axis=0)
        nets=nets[1:,:]
        uav=tf.reshape(tf.reduce_mean(nets, axis=0), [1,365])
    du=g.jacobian(uav, yy)
    ll.append(du)
#%%

broj_inputa=10 #this refers to the year we want to observe

redni_br_dana=0 #the number of the day predicted
y0=tf.reshape(ll[broj_inputa][0][redni_br_dana][0], [1,365]) #derivative w.r.t. X[0]
y1=tf.reshape(ll[broj_inputa][0][redni_br_dana][1], [1,365]) #derivative w.r.t. X[1]
y2=tf.reshape(ll[broj_inputa][0][redni_br_dana][2], [1,365]) #etc.
y3=tf.reshape(ll[broj_inputa][0][redni_br_dana][3], [1,365]) 
y4=tf.reshape(ll[broj_inputa][0][redni_br_dana][4], [1,365]) 


for k in range(1,365):
    y0=tf.concat([y0,tf.reshape(ll[broj_inputa][0][k][0], [1,365])],axis=0)
    y1=tf.concat([y1, tf.reshape(ll[broj_inputa][0][k][1], [1,365])], axis=0)
    y2=tf.concat([y2, tf.reshape(ll[broj_inputa][0][k][2], [1,365])], axis=0)
    y3=tf.concat([y3, tf.reshape(ll[broj_inputa][0][k][3], [1,365])], axis=0)
    y4=tf.concat([y4, tf.reshape(ll[broj_inputa][0][k][4], [1,365])], axis=0)


#%%
fp7='C:/Users/NN'
tt=pd.DataFrame(y2.numpy())
tt.index.name='turbidity'
tt.columns.name='derivative w.r.t. water level'

cmap = sns.diverging_palette(240, 10, s=100, sep=40, center='light', as_cmap=True)
sns.heatmap(tt,cmap=cmap, center=0) #, vmin=-0.03, vmax=0.03
plt.savefig(fp7, dpi=300, bbox_inches='tight')
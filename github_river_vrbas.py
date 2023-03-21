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
############# choose one of the following models #########

# slm
def run_net_slm(y0, trainable_vars):
    u = tf.matmul(y0, trainable_vars[0]) + trainable_vars[1]
    u = tf.transpose(u)  # [num_params x 2]
    u = tf.matmul(u, trainable_vars[2])
    u = tf.transpose(u)
    return u


# layers and activations
def run_net_tanh(y0, trainable_vars):
    W1, W2, W3, W4, W5, W6, W7, b1, b2, b3, b4, b5, b6 = trainable_vars
    h = tf.tanh(tf.matmul(y0, W1) + b1)
    h = tf.tanh(tf.matmul(h, W2) + b2)
    h = tf.tanh(tf.matmul(h, W3) + b3)
    h = tf.tanh(tf.matmul(h, W4) + b4)  # [np x nd]
    h = tf.tanh(tf.matmul(h, W5) + b5)
    h = tf.tanh(tf.matmul(h, W6) + b6)
    h = tf.transpose(h)  # [num_params x num_days]
    h = tf.sigmoid(tf.matmul(h, W7))  # [num_days x 1]
    u = tf.transpose(h)
    return u


# net with layers and sigmoid:
def run_net(y0, trainable_vars):
    W1, W2, W3, W4, W5, W6, W7, b1, b2, b3, b4, b5, b6 = trainable_vars
    u = tf.matmul(y0, W1) + b1
    u = tf.matmul(u, W2) + b2
    u = tf.matmul(u, W3) + b3
    u = tf.matmul(u, W4) + b4  # [np x nd]
    u = tf.matmul(u, W5) + b5
    u = tf.matmul(u, W6) + b6
    u = tf.transpose(u)  # [num_params x num_days]
    u = tf.sigmoid(tf.matmul(u, W7))  # [num_days x 1]
    u = tf.transpose(u)
    return u


# net with no activations:
def run_net_noact(y0, trainable_vars):
    W1, W2, W3, W4, W5, W6, W7, b1, b2, b3, b4, b5, b6 = trainable_vars
    u = tf.matmul(y0, W1) + b1
    u = tf.matmul(u, W2) + b2
    u = tf.matmul(u, W3) + b3
    u = tf.matmul(u, W4) + b4
    u = tf.matmul(u, W5) + b5
    u = tf.matmul(u, W6) + b6
    u = tf.transpose(u)  # [num_params x num_days]
    u = tf.matmul(u, W7)  # [num_days x 1]
    u = tf.transpose(u)
    return u


def create_weights(s, m, nd, np, stdev):
    """Initialization of weights and biases"""
    shapes_weights = [
        [s, m],
        [m, m],
        [m, m],
        [m, m],
        [m, m],
        #    [m, m],
        [m, nd],
        [np, 1],
    ]
    shapes_biases = [
        [1, m],
        [1, m],
        [1, m],
        [1, m],
        [1, m],
        #    [1, m],
        [1, nd],
    ]
    trainable_vars = []
    for shape in shapes_weights:
        trainable_vars.append(
            tf.Variable(
                tf.random.truncated_normal(shape, stddev=stdev, dtype=tf.float32)
            )
        )
    for shape in shapes_biases:
        trainable_vars.append(
            tf.Variable(
                tf.random.truncated_normal(shape, stddev=stdev, dtype=tf.float32)
            )
        )
    return trainable_vars


#%%
def calc_loss():
    u = run_net(y0, trainable_vars)
    error = tf.reduce_mean(tf.square(u - output_target))
    return error

#%%
# uploading the cleaned data
file_path = "C:/Users..."

with open(file_path, "rb") as f:
    [X] = pickle.load(f)

#%%
# creating a correlation matrix
fp2 = "C:/Users/..."
X_df = pd.DataFrame(
    {
        "precipitation": X[:, 0],
        "water level": X[:, 3],
        "water temp.": X[:, 4],
        "air temperature": X[:, 1],
        "turbidity": X[:, 2],
    }
)
corr = X_df.corr()
sns.heatmap(corr, vmin=-1, vmax=1, cmap="viridis")
plt.savefig(fp2, dpi=300, bbox_inches="tight")
#%%
# standardization
x_pad = (X[:, 0:1] - tf.reduce_min(X[:, 0])) / (
    tf.reduce_max(X[:, 0]) - tf.reduce_min(X[:, 0])
)  # padavine
x_tvaz = (X[:, 1:2] - tf.reduce_min(X[:, 1])) / (
    tf.reduce_max(X[:, 1]) - tf.reduce_min(X[:, 1])
)  # temp_vazd
x_mutn = (X[:, 2:3] - tf.reduce_min(X[:, 2])) / (
    tf.reduce_max(X[:, 2]) - tf.reduce_min(X[:, 2])
)  # vodostaj
x_vod = (X[:, 3:4] - tf.reduce_min(X[:, 3])) / (
    tf.reduce_max(X[:, 3]) - tf.reduce_min(X[:, 3])
)  # temp vode
x_tvod = (X[:, 4:] - tf.reduce_min(X[:, 4])) / (
    tf.reduce_max(X[:, 4]) - tf.reduce_min(X[:, 4])
)

target = x_mutn
X = tf.concat([x_pad, x_tvaz, x_vod, x_tvod, target], 1)  # x_tvaz,

num_params = len(X[0, :])
redni_br_targeta = num_params - 1
#%%
n = len(X)
perc = int(0.8 * n)
podela = n - 365 * 2

X_train = X[: (podela + 365)]
X_test = X[podela:]

ni = 0.001
batch_size = 365
num_batches = int(len(X_train) / batch_size)

input_data = tf.transpose(X_train[:batch_size])  # (3,289)
input_data = tf.reshape(input_data, (1, 1, num_params, batch_size))
for i in range(1, num_batches):
    input_data = tf.concat(
        [
            input_data,
            tf.reshape(
                tf.transpose(X_train[i * batch_size : (i + 1) * batch_size]),
                (1, 1, num_params, batch_size),
            ),
        ],
        1,
    )

input_data = tf.reshape(input_data, (num_batches, num_params, batch_size))


output_data = input_data[1:]

input_data = input_data[: num_batches - 1]

data_divided_to_batches = tf.data.Dataset.from_tensor_slices(
    (input_data, output_data)
)  # this has to be used for it to work

input_test = tf.transpose(X_test[:batch_size])  # (3,289)
input_test = tf.reshape(input_test, (1, 1, num_params, batch_size))
for i in range(1, int(len(X_test) / batch_size)):
    input_test = tf.concat(
        [
            input_test,
            tf.reshape(
                tf.transpose(X_test[i * batch_size : (i + 1) * batch_size]),
                (1, 1, num_params, batch_size),
            ),
        ],
        1,
    )

input_test = tf.reshape(
    input_test, (int(len(X_test) / batch_size), num_params, batch_size)
)
output_test = input_test[1:]
input_test = input_test[: len(input_test) - 1]

test_data = tf.data.Dataset.from_tensor_slices((input_test, output_test))
#%%

m = 300  # dimension of the hidden layer, when hidden layers are used
stdev = 0.1  # for weight initialization
ni = 0.001  # Adam optimizer
num_days = 365

y0 = input_data[0]
output_target = output_data[0][redni_br_targeta, :num_days]

trainable_vars = create_weights(batch_size, m, num_days, num_params, 0.1)
u = run_net(y0, trainable_vars)
#%%
############ second variant, when the linear model is used ##################
w = tf.Variable(
    tf.random.truncated_normal([365, num_days], stddev=0.1, dtype=tf.float32)
)
b = tf.Variable(tf.random.truncated_normal([1, num_days], stddev=0.1, dtype=tf.float32))
W = tf.Variable(
    tf.random.truncated_normal([num_params, 1], stddev=0.1, dtype=tf.float32)
)
trainable_vars = [w, b, W]
optimizer = tf.keras.optimizers.Adam(ni)
y0 = input_data[0]
u = run_net_slm(y0, trainable_vars)
#%%
errors_sigmoid = []
trains_seq = []
optimizer = tf.keras.optimizers.Adam(0.0001)
for i in range(600):
    y_whole_train = tf.zeros([1, 3])
    y_real = tf.zeros([1, 3])
    for data in data_divided_to_batches:
        optimizer.minimize(calc_loss, trainable_vars)
        y0 = data[0]  # (3,289)
        output_target = data[1][redni_br_targeta, :num_days]
        output_target = tf.reshape(output_target, (1, num_days))
        y_real = tf.concat([y_real, output_target], 1)
        y = run_net(y0, trainable_vars)
        y_whole_train = tf.concat([y_whole_train, y], 1)
        loss_train = calc_loss()
        poslednja_godina = data[1]
    y_plot_train = y_whole_train[0, 3:]
    y_real = y_real[0, 3:]
    error = y_real - y_plot_train
    error_sq = tf.reduce_mean(tf.square(error))
    error_sqrt = tf.sqrt(error_sq)
    print("train error total", error_sqrt.numpy())
    print()

    y_whole_test = tf.zeros([1, 3])
    target_tt = tf.zeros([1, 3])
    for data in test_data:
        yyy = data[0]
        p = run_net(yyy, trainable_vars)
        y_whole_test = tf.concat([y_whole_test, p], 1)
        output_targettt = data[1][redni_br_targeta, :num_days]
        output_targettt = tf.reshape(output_targettt, (1, num_days))
        target_tt = tf.concat([target_tt, output_targettt], 1)

    target_tt = target_tt[0, 3:]
    y_test_plot = y_whole_test[0, 3:]
    error = target_tt - y_test_plot
    error_sq = tf.reduce_mean(tf.square(error))
    error_sqrt = tf.sqrt(error_sq)
    errors_sigmoid.append(error_sqrt.numpy())
    print("test error total:", error_sqrt.numpy())
    print()

    trains_seq.append(trainable_vars)

#%%
############# the final model can be an average (by different trained variables) of models ############
nets = tf.zeros([1, 365])
for trains in trains_seq:
    p = run_net(input_test[0], trains)
    nets = tf.concat([nets, p], axis=0)
nets = nets[1:, :]
uav = tf.reduce_mean(nets, axis=0)

#%%
################ test plot ###############
# uavv=tf.reshape(uav, [365]) this can be instead of y_test_plot
fp3 = "C:/Users/NN..."
dic = {"real values": target_tt.numpy(), "predicted values": y_test_plot.numpy()}
tt = pd.DataFrame(dic)
ax = tt.plot(colormap="viridis")
fig = ax.get_figure()
fig.savefig(fp3, dpi=300)
#%%
############### train plot ######################
fp4 = "C:/Users/NN..."
dic = {"target train": y_real.numpy(), "model train": y_plot_train.numpy()}
tt = pd.DataFrame(dic)
ax = tt.plot(colormap="viridis")
fig = ax.get_figure()
fig.savefig(fp4, dpi=300)

#%%
############## errors ###############
fp6 = "C:/Users/NN..."  # saved different kinds of errors
errors = pickle.load(open(fp6, "rb"))
y6, y1, y2, y3 = errors[0]

x6 = [k for k in range(len(y6))]
x1 = [k for k in range(len(y1))]
x2 = [k for k in range(len(y2))]
x3 = [k for k in range(len(y3))]


colors6 = cm.Blues(np.linspace(0.5, 1, len(y6)))
colors1 = cm.Purples(np.linspace(0.5, 1, len(y1)))
colors2 = cm.Greens(np.linspace(0.5, 1, len(y2)))
colors3 = cm.Reds(np.linspace(0.5, 1, len(y3)))

s_linear6 = [1 for n in range(len(x6))]
s_linear1 = [1 for n in range(len(x1))]
s_linear2 = [1 for n in range(len(x2))]
s_linear3 = [1 for n in range(len(x3))]

fig, ax = plt.subplots()

ax.scatter(x6, y6, c=colors6, label="simple linear model", s=s_linear6)
ax.scatter(x1, y1, c=colors1, label="6 layer", s=s_linear1)
ax.scatter(x2, y2, c=colors2, label="6 layer sigmoid", s=s_linear2)
ax.scatter(x3, y3, c=colors3, label="6 layer activations", s=s_linear3)
ax.set(xlabel="epochs", ylabel="test RMSE")
ax.set_ylim([0, 0.14])
ax.grid()
legend = ax.legend(loc="upper right", shadow=True, fontsize="medium", markerscale=2.0)

fig.savefig(fp6, dpi=300)
# plt.show()

#%%
############# sensitivity analysis #################
# for this I use the jacobian matrix and analyze derivatives
ll = []
for yy in input_data:
    with tf.GradientTape() as g:
        g.watch(yy)
        u = run_net(yy, trainable_vars)
    du = g.jacobian(u, yy)
    ll.append(du)

#%%
############## jacobian matrix of an averaged model ###################
# also make an averaged model and then look at the derivatives
ll = []
for yy in input_data:
    with tf.GradientTape() as g:
        g.watch(yy)
        nets = tf.zeros([1, 365])
        for trains in trains_seq:
            u = run_net(yy, trains)
            nets = tf.concat([nets, u], axis=0)
        nets = nets[1:, :]
        uav = tf.reshape(tf.reduce_mean(nets, axis=0), [1, 365])
    du = g.jacobian(uav, yy)
    ll.append(du)
#%%

broj_inputa = 10  # this refers to the year we want to observe

redni_br_dana = 0  # the number of the day predicted
y0 = tf.reshape(
    ll[broj_inputa][0][redni_br_dana][0], [1, 365]
)  # derivative w.r.t. X[0]
y1 = tf.reshape(
    ll[broj_inputa][0][redni_br_dana][1], [1, 365]
)  # derivative w.r.t. X[1]
y2 = tf.reshape(ll[broj_inputa][0][redni_br_dana][2], [1, 365])  # etc.
y3 = tf.reshape(ll[broj_inputa][0][redni_br_dana][3], [1, 365])
y4 = tf.reshape(ll[broj_inputa][0][redni_br_dana][4], [1, 365])


for k in range(1, 365):
    y0 = tf.concat([y0, tf.reshape(ll[broj_inputa][0][k][0], [1, 365])], axis=0)
    y1 = tf.concat([y1, tf.reshape(ll[broj_inputa][0][k][1], [1, 365])], axis=0)
    y2 = tf.concat([y2, tf.reshape(ll[broj_inputa][0][k][2], [1, 365])], axis=0)
    y3 = tf.concat([y3, tf.reshape(ll[broj_inputa][0][k][3], [1, 365])], axis=0)
    y4 = tf.concat([y4, tf.reshape(ll[broj_inputa][0][k][4], [1, 365])], axis=0)


#%%
fp7 = "C:/Users/NN"
tt = pd.DataFrame(y2.numpy())
tt.index.name = "turbidity"
tt.columns.name = "derivative w.r.t. water level"

cmap = sns.diverging_palette(240, 10, s=100, sep=40, center="light", as_cmap=True)
sns.heatmap(tt, cmap=cmap, center=0)  # , vmin=-0.03, vmax=0.03
plt.savefig(fp7, dpi=300, bbox_inches="tight")

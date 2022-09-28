import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time


def sigmoid(x):
    return  1 /( 1 + (math.e)**(-1 * x))

def sigmoid_deriviate(x):
    a = sigmoid(x)
    a = np.reshape(a,(-1,1))
    b = 1 - sigmoid(x)
    b = np.reshape(b,(-1,1))
    b = np.transpose(b)
    return np.diag(np.diag(np.matmul(a,b)))

def F(x):
    a = np.multiply(x,(1-x))
    return a



data = pd.read_excel('data.xlsx',header=None)
data = np.array(data)
data = data[:3740,:]


minn = np.min(data[:,0])
maxx = np.max(data[:,0])

for i in range(np.shape(data)[0]):
    for j in range(np.shape(data)[1]):
        data[i,j] = (data[i,j] - minn) / (maxx - minn)




train_rate=0.7;
eta_e1 = 0.05;
eta_e2 = 0.03;
eta_p = 0.07;
epochs_ae = 30;
max_epoch_p = 30;



split_line_number = int(np.shape(data)[0] * train_rate)
x_train = data[:split_line_number,:5]
x_test = data[split_line_number:,:5]
y_train = data[:split_line_number,5]
y_test = data[split_line_number:,5]
input_dimension = np.shape(x_train)[1]

n0_neurons = input_dimension;
n1_neurons = 3;
n2_neurons = 2;

w_i1 = np.random.uniform(low=-1,high=1,size=(n1_neurons,n0_neurons))
w_o1 = np.random.uniform(low=-1,high=1,size=(n1_neurons,n0_neurons))
w_f1 = np.random.uniform(low=-1,high=1,size=(n1_neurons,n0_neurons))
w_c1 = np.random.uniform(low=-1,high=1,size=(n1_neurons,n0_neurons))
C1 = 1
h1 = 0
f1 = 0
c_1 = 0
i1 = 0

w_i2 = np.random.uniform(low=-1,high=1,size=(n2_neurons,n1_neurons))
w_o2 = np.random.uniform(low=-1,high=1,size=(n2_neurons,n1_neurons))
w_f2 = np.random.uniform(low=-1,high=1,size=(n2_neurons,n1_neurons))
w_c2 = np.random.uniform(low=-1,high=1,size=(n2_neurons,n1_neurons))
C2 = 1
h2 = 0
f2 = 0
c_2 = 0
i2 = 0


w_d1 = np.random.uniform(low=-1,high=1,size=(n0_neurons,n1_neurons))
w_d2 = np.random.uniform(low=-1,high=1,size=(n1_neurons,n2_neurons))

#MLP
l1_neurons=1;
w1 = np.random.uniform(low=-1,high=1,size=(l1_neurons,n2_neurons))





MSE_train = []
MSE_test = []
MSE_train_AE1 = []


#Encoder1
for i in range(epochs_ae):
    for j in range(np.shape(x_train)[0]):
        # Feed-Forward

        #Encoder 1
        
        x_previous = np.transpose(np.reshape(x_train[j-1],(-1,1)))
        f1_pre = f1
        c_1_pre = c_1
        i1_pre = i1
        C1_pre = C1

        net_i1 = np.reshape(np.matmul(w_i1, x_train[j]),(-1,1)) + h1
        i1 = sigmoid(net_i1)
        
        net_o1 = np.reshape(np.matmul(w_o1, x_train[j]),(-1,1)) + h1
        o1 = sigmoid(net_o1)
        
        net_f1 = np.reshape(np.matmul(w_f1, x_train[j]),(-1,1)) + h1
        f1 = sigmoid(net_f1)

        net_c1 = np.reshape(np.matmul(w_c1, x_train[j]),(-1,1)) + h1
        c_1 = np.tanh(net_c1)
        
        C1 = np.multiply(f1,C1) + np.multiply(i1,c_1)
        h1 = np.multiply(o1,np.tanh(C1)) 
        
        #Decoder1
        net_d1 = np.matmul(w_d1, h1)
        x_hat = sigmoid(net_d1)
        x_hat = np.reshape(x_hat,(-1,1))


        # Error
        err = np.reshape(x_train[j],(-1,1)) - x_hat
        
        # Back propagation
        
        # Train w_d1
        w_d1 = np.subtract(w_d1 , (eta_e1 * err * -1 * 1 * np.transpose(h1) ))
        
        # Train w_i1
        temp = np.matmul(np.transpose(err),w_d1)
        temp1 = o1 * (1 - (np.tanh(C1)**2))
        temp = np.matmul(temp,temp1)
        temp1 = f1 * c_1_pre * F(i1_pre)
        temp1 = np.matmul(temp1,x_previous)
        temp2 = c_1 * F(i1)
        temp2 = np.matmul(temp2,np.transpose(np.reshape(x_train[j],(-1,1))))
        temp1 = temp1 + temp2
        temp = temp * temp1
        w_i1 = np.subtract(w_i1 , (eta_e1 * -1 * 1 * temp ))
        
        # Train w_c1
        temp = np.matmul(np.transpose(err),w_d1)
        temp1 = o1 * (1 - (np.tanh(C1)**2))
        temp = np.matmul(temp,temp1)
        temp1 = f1 * i1_pre * F(C1_pre)
        temp1 = np.matmul(temp1,x_previous)
        temp2 = i1 * F(C1)
        temp2 = np.matmul(temp2,np.transpose(np.reshape(x_train[j],(-1,1))))
        temp1 = temp1 + temp2
        temp = temp * temp1
        w_c1 = np.subtract(w_c1 , (eta_e1 * -1 * 1 * temp ))
        
        # Train w_f1
        temp = np.matmul(np.transpose(err),w_d1)
        temp1 = o1 * (1 - (np.tanh(C1)**2))
        temp = np.matmul(temp,temp1)
        temp1 = C1_pre * F(f1)
        temp1 = np.matmul(temp1,np.transpose(np.reshape(x_train[j],(-1,1))))
        temp = temp * temp1
        w_f1 = np.subtract(w_f1 , (eta_e1 * -1 * 1 * temp ))

        # Train w_o1
        temp = np.matmul(np.transpose(err),w_d1)
        temp1 = np.tanh(C1)
        temp = np.matmul(temp,temp1)
        temp = temp * F(o1)
        temp = np.matmul(temp,np.transpose(np.reshape(x_train[j],(-1,1))))
        w_o1 = np.subtract(w_o1 , (eta_e1 * -1 * 1 * temp ))
        
        


#Encoder2
for i in range(epochs_ae):
    for j in range(np.shape(x_train)[0]):
        # Feed-Forward

        #Encoder 1
        
        net_i1 = np.reshape(np.matmul(w_i1, x_train[j]),(-1,1)) + h1
        i1 = sigmoid(net_i1)
        
        net_o1 = np.reshape(np.matmul(w_o1, x_train[j]),(-1,1)) + h1
        o1 = sigmoid(net_o1)
        
        net_f1 = np.reshape(np.matmul(w_f1, x_train[j]),(-1,1)) + h1
        f1 = sigmoid(net_f1)

        net_c1 = np.reshape(np.matmul(w_c1, x_train[j]),(-1,1)) + h1
        c_1 = np.tanh(net_c1)
        
        C1 = np.multiply(f1,C1) + np.multiply(i1,c_1)
        h1_previous = np.transpose(h1)
        h1 = np.multiply(o1,np.tanh(C1)) 


        #Encoder 2

        f2_pre = f2
        c_2_pre = c_2
        i2_pre = i2
        C2_pre = C2

        net_i2 = np.matmul(w_i2, h1) + h2
        i2 = sigmoid(net_i2)
        
        net_o2 = np.matmul(w_o2, h1) + h2
        o2 = sigmoid(net_o2)
        
        net_f2 = np.matmul(w_f2, h1) + h2
        f2 = sigmoid(net_f2)

        net_c2 = np.matmul(w_c2, h1) + h2
        c_2 = np.tanh(net_c2)
        
        C2 = np.multiply(f2,C2) + np.multiply(i2,c_2)
        h2 = np.multiply(o2,np.tanh(C2)) 
        
        #Decoder2
 
        net_d2 = np.matmul(w_d2, h2)
        h1_hat = sigmoid(net_d2)
        h1_hat = np.reshape(h1_hat,(-1,1))

        # Error
        err = h1 - h1_hat
        
        # Back propagation
        
        # Train w_d2
        w_d2 = np.subtract(w_d2 , (eta_e2 * err * -1 * 1 * np.transpose(h2)))

        # Train w_i2
        temp = np.matmul(np.transpose(err),w_d2)
        temp1 = o2 * (1 - (np.tanh(C2)**2))
        temp = np.matmul(temp,temp1)
        temp1 = f2 * c_2_pre * F(i2_pre)
        temp1 = np.matmul(temp1,h1_previous)
        temp2 = c_2 * F(i2)
        temp2 = np.matmul(temp2,np.transpose(h1))
        temp1 = temp1 + temp2
        temp = temp * temp1
        w_i2 = np.subtract(w_i2 , (eta_e2 * -1 * 1 * temp ))
        
        # Train w_c2
        temp = np.matmul(np.transpose(err),w_d2)
        temp1 = o2 * (1 - (np.tanh(C2)**2))
        temp = np.matmul(temp,temp1)
        temp1 = f2 * i2_pre * F(C2_pre)
        temp1 = np.matmul(temp1,h1_previous)
        temp2 = i2 * F(C2)
        temp2 = np.matmul(temp2,np.transpose(h1))
        temp1 = temp1 + temp2
        temp = temp * temp1
        w_c2 = np.subtract(w_c2 , (eta_e2 * -1 * 1 * temp ))
        
        # Train w_f2
        temp = np.matmul(np.transpose(err),w_d2)
        temp1 = o2 * (1 - (np.tanh(C2)**2))
        temp = np.matmul(temp,temp1)
        temp1 = C2_pre * F(f2)
        temp1 = np.matmul(temp1,np.transpose(h1))
        temp = temp * temp1
        w_f2 = np.subtract(w_f2 , (eta_e2 * -1 * 1 * temp ))

        # Train w_o2
        temp = np.matmul(np.transpose(err),w_d2)
        temp1 = np.tanh(C2)
        temp = np.matmul(temp,temp1)
        temp = temp * F(o2)
        temp = np.matmul(temp,np.transpose(h1))
        w_o2 = np.subtract(w_o2 , (eta_e2 * -1 * 1 * temp ))
        
        
        


# Perceptron 1 Layer
for i in range(max_epoch_p):

    sqr_err_epoch_train = []
    sqr_err_epoch_test = []

    output_train = []
    output_test = []

    for j in range(np.shape(x_train)[0]):
        # Feed-Forward

        #Encoder 1
        
        net_i1 = np.reshape(np.matmul(w_i1, x_train[j]),(-1,1)) + h1
        i1 = sigmoid(net_i1)
        
        net_o1 = np.reshape(np.matmul(w_o1, x_train[j]),(-1,1)) + h1
        o1 = sigmoid(net_o1)
        
        net_f1 = np.reshape(np.matmul(w_f1, x_train[j]),(-1,1)) + h1
        f1 = sigmoid(net_f1)

        net_c1 = np.reshape(np.matmul(w_c1, x_train[j]),(-1,1)) + h1
        c_1 = np.tanh(net_c1)
        
        C1 = np.multiply(f1,C1) + np.multiply(i1,c_1)
        h1_previous = np.transpose(h1)
        h1 = np.multiply(o1,np.tanh(C1)) 


        #Encoder 2

        net_i2 = np.matmul(w_i2, h1) + h2
        i2 = sigmoid(net_i2)
        
        net_o2 = np.matmul(w_o2, h1) + h2
        o2 = sigmoid(net_o2)
        
        net_f2 = np.matmul(w_f2, h1) + h2
        f2 = sigmoid(net_f2)

        net_c2 = np.matmul(w_c2, h1) + h2
        c_2 = np.tanh(net_c2)
        
        C2 = np.multiply(f2,C2) + np.multiply(i2,c_2)
        h2 = np.multiply(o2,np.tanh(C2)) 

        #MLP 1
        net1 = np.matmul(w1,h2)
        o1 = net1

        
        output_train.append(o1[0])

        # Error
        err = y_train[j] - o1[0]
        sqr_err_epoch_train.append(err**2)


        # Back propagation
        f_driviate = sigmoid_deriviate(net1)



        w1 = np.subtract(w1 , np.transpose((eta_p * err * -1 * 1 * h2)))


    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train))/np.shape(x_train)[0])
    MSE_train.append(mse_epoch_train)

    for j in range(np.shape(x_test)[0]):
        # Feed-Forward

        #Encoder 1
        
        net_i1 = np.reshape(np.matmul(w_i1, x_train[j]),(-1,1)) + h1
        i1 = sigmoid(net_i1)
        
        net_o1 = np.reshape(np.matmul(w_o1, x_train[j]),(-1,1)) + h1
        o1 = sigmoid(net_o1)
        
        net_f1 = np.reshape(np.matmul(w_f1, x_train[j]),(-1,1)) + h1
        f1 = sigmoid(net_f1)

        net_c1 = np.reshape(np.matmul(w_c1, x_train[j]),(-1,1)) + h1
        c_1 = np.tanh(net_c1)
        
        C1 = np.multiply(f1,C1) + np.multiply(i1,c_1)
        h1_previous = np.transpose(h1)
        h1 = np.multiply(o1,np.tanh(C1)) 


        #Encoder 2

        net_i2 = np.matmul(w_i2, h1) + h2
        i2 = sigmoid(net_i2)
        
        net_o2 = np.matmul(w_o2, h1) + h2
        o2 = sigmoid(net_o2)
        
        net_f2 = np.matmul(w_f2, h1) + h2
        f2 = sigmoid(net_f2)

        net_c2 = np.matmul(w_c2, h1) + h2
        c_2 = np.tanh(net_c2)
        
        C2 = np.multiply(f2,C2) + np.multiply(i2,c_2)
        h2 = np.multiply(o2,np.tanh(C2)) 

        #MLP 1
        net1 = np.matmul(w1,h2)
        o1 = net1
        output_test.append(o1[0])

        # Error
        err = y_test[j] - o1[0]
        sqr_err_epoch_test.append(err ** 2)

    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test))/np.shape(x_test)[0])
    MSE_test.append(mse_epoch_test)

    # Ploy fits

        # Train
    m_train , b_train = np.polyfit(y_train,output_train,1)

        # Test

    m_test , b_test = np.polyfit(y_test, output_test, 1)

    print(m_train,b_train,m_test,b_test)

    # Plots
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(MSE_train,'b')
    axs[0, 0].set_title('MSE Train')
    axs[0, 1].plot(MSE_test,'r')
    axs[0, 1].set_title('Mse Test')

    axs[1, 0].plot(y_train, 'b')
    axs[1, 0].plot(output_train,'r')
    axs[1, 0].set_title('Output Train')
    axs[1, 1].plot(y_test, 'b')
    axs[1, 1].plot(output_test,'r')
    axs[1, 1].set_title('Output Test')

    axs[2, 0].plot(y_train, output_train, 'b*')
    axs[2, 0].plot(y_train, m_train*y_train+b_train,'r')
    axs[2, 0].set_title('Regression Train')
    axs[2, 1].plot(y_test, output_test, 'b*')
    axs[2, 1].plot(y_test,m_test*y_test+b_test,'r')
    axs[2, 1].set_title('Regression Test')
    if i == (max_epoch_p - 1):
        plt.savefig('Results.jpg')
    plt.show()
    time.sleep(1)
    plt.close(fig)




    

        
        
        
        
D = 2**18
w = [0.] * D
n = [0.] * D # sum of previous gradients
alpha = 0.1 #learning rate


#aquisition of new obervation and parsing using hashing trick
def get_x(row):
    x=[0]
    for key, v in row.items():
        index = abs(hash(key+'_'+str(v)))%D
        x.append(index)
    return x

# estimation of probability to belong to a label using weight vector w
def predict(x,w):
    wTx = 0.
    for i in x:
        wTx += w[i] * 1
        #estimated probability for x
        pred = 1. / (1. + exp(-wTx))
        return pred



#update weights given prediction versus reality
def update(alpha, w, n, x, p, y):
    for i in x:
        n[i] += abs(p - y)
        w[i] -= (p-y)*1.*alpha/sqrt(n[i])

import numpy as np

# data I/O
data = open('inputeasy.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("chars: ", chars)
#one-hot encoding
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

iteration=50000
MAX_HIDDEN_SIZE=100
hidden_size = 40
seq_length = 40
learning_rate = 1e-2

# model parameters
U = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
W = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
V = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  x, h, yprime = {}, {}, {}
  h[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in range(len(inputs)):
    x[t] = np.zeros((vocab_size,1)) 
    x[t][inputs[t]] = 1 # encode-1ofk representation    
    h[t] = np.tanh(np.dot(U, x[t]) + np.dot(W, h[t-1]) + bh)    
    temp=np.dot(V, h[t]) + by
    yprime[t] = np.exp(temp) / np.sum(np.exp(temp))
    loss += -np.log(yprime[t][targets[t],0]) # softmax (cross-entropy loss) for 1-of-k representaiton

  # backprop
  dU, dW, dV = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(h[0])

  for t in reversed(range(len(inputs))):
    dy = np.copy(yprime[t])
    dy[targets[t]] -= 1 # backprop into y. http://cs231n.github.io/neural-networks-case-study/#grad
    dV += np.dot(dy, h[t].T)
    dby += dy
    dh = np.dot(V.T, dy) + dhnext # backprop into h
    dhraw = (1 - h[t] * h[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dU += np.dot(dhraw, x[t].T)
    dW += np.dot(dhraw, h[t-1].T)
    dhnext = np.dot(W.T, dhraw)
  for dparam in [dU, dW, dV, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dU, dW, dV, dbh, dby, h[len(inputs)-1]

n, p = 0, 0
mU, mW, mV = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

for n in range(iteration):
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  loss, dU, dW, dV, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001  

  if n % 10 == 0: 
    print(len(W[1]))
    # print (n,smooth_loss)

  param=[U, W, V, bh, by]
  dparam=[dU, dW, dV, dbh, dby]
  mem=[mU, mW, mV, mbh, mby]
  for i in range(len(param)):    
    mem[i] += dparam[i] * dparam[i]
    param[i] += -learning_rate * dparam[i] / np.sqrt(mem[i] + 1e-8) # adagrad update

  normz = .5 * (np.sum(np.abs(V), axis = 0) + 4.0 * np.sum(np.abs(W), axis = 0) / hidden_size)
  sel=abs(normz)>0.05
  
  if sum(sel)<hidden_size-1:
    deletable=np.where(sel==False)[0]
    np.random.shuffle(deletable)
    for xx in range(1):
      sel[deletable[xx]]=True
    deletable=deletable[1:]
    for x in deletable:
      if np.random.rand()>0.05:        
        sel[x]=True

    hidden_size = sum(sel)
    W = W[sel,:][:, sel]      
    U = U[sel, :]
    normz = normz[sel]
    V = V[:, sel]
    bh = bh[sel]
    hprev = hprev[sel]    
    mU = mU[sel, :]
    mW = mW[sel,:][:, sel]
    mV = mV[:, sel]
    mbh = mbh[sel]    

  if hidden_size < MAX_HIDDEN_SIZE -1:
      if ( (sum((abs(normz) >0.05)) > hidden_size - 1) & (np.random.rand() < 0.01))  \
        | (np.random.rand() < 1e-4):

          W = np.append(W, np.random.randn(1, hidden_size)*0.01, axis=0)
          U = np.append(U, np.random.randn(1, vocab_size)*0.01, axis=0)


          newV = np.random.randn(vocab_size,1)
          newV = .5 * 0.05 * newV / (1e-8 + np.sum(abs(newV)))
          V = np.append(V, newV, axis=1)
          
          newW = np.random.randn(hidden_size+1, 1)
          newW = .5 * hidden_size * 0.05 * newW / (1e-8 + 4.0 * np.sum(abs(newW)))

          W = np.append(W, newW, axis=1)

          bh = np.append(bh, np.zeros((1,1)), axis=0)
          hprev = np.append(hprev, np.zeros((1,1)), axis=0)

          mW = np.append(mW, .01 * np.ones((1, hidden_size)), axis=0)
          mW = np.append(mW, .01 * np.ones((hidden_size+1, 1)), axis=1)
          mU = np.append(mU, .01 * np.ones((1, vocab_size)), axis=0)
          mV = np.append(mV, .01 * np.ones((vocab_size,1)), axis=1)
          mbh = np.append(mbh, .01 * np.ones((1,1)), axis=0)

          hidden_size += 1
  p += seq_length 














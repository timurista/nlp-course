## Recurrent Nueral Networks
like short term memory, they learn a little bit and remember it going forward.

## Vanishing Gradient Problem
All nuerons as far back you have to propogate through network.

W_rec is weight recurring. To get from layer Xt-3 to Xt-2, you are multiplying the output by weight to get next layer.

When you multiple by something small, your value decreases very quickly. Weights are assigned at random value close to zero, then they get trained up. More you mutliple lower value gets.

Lower gradient is, as it travels the gradient gets smaller, so it becomes slower. And because gradient is so much smaller, it takes longer to train half the network.

Other problem, the training has been happening based on inputs coming from untrained layers.

domino effect, whole network is not being trained properly

## rule of thumb
Small weights you get vanishing gradient problem. If you throw in larger gradient, it explodes then.

## Solutions
1. Exploding
- truncated backpropogation
- penalties
- gradient clipping
  ( it just stays as it propogates)

2. Vanishing
- weight initalizations
- echo state networks
- Long short term memory networks (LSTMs)

## LSTM
long short term memory

### History
Vanishing gradient problem, as we propogate error it goes through layers connected to themselves by recurrent weights, it causes weights on far left to be updated much slower. Whole training of network fails

### LSTM arch

Vanishing gradient problem. It goes through temporal loop, layers connected to themselves. Connected by double recurrent. So weights on far left updated less than weights of the right.

We could penalize the wrec on the right etc. But what is easiest solution?

LSTM, we make the W_rec = 1. We just make the weighted to be 1. 

`colah.github.io`

LSTM in literature.

vector transfer. We keep the same weights.

`tanh` layer operation. -1/1

sigmoid 0-1 in layers

We have a memory pipeline as either fully augment output or just to transfer over. 

Shi Yan 2016, understanding lstm and it's diagrams.
`https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714`

## LSTM Practical Intuition
How do they work inside practical applications?

`tanh` function review near ned. From Andrei blog. RNN_Effectiveness.
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

It idenfities how deeply inside a hidden state you are, in the memory cell it assigns them to keep track of certain things when it is important.

`karpath.github.io` blog for rnn effectiveness

## LSTM variations
variation #1 you can add defaults, allows decisions about valves to be made.

connecting forget and memory valve together. You combine.
Whenever you close memory off, you HAVE to put something in. If you keep memory then you put thing in (-1)

hiddent pipleine, you have 1 for hidden state and memory state. One pipeline that takes care of everything

LSTM a search space odyssey and comparison.

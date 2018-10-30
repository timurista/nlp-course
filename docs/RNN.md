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

### LSTM walkthrough
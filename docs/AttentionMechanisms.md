## Attention Mechanisms
We include nuerons in the form of

[h0] => [h1] => ... => [hn] => [y^]

In ML it uses weights to pay attention to the word that comes after and make s a decision based on weights on which word to use. 

What you can see is that attention mechanisms are the use of weights and backpropogation to construct the context vector which determines the importance of each word in the encoder as a vector. The importance of each word is determined by its weight.

## Global attention vs local attention
article: https://www.google.com/search?q=effective+approaches+to+attention-based+neural+machine+translation&oq=effective+approaches+to+attent&aqs=chrome.0.0j69i57j0l4.4114j1j7&sourceid=chrome&ie=UTF-8

The key takeaway here is that global approach which tends to focus on al source words in the encoded stream and a local pproach that only looks at a subset of words at a time.


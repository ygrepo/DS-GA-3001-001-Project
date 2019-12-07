# Project for DS-GA-3001-001
Description:
  In this work, we seek to replicate and improve the results reached by two neural networks: **ES-RNN**  
   and **N-BEATS** on the M4 dataset competition.
  We also run different experiments to compare the performances of these two deep learning techniques 
  to a more classical statistical approach like ARIMA or Gaussian Process.
  We demonstrate that although Gaussian processes could be powerful for sampling tasks and simpler to configure, 
  these neural networks outperform it for forecasting. Neural networks could have an overhead 
  in term of the number of hyper-parameters to tune, but when using batching they scale up very easily and
   generalize well to a large number of time series (100K for M4). We are thus, pretty confident that, 
   the two neural networks could forecast with the appropriate setting of hyper-parameters, 
   other univariate time series beyond the M4 dataset.  
# Personalization of Hearing Aid Compression by Human-in-the-Loop Deep Reinforcement Learning


The idea of this work is fine-tuning and personalizing compression ratios of hearing aid dynamic range compression.
In order to decrease the number of user feedbacks and thus enable a practical implementation of the personalized compression, first a reward function is considered to model hearing preferences of a user in an asynchronous manner. This is achieved by carrying out A/B comparison between instances of two different compressed audios. Then, an agent is trained to maximize reward. Following figure shows a block diagram of this approach:

![Screen Shot 2021-08-22 at 7 42 07 PM](https://user-images.githubusercontent.com/49213632/130375788-4ca28908-eb58-4540-9e3f-944a7ec6f8b6.png)

Figure 1:  Block diagram of Human-in-the-Loop Deep Reinforcement Learning.

Below shows how data is passes through different blocks of the personalization framework both in training and testing modes. Please refer to [1] for more details.

![Screen Shot 2021-08-22 at 7 43 08 PM](https://user-images.githubusercontent.com/49213632/130375844-4386688d-cdfe-4776-a002-6567fa6fefd2.png)
![Screen Shot 2021-08-22 at 7 42 59 PM](https://user-images.githubusercontent.com/49213632/130375845-203bf366-7bf9-4313-bd3c-3d7cebb88cf9.png)
Figure 2 Developed personalized compression DRL framework for (a) training mode and (b) operation mode.


[1] N.Alamdari, E.Lobarinas, N.Kehtarnavaz, “Personalization of Hearing Aid Compression by Human-in-the-Loop Deep Reinforcement Learning”, IEEE Access, vol. 8, pp. 203503-203515, 2020. 

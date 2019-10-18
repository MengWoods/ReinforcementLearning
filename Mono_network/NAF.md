# Continuous Deep Q-Learning with Model-based Acceleration  (NAF)

> Shixiang Gu, University of Cambridge , 2016



• Abstract
	○ The sample complexity of model free algorithm limit their applicability to physical system.
	○ We derive a continuous variant of the Q-learning algorithm we call NAF. 
	○ Combining NAF with models to accelerate learning.
• Related work
	○ Model free algorithms tend to be more generally applicable but substantially slower.
• Background
	○ Model-free RL: Policy gradient methods provide a simple, direct approach to RL, which can succeed on high-dimensinal problems, but potentially requires a large number of samples.
	○ Off-policy algorithms that use value or Q-function approximation can in principle achieve better data efficiency. However it requires optimizing two function approximators on different objectives.
	○ Advantage = Q - V
	○ If we know the dynamic p(x1|x0,u), we can use model-based RL and optimal control.
• NAF
	○ The idea behind NAF is to represent the Q function in Q-learning in such a way that its maximum, argmaxQ.
	○ P is a state-dependent, positive-definite square matrix
	○ ![alg.NAF](https://github.com/MorganWoods/ReinforcementLearning/blob/master/Mono_network/NAF.png)



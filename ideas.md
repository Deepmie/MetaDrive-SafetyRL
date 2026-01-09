Reinforcement Learning作为High-level decision-maker输出actions: $z_{ref} = (v_{ref}, \theta_{ref})$，然后MPC作为Bottom-level controller做一个类似轨迹跟随的最优控制任务：
$$
\begin{aligned}
J =& \underset{u}{\min}\sum_{k=0}^{n_p}\Vert z_{ref} - z_k\Vert_Q+\Vert u_k\Vert_R \\
&\left\{\begin{array}{lll}
x_{k+1} = f(x_k, u_k) \\
z_k = Cx_k
\end{array}\right.
\end{aligned}
$$
其中$C=[0,0,1,1]$，$x = (p_x, p_y, v, \theta)$分别代表vehicle的横坐标、纵坐标、速度以及角度，因此$z$相当于取$x$的后两项。现在，我要给出一个可能的论断：
> 随着时间的推移($t$增加)，希望尽可能减小$z_{n_p}(t)$与$z_{ref}(t)$之间的差距从而使得MPC与RL的预设值尽可能贴近，这样（可能）有益于RL的训练。

为此采用Perscribed Performance Control(PPC)的技术。大概思路是：
定义误差为：
$$
e(s) = z_{n_p}(s) - z_{ref}(s)
$$
其中$s$代表训练的step，不限定在一局中（当然，应该也可以限定在一局当中...）。设$e$进行error transformation后的变量为$\zeta$，则PPC的$e$有约束问题转化为了无约束的$\zeta$有界问题。
1. 可以给MPC的cost function中添加一项：$\Vert\zeta\Vert_T$，通过这种手法帮助MPC逐渐收敛到$z_{ref}$上去。
2. 不能单调地只去允许MPC，还要通过某种手法让RL同样也能够考虑MPC的感受。
对此，你有什么补充的想法或者建议么？
# Solving the open-pit mine truck scheduling problem based on Gurobi 

This code repository uses the Gurobi mathematical programming solver to model and solve the open-pit mine truck scheduling optimization problem, modeling it as a mixed integer programming model, and implementing the solution process based on Python and C++ programming languages. In order to further improve the optimization speed of the Gurobi solver, heuristic rules of actual problems are used to conduct random searches for feasible solutions and provide the solver with initial feasible solutions. In the future, evolutionary algorithms can be combined to provide a set of feasible and high-quality solutions. Experimental results show that using prior knowledge or heuristic information of actual problems can significantly improve solver search efficiency; at the same time, the results also show that for NP-hard problems such as mixed integer programming problems, the solution complexity of mathematical programming methods will increase with the scale of the problem. The increase is exponential. On the contrary, the evolutionary algorithm has considerable application prospects for this scenario. The modeling, solution and related experimental ideas of this problem are posted on the official blog: [Gurobi 露天矿卡车调度问题求解](https://mp.weixin.qq.com/s/BfigccrrrdIFzDX71kbV8A)
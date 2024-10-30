import gurobipy as gp
from gurobipy import GRB

# 初始化数据
K = 3  # 矿卡数量
I = 2  # 装载点数量
J = 2  # 卸载点数量
M = I + J  # 装卸点数量
T = 60.0  # 班次时间
MLk = [15 for _ in range(K)]  # 每辆卡车的最大运输路线长度
ck = [20 for _ in range(K)]  # 每辆卡车的容量
ttk = [[
    [ 0, 0, 9.01, 6.19 ],
    [ 0, 0, 4.82, 4.92 ],
    [ 5.43, 2.50, 0, 0 ],
    [ 5.51, 2.58, 0, 0 ]
] for _ in range(K)]  # 装卸点之间的重载/空载时间
ltk = [[3.8 for _ in range(M)] for _ in range(K)]  # 每辆卡车在装载点的装载时间
utk = [1.5 for _ in range(K)]  # 每辆卡车的卸载时间

# 创建模型
model = gp.Model("TruckScheduling")

# 参数设置

# model.Params.Method = 4  # 默认值：-1，自动决定优化方法。0：原始单纯型；1：对偶；2：Barrier; 3: 随机并行；4：确定并行。
# model.Params.TimeLimit = 180  # 当达到规定的运行时间后，优化终止。单位为秒。
model.Params.MipGap = 0.02  # 当整数规划的偏差下降到设定值后，优化终止，默认为 0。

# 决策变量
x = {}  # 矿卡运输路线
for k in range(K):
    for m in range(M):
        for n in range(MLk[k]):
            x[k, m, n] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{m}_{n}")

t = {}  # 矿卡到达时间
for k in range(K):
    for n in range(MLk[k]):
        t[k, n] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=2*T, name=f"t_{k}_{n}")

w = {}  # 矿卡排队等待时间
w_aux = {}  # 等待时间辅助变量
w_aux1 = {}
for k in range(K):
    for n in range(MLk[k]):
        w[k, n] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=2*T, name=f"w_{k}_{n}")
        w_aux[k, n] = model.addVar(vtype=GRB.CONTINUOUS, lb=-2*T, ub=2*T, name=f"w_aux_{k}_{n}")
        w_aux1[k, n] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=2*T, name=f"w_{k}_{n}")

tt = {}  # 矿卡运输时间
for k in range(K):
    for n in range(MLk[k]):
        tt[k, n] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=T, name=f"tt_{k}_{n}")

rl_last = {}  # 运输路线长度
rl_next = {}
rl_state = {}
rl = {}  # rl_next长度
for k in range(K):
    rl[k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=MLk[k], name=f"rl_{k}")
    for n in range(MLk[k]):
        rl_last[k, n] = model.addVar(vtype=GRB.BINARY, name=f"rl_last_{k}_{n}")
        rl_next[k, n] = model.addVar(vtype=GRB.BINARY, name=f"rl_next_{k}_{n}")
        rl_state[k, n] = model.addVar(vtype=GRB.BINARY, name=f"rl_next_{k}_{n}")

front_queue = {}  # 当前卡车前置队列，用于判断哪些卡车提前到达
for k in range(K):
    for n in range(MLk[k]):
        for k_prime in range(K):
            for n_prime in range(MLk[k]):
                front_queue[k, n, k_prime, n_prime] = model.addVar(vtype=GRB.BINARY, name=f"front_queue_{k}_{n}_{k_prime}_{n_prime}")

x_aux = {}  # x[k, m, n] * x[k_prime, m, n_prime] 辅助变量
x_aux_prime = {}  # x[k, m, n] * x[k_prime, m, n_prime] * front_queue[k, n, k_prime, n_prime] 辅助变量
for k in range(K):
    for n in range(MLk[k]):
        for k_prime in range(K):
            for n_prime in range(MLk[k]):
                for m in range(M):
                    x_aux[k, n, k_prime, n_prime, m] = model.addVar(vtype=GRB.BINARY, name=f"x_aux_{k}_{n}_{k_prime}_{n_prime}_{m}")
                    x_aux_prime[k, n, k_prime, n_prime, m] = model.addVar(vtype=GRB.BINARY, name=f"x_aux_prime_{k}_{n}_{k_prime}_{n_prime}_{m}")

wait_time = {}  # 其他卡车排队时间
for k in range(K):
    for n in range(MLk[k]):
        for k_prime in range(K):
            for n_prime in range(MLk[k]):
                wait_time[k, n, k_prime, n_prime] = model.addVar(vtype=GRB.CONTINUOUS, lb=-2*T, ub=2*T, name=f"wait_time_{k}_{n}_{k_prime}_{n_prime}")


# 设定一个极小的松弛量 epsilon
epsilon = 1e-4

# 目标函数：最大化运输量
model.setObjective(gp.quicksum(ck[k] * rl[k] / 2 for k in range(K)), GRB.MAXIMIZE)

# 约束1: 运输路线长度限制
model.addConstrs((gp.quicksum(rl_last[k, n] for n in range(MLk[k])) == 1 for k in range(K)), name="rl_last")
model.addConstrs((gp.quicksum(rl_next[k, n] for n in range(MLk[k])) == 1 for k in range(K)), name="rl_next")
model.addConstrs((gp.quicksum((rl_next[k, n] - rl_last[k, n]) * n for n in range(MLk[k])) == 1 for k in range(K)), name="diff_last_next")
model.addConstrs((rl[k] == gp.quicksum(rl_next[k, n] * n for n in range(MLk[k])) for k in range(K)), name="rl")
model.addConstrs((((rl_state[k, n] == 1) >> (n <= rl[k])) for k in range(K) for n in range(MLk[k])), name="rl_state1")
model.addConstrs((((rl_state[k, n] == 0) >> (n >= rl[k] + 1)) for k in range(K) for n in range(MLk[k])), name="rl_state0")

# 约束2：每辆卡车最后一个目的地的到达时间不能超过班次时间且卡车正好运输完整个班次
model.addConstrs((gp.quicksum(rl_last[k, n] * t[k, n] for n in range(MLk[k])) <= T for k in range(K)), name="last_time_limit")
model.addConstrs((gp.quicksum(rl_next[k, n] * t[k, n] for n in range(MLk[k])) >= T + epsilon for k in range(K)), name="next_time_limit")

# 约束3：矿卡每个目的地的到达时间应为上一个点的到达时间+排队时间+运输时间+装/卸载时间
model.addConstrs((t[k, n + 1] == t[k, n] + w[k, n] + gp.quicksum(x[k, m, n] * ltk[k][m] for m in range(M)) +
                  gp.quicksum(x[k, m, n] * x[k, m_prime, n + 1] * ttk[k][m][m_prime] for m in range(M) for m_prime in range(M))
                  for n in range(0, MLk[k] - 1, 2) for k in range(K)))

model.addConstrs((t[k, n + 1] == t[k, n] + w[k, n] + utk[k] +
                  gp.quicksum(x[k, m, n] * x[k, m_prime, n + 1] * ttk[k][m][m_prime] for m in range(M) for m_prime in range(M))
                  for n in range(1, MLk[k] - 1, 2) for k in range(K)))

# 约束4：卡车的排队等待时间取决于当前装卸点前面的卡车是否正在进行装卸
model.addConstrs(((front_queue[k, n, k_prime, n_prime] == 1) >> (t[k_prime, n_prime] <= t[k, n]) for k in range(K) for n in range(MLk[k]) for k_prime in range(K) for n_prime in range(MLk[k])), name="front_queue1")
model.addConstrs(((front_queue[k, n, k_prime, n_prime] == 0) >> (t[k_prime, n_prime] >= t[k, n] + epsilon) for k in range(K) for n in range(MLk[k]) for k_prime in range(K) for n_prime in range(MLk[k])), name="front_queue0")
model.addConstrs((x_aux[k, n, k_prime, n_prime, m] == x[k, m, n] * x[k_prime, m, n_prime] for k in range(K) for n in range(MLk[k]) for k_prime in range(K) for n_prime in range(MLk[k]) for m in range(M)), name="x_aux")
model.addConstrs((x_aux_prime[k, n, k_prime, n_prime, m] == x_aux[k, n, k_prime, n_prime, m] * front_queue[k, n, k_prime, n_prime] for k in range(K) for n in range(MLk[k]) for k_prime in range(K) for n_prime in range(MLk[k]) for m in range(M)), name="x_aux_prime")

for k in range(K):
    for n in range(0, MLk[k], 2):  # 奇数点（装载点）

        model.addConstrs(wait_time[k, n, k_prime, n_prime] == gp.quicksum(x_aux_prime[k, n, k_prime, n_prime, m] * (t[k_prime, n_prime] + w[k_prime, n_prime] + ltk[k_prime][m] - t[k, n]) for m in range(M)) for k_prime in range(K) for n_prime in range(MLk[k_prime]) if k_prime != k)
        model.addConstr(w_aux[k, n] == gp.max_(wait_time[k, n, k_prime, n_prime] for k_prime in range(K) for n_prime in range(MLk[k_prime]) if k_prime != k), name=f"w_aux_{k}_{n}")

    for n in range(1, MLk[k], 2):  # 偶数点（卸载点）

        model.addConstrs(wait_time[k, n, k_prime, n_prime] == gp.quicksum(x_aux_prime[k, n, k_prime, n_prime, m] * (t[k_prime, n_prime] + w[k_prime, n_prime] + utk[k_prime] - t[k, n]) for m in range(M)) for k_prime in range(K) for n_prime in range(MLk[k_prime]) if k_prime != k)
        model.addConstr(w_aux[k, n] == gp.max_(wait_time[k, n, k_prime, n_prime] for k_prime in range(K) for n_prime in range(MLk[k_prime]) if k_prime != k), name=f"w_aux_{k}_{n}")

# model.addConstrs((w[k, n] == gp.max_(0, w_aux[k, n]) for k in range(K) for n in range(MLk[k])), name="waiting_time")

model.addConstrs((w_aux1[k, n] == gp.max_(0, w_aux[k, n]) for k in range(K) for n in range(MLk[k])))
model.addConstrs(((rl_state[k, n] == 1) >> (w[k, n] == w_aux1[k, n]) for k in range(K) for n in range(MLk[k])), name="waiting_time")

# 约束5：所有卡车默认同时由装载点出发，第一个目的地的时间为0
model.addConstrs((t[k, 0] == epsilon * k for k in range(K)), name="start_time")

# 约束6：每辆卡车每次只能前往一个目的地
model.addConstrs((gp.quicksum(x[k, m, n] for m in range(M)) == 1 for k in range(K) for n in range(MLk[k])), name="des_constr")

# 约束7：卡车初始由装载点出发，循环往返于装卸点之间
model.addConstrs((gp.quicksum(x[k, m, n] for m in range(I)) == 1 for k in range(K) for n in range(0, MLk[k], 2)), name="load_constr")
model.addConstrs((gp.quicksum(x[k, m, n] for m in range(I)) == 0 for k in range(K) for n in range(1, MLk[k], 2)), name="unload_constr")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    # for var in model.getVars():
    #     print(var.VarName, var.X)
else:
    print("No optimal solution found")


if model.status == GRB.OPTIMAL:

    routes = []

    print("\ntruck routes:")
    for k in range(K):
        truck_x = [[round(x[k, m, n].X) for m in range(M)] for n in range(MLk[k])]
        route = [sum(i * m[i] for i in range(len(m))) for m in truck_x]
        routes.append(route)
        print(truck_x)

    for k in range(K):
        for n in range(len(routes[k])):
            if (n + 1) % 2 == 1:
                routes[k][n] = chr(ord('A') + routes[k][n])
            else:
                routes[k][n] = chr(ord('a') + routes[k][n] - I)

    print("\narrive time:")    
    for k in range(K):
        for n in range(MLk[k]):
            time = round(t[k, n].X, 2)
            print(f'{time}({routes[k][n]})', end = ' ')
        print('')

    print("\nwait time:")
    for k in range(K):
        time = [round(w[k, n].X, 2) for n in range(MLk[k])]
        print(time)

    print("\naux wait time:")
    for k in range(K):
        time = [round(w_aux[k, n].X, 2) for n in range(MLk[k])]
        print(time)

    print("\nroute length last:")
    for k in range(K):
        length = [round(rl_last[k, n].X) for n in range(MLk[k])]
        print(length)

    print("\nroute length next:")
    for k in range(K):
        length = [round(rl_next[k, n].X) for n in range(MLk[k])]
        print(length)

    print("\nroute state:")
    for k in range(K):
        length = [round(rl_state[k, n].X) for n in range(MLk[k])]
        print(length)

    print("\nlength:")
    length = [round(rl[k].X) for k in range(K)]
    print(length)

    with open('output.txt', 'w') as file:

        file.write('\nfront_queue \n')

        for k in range(K):
            for n in range(MLk[k]):
                file.write(f'\nk: {k}, n: {n}\n')

                for k_prime in range(K):
                    for n_prime in range(MLk[k]):
                        file.write(f'{round(front_queue[k, n, k_prime, n_prime].X)} ')
                    file.write('\n')

        file.write('\nwait_time \n')

        for k in range(K):
            for n in range(MLk[k]):
                file.write(f'\nk: {k}, n: {n}\n')

                for k_prime in range(K):
                    for n_prime in range(MLk[k]):
                        file.write(f'{round(wait_time[k, n, k_prime, n_prime].X, 2)} ')
                    file.write('\n')

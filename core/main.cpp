#include <gurobi_c++.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <limits>

int randi(int a, int b) {
    return a + rand() % (b - a + 1);
}

struct TruckInf {
    double arriveTime;       //到达时间
    double workTime;         //开始装/卸车时间
    double leaveTime;        //装/卸车完成时间
    size_t truck;
    size_t point;
    double progress;         //当前装载点挖掘进度
    size_t shovelPoint;      //当前走铲点位置
    double pointProgress;    //当前走铲点挖掘进度
    double grade;            //当前走铲点品位
};

int main() {

    srand(time(NULL));

    // 初始化数据
    int K = 3; // 矿卡数量
    int I = 2; // 装载点数量
    int J = 2; // 卸载点数量
    int M = I + J; // 装卸点数量
    double T = 120.0; // 班次时间
    std::vector<int> MLk(K, 25); // 每辆卡车的最大运输路线长度
    std::vector<int> ck(K, 20); // 每辆卡车的容量
    std::vector<std::vector<std::vector<double>>> ttk(K, {
        { 0, 0, 9.01, 6.19 },
        { 0, 0, 4.82, 4.92 },
        { 5.43, 2.50, 0, 0 },
        { 5.51, 2.58, 0, 0 }
    });  // 装卸点之间的重载/空载时间
    std::vector<std::vector<double>> ltk(K, std::vector<double>(M, 3.8)); // 装载时间
    std::vector<double> utk(K, 1.5); // 卸载时间

    int Num_Initial_Solution = 1000;

    try {

        // 创建模型
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);

        // 参数设置
        model.set(GRB_DoubleParam_MIPGap, 0.02);  // 当整数规划的偏差下降到设定值后，优化终止，默认为 0。
        // model.set(GRB_IntParam_Method, 3);  // 默认值：-1，自动决定优化方法。0：原始单纯型；1：对偶；2：Barrier; 3: 随机并行；4：确定并行。
        // model.set(GRB_DoubleParam_TimeLimit, 180);  // 当达到规定的运行时间后，优化终止。单位为秒。

        // 决策变量
        std::vector<std::vector<std::vector<GRBVar>>> x(K);  // 矿卡运输路线
        std::vector<std::vector<GRBVar>> t(K);  // 矿卡到达时间
        std::vector<std::vector<GRBVar>> tt(K);  // 运输时间
        std::vector<std::vector<GRBVar>> w(K), w_aux(K), w_aux_prime(K);  // 矿卡排队等待时间
        std::vector<GRBVar> rl(K);  // 运输路线长度
        std::vector<std::vector<GRBVar>> rl_last(K), rl_next(K), rl_state(K);
        std::vector<std::vector<std::vector<std::vector<GRBVar>>>> front_queue(K);  // 当前卡车前置队列，用于判断哪些卡车提前到达
        std::vector<std::vector<std::vector<std::vector<std::vector<GRBVar>>>>> x_aux(K), x_aux_prime(K);  // x_aux = x[k, m, n] * x[k_prime, m, n_prime], x_aux_prime = x_aux * front_queue[k, n, k_prime, n_prime]
        std::vector<std::vector<std::vector<std::vector<GRBVar>>>> wait_time(K);  // 其他卡车排队时间

        double epsilon = 1e-4;

        for (int k = 0; k < K; ++k) {
            x[k].resize(M);

            for (int m = 0; m < M; ++m) {
                x[k][m].resize(MLk[k]);
                for (int n = 0; n < MLk[k]; ++n) {
                    x[k][m][n] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                }
            }

            rl[k] = model.addVar(0.0, MLk[k], 0.0, GRB_INTEGER);

            t[k].resize(MLk[k]);
            tt[k].resize(MLk[k]);
            w[k].resize(MLk[k]);
            w_aux[k].resize(MLk[k]);
            w_aux_prime[k].resize(MLk[k]);
            rl_last[k].resize(MLk[k]);
            rl_next[k].resize(MLk[k]);
            rl_state[k].resize(MLk[k]);
            front_queue[k].resize(MLk[k]);
            x_aux[k].resize(MLk[k]);
            x_aux_prime[k].resize(MLk[k]);
            wait_time[k].resize(MLk[k]);

            for (int n = 0; n < MLk[k]; ++n) {
                front_queue[k][n].resize(K);
                x_aux[k][n].resize(K);
                x_aux_prime[k][n].resize(K);
                wait_time[k][n].resize(K);

                t[k][n] = model.addVar(0.0, 2 * T, 0.0, GRB_CONTINUOUS);
                tt[k][n] = model.addVar(0.0, T, 0.0, GRB_CONTINUOUS);
                w[k][n] = model.addVar(0.0, 2 * T, 0.0, GRB_CONTINUOUS);
                w_aux[k][n] = model.addVar(-2 * T, 2 * T, 0.0, GRB_CONTINUOUS);
                w_aux_prime[k][n] = model.addVar(0.0, 2 * T, 0.0, GRB_CONTINUOUS);
                rl_last[k][n] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                rl_next[k][n] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                rl_state[k][n] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);

                for (int k_prime = 0; k_prime < K; ++k_prime) {
                    front_queue[k][n][k_prime].resize(MLk[k_prime]);
                    x_aux[k][n][k_prime].resize(MLk[k_prime]);
                    x_aux_prime[k][n][k_prime].resize(MLk[k_prime]);
                    wait_time[k][n][k_prime].resize(MLk[k_prime]);

                    for (int n_prime = 0; n_prime < MLk[k_prime]; ++n_prime) {
                        x_aux[k][n][k_prime][n_prime].resize(M);
                        x_aux_prime[k][n][k_prime][n_prime].resize(M);

                        front_queue[k][n][k_prime][n_prime] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                        wait_time[k][n][k_prime][n_prime] = model.addVar(-2 * T, 2 * T, 0.0, GRB_CONTINUOUS);

                        for (int m = 0; m < M; ++m) {
                            x_aux[k][n][k_prime][n_prime][m] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                            x_aux_prime[k][n][k_prime][n_prime][m] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
                        }
                    }
                }
            }
        }

        // 目标函数：最大化运输量

        GRBLinExpr objective = 0;
        for (int k = 0; k < K; ++k) {
            objective += ck[k] * rl[k] / 2;
        }
        model.setObjective(objective, GRB_MAXIMIZE);

        // 约束1: 运输路线长度限制

        for (int k = 0; k < K; ++k) {
            GRBLinExpr rl_last_expr = 0;
            GRBLinExpr rl_next_expr = 0;
            GRBLinExpr last_next_expr = 0;
            GRBLinExpr rl_expr = 0;

            for (int n = 0; n < MLk[k]; ++n) {
                rl_last_expr += rl_last[k][n];
                rl_next_expr += rl_next[k][n];
                last_next_expr += (rl_next[k][n] - rl_last[k][n]) * n;
                rl_expr += rl_next[k][n] * n;
            }
            model.addConstr(rl_last_expr == 1);
            model.addConstr(rl_next_expr == 1);
            model.addConstr(last_next_expr == 1);
            model.addConstr(rl_expr == rl[k]);

            for (int n = 0; n < MLk[k]; ++n) {
                model.addGenConstrIndicator(rl_state[k][n], 1, n <= rl[k]);
                model.addGenConstrIndicator(rl_state[k][n], 0, n >= rl[k] + 1);
            }
        }

        // 约束2：每辆卡车最后一个目的地的到达时间不能超过班次时间且卡车正好运输完整个班次

        for (int k = 0; k < K; ++k) {
            GRBQuadExpr last_time_expr = 0;
            GRBQuadExpr next_time_expr = 0;
            for (int n = 0; n < MLk[k]; ++n) {
                last_time_expr += rl_last[k][n] * t[k][n];
                next_time_expr += rl_next[k][n] * t[k][n];
            }
            model.addQConstr(last_time_expr <= T);
            model.addQConstr(next_time_expr >= T + epsilon);
        }

        // 约束3：矿卡每个目的地的到达时间应为上一个点的到达时间+排队时间+运输时间+装/卸载时间

        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < MLk[k] - 1; ++n) {
                GRBQuadExpr quadExpr;
                for (int m = 0; m < M; ++m) {
                    for (int m_prime = 0; m_prime < M; ++m_prime) {
                        quadExpr += x[k][m][n] * x[k][m_prime][n + 1] * ttk[k][m][m_prime];
                    }
                }
                model.addQConstr(tt[k][n] == quadExpr);

                GRBLinExpr expr = t[k][n] + w[k][n] + tt[k][n];
                if((n + 1) % 2 == 1) {
                    for (int m = 0; m < M; ++m) {
                        expr += x[k][m][n] * ltk[k][m];
                    }
                }
                else expr += utk[k];
                model.addConstr(t[k][n + 1] == expr);
                // model.addGenConstrIndicator(rl_state[k][n + 1], 1, t[k][n + 1] == expr);
                // model.addGenConstrIndicator(rl_state[k][n + 1], 0, t[k][n + 1] == 2 * T);
            }
        }

        // 约束4：卡车的排队等待时间取决于当前装卸点前面的卡车是否正在进行装卸

        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < MLk[k]; ++n) {
                for (int k_prime = 0; k_prime < K; ++k_prime) {
                    for (int n_prime = 0; n_prime < MLk[k_prime]; ++n_prime) {
                        model.addGenConstrIndicator(front_queue[k][n][k_prime][n_prime], 1, t[k_prime][n_prime] <= t[k][n]);
                        model.addGenConstrIndicator(front_queue[k][n][k_prime][n_prime], 0, t[k_prime][n_prime] >= t[k][n] + epsilon);

                        for (int m = 0; m < M; ++m) {
                            model.addQConstr(x_aux[k][n][k_prime][n_prime][m] == x[k][m][n] * x[k_prime][m][n_prime]);
                            model.addQConstr(x_aux_prime[k][n][k_prime][n_prime][m] == x_aux[k][n][k_prime][n_prime][m] * front_queue[k][n][k_prime][n_prime]);
                        }

                        if(k_prime == k) continue;
                        GRBQuadExpr quadExpr;
                        for (int m = 0; m < M; ++m) {
                            if((n + 1) % 2 == 1) quadExpr += x_aux_prime[k][n][k_prime][n_prime][m] * (t[k_prime][n_prime] + w[k_prime][n_prime] + ltk[k_prime][m] - t[k][n]);
                            else quadExpr += x_aux_prime[k][n][k_prime][n_prime][m] * (t[k_prime][n_prime] + w[k_prime][n_prime] + utk[k_prime] - t[k][n]);
                        }
                        model.addQConstr(wait_time[k][n][k_prime][n_prime] == quadExpr);
                    }
                }

                std::vector<GRBVar> waitVars;
                for (int k_prime = 0; k_prime < K; ++k_prime) {
                    if (k_prime != k) {
                        for (int n_prime = 0; n_prime < MLk[k_prime]; ++n_prime) {
                            waitVars.push_back(wait_time[k][n][k_prime][n_prime]);
                        }
                    }
                }
                GRBVar* waitVarsPtr = waitVars.data();
                int len = waitVars.size();
                model.addGenConstrMax(w_aux[k][n], waitVarsPtr, len);
                model.addGenConstrMax(w_aux_prime[k][n], &w_aux[k][n], 1, 0);
                model.addGenConstrIndicator(rl_state[k][n], 1, w[k][n] == w_aux_prime[k][n]);
                // model.addConstr(w[k][n] == w_aux_prime[k][n]);
            }
        }

        // 约束5：所有卡车默认同时由装载点出发，第一个目的地的时间为0

        for (int k = 0; k < K; ++k) {
            model.addConstr(t[k][0] == epsilon * k);
        }

        // 约束6：每辆卡车每次只能前往一个目的地

        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < MLk[k]; ++n) {
                GRBLinExpr expr = 0;
                for (int m = 0; m < M; ++m) {
                    expr += x[k][m][n];
                }
                model.addConstr(expr == 1);
            }
        }

        // 约束7：卡车初始由装载点出发，循环往返于装卸点之间

        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < MLk[k]; ++n) {
                GRBLinExpr expr = 0;
                for (int m = 0; m < I; ++m) {
                    expr += x[k][m][n];
                }
                if((n + 1) % 2 == 1) model.addConstr(expr == 1);
                else model.addConstr(expr == 0);
            }
        }

        // 设置初始可行解

        model.set(GRB_IntAttr_NumStart, 1);
        model.set(GRB_IntParam_StartNumber, 0);

        std::vector<std::vector<int>> best_routes(K);
        std::vector<std::vector<double>> best_arrive_time(K);
        std::vector<std::vector<double>> best_queue_time(K);
        double max_p = std::numeric_limits<double>::min();

        for (size_t index = 0; index < Num_Initial_Solution; index++)
        {

            // 生成路线
            std::vector<std::vector<int>> routes(K);
            std::vector<std::vector<double>> arrive_time(K);
            std::vector<std::vector<double>> queue_time(K);
            for (int k = 0; k < K; ++k) {
                routes[k].resize(MLk[k]);
                arrive_time[k].resize(MLk[k]);
                queue_time[k].resize(MLk[k]);
                for (int n = 0; n < MLk[k]; ++n) {
                    if((n + 1) % 2 == 1) routes[k][n] = randi(0, I - 1);
                    else routes[k][n] = randi(I, M - 1);
                }
            }
            
            // 计算初始到达时间
            for (int k = 0; k < K; ++k) {
                double cur_route_time = epsilon * k;
                arrive_time[k][0] = epsilon * k;
                for (int n = 1; n < MLk[k]; ++n) {
                    bool is_load = (n + 1) % 2;
                    if(is_load) cur_route_time += utk[k] + ttk[k][routes[k][n - 1]][routes[k][n]];
                    else cur_route_time += ltk[k][routes[k][n - 1]] + ttk[k][routes[k][n - 1]][routes[k][n]];
                    arrive_time[k][n] = cur_route_time;
                }
            }

            std::vector<int> count(K, 0);
            std::vector<std::vector<TruckInf>> point_queue(M);

            // 模拟调度运输

            while(1) {
                int best_truck = -1;
				double min_time = std::numeric_limits<double>::max();
				for (size_t i = 0; i < count.size(); i++)
				{
					if (count[i] == -1) continue;
					if (arrive_time[i][count[i]] < min_time) {
						min_time = arrive_time[i][count[i]];
						best_truck = i;
					}
				}
				if (best_truck == -1) break;

				bool is_load = (count[best_truck] + 1) % 2;
				int cur_point = routes[best_truck][count[best_truck]];
				std::vector<TruckInf>& sequence = point_queue[cur_point];
				double arrive = arrive_time[best_truck][count[best_truck]];
				double handle = 0;
				if (is_load) handle = ltk[best_truck][cur_point];
				else handle = utk[best_truck];

				TruckInf info;
				info.truck = best_truck;
				info.point = count[best_truck];
				info.arriveTime = arrive;

				bool need_queue = false;
				if (sequence.empty()) {
					info.workTime = arrive;
					info.leaveTime = arrive + handle;
				}
				else {
					TruckInf& last_info = sequence.back();
                    if (last_info.leaveTime > arrive) {
                        info.workTime = last_info.leaveTime;
                        info.leaveTime = last_info.leaveTime + handle;
                        need_queue = true;
                    }
                    else {
                        info.workTime = arrive;
                        info.leaveTime = arrive + handle;
                    }
				}
				sequence.push_back(info);
                queue_time[info.truck][info.point] = info.workTime - info.arriveTime;

				if (need_queue) {
					for (size_t i = count[best_truck] + 1; i < arrive_time[best_truck].size(); i++)
					{
						arrive_time[best_truck][i] += info.workTime - info.arriveTime;
					}
				}
				count[best_truck]++;
				if (count[best_truck] == routes[best_truck].size()) count[best_truck] = -1;
            }

            double production = 0;
            for (size_t k = 0; k < K; k++)
            {
                double length = -1;
                for (int n = 1; n < MLk[k]; ++n) {
                    if(arrive_time[k][n] > T && arrive_time[k][n - 1] <= T) length = n;
                }
                production += ck[k] * length / 2;
            }

            if(production > max_p) {
                max_p = production;
                best_routes = routes;
                best_arrive_time = arrive_time;
                best_queue_time = queue_time;

                std::cout << "Find better solution: " << production << "(" << index + 1 << ")" << std::endl;
            }
            
            
            // 设置决策变量

            if(index == Num_Initial_Solution - 1) {
                for (int k = 0; k < K; ++k) {
                    int length = -1;
                    for (int n = 0; n < MLk[k]; ++n) {
                        for (int m = 0; m < M; ++m) {
                            if(m == best_routes[k][n]) x[k][m][n].set(GRB_DoubleAttr_Start, 1);
                            else x[k][m][n].set(GRB_DoubleAttr_Start, 0);
                        }
                        t[k][n].set(GRB_DoubleAttr_Start, best_arrive_time[k][n]);
                        if(n < MLk[k] - 1) tt[k][n].set(GRB_DoubleAttr_Start, ttk[k][best_routes[k][n]][best_routes[k][n + 1]]);
                        else tt[k][n].set(GRB_DoubleAttr_Start, 0);

                        if(n > 0) {
                            if(best_arrive_time[k][n] > T && best_arrive_time[k][n - 1] <= T) length = n;
                        }

                        double max_time = std::numeric_limits<double>::min();
                        for (int k_prime = 0; k_prime < K; ++k_prime) {
                            for (int n_prime = 0; n_prime < MLk[k_prime]; ++n_prime) {
                                int front_state = 0;
                                if(best_arrive_time[k_prime][n_prime] <= best_arrive_time[k][n]) front_state = 1;
                                else front_state = 0;
                                front_queue[k][n][k_prime][n_prime].set(GRB_DoubleAttr_Start, front_state);

                                for (int m = 0; m < M; ++m) {
                                    int aux_state = 0;
                                    if(best_routes[k][n] == m && best_routes[k_prime][n_prime] == m) aux_state = 1;
                                    else aux_state = 0;
                                    x_aux[k][n][k_prime][n_prime][m].set(GRB_DoubleAttr_Start, aux_state);
                                    x_aux_prime[k][n][k_prime][n_prime][m].set(GRB_DoubleAttr_Start, front_state * aux_state);
                                }
                                
                                double time = 0;
                                if(k_prime != k) {
                                    if(front_state == 1 && best_routes[k][n] == best_routes[k_prime][n_prime]) {
                                        time += best_arrive_time[k_prime][n_prime] + best_queue_time[k_prime][n_prime] - best_arrive_time[k][n];
                                        if((n + 1) % 2 == 1) time += ltk[k_prime][best_routes[k][n]];
                                        else time += utk[k_prime];
                                    }
                                }
                                if(time > max_time) max_time = time;
                                wait_time[k][n][k_prime][n_prime].set(GRB_DoubleAttr_Start, time);
                            }
                        }
                        w_aux[k][n].set(GRB_DoubleAttr_Start, max_time);
                        w_aux_prime[k][n].set(GRB_DoubleAttr_Start, max_time < 0 ? 0 : max_time);
                        w[k][n].set(GRB_DoubleAttr_Start, best_queue_time[k][n]);
                    }
                    rl[k].set(GRB_DoubleAttr_Start, length);
                    for (int n = 0; n < MLk[k]; ++n) {
                        if(n == length - 1) rl_last[k][n].set(GRB_DoubleAttr_Start, 1);
                        else rl_last[k][n].set(GRB_DoubleAttr_Start, 0);

                        if(n == length) rl_next[k][n].set(GRB_DoubleAttr_Start, 1);
                        else rl_next[k][n].set(GRB_DoubleAttr_Start, 0);

                        if(n <= length) rl_state[k][n].set(GRB_DoubleAttr_Start, 1);
                        else rl_state[k][n].set(GRB_DoubleAttr_Start, 0);
                    }
                }
            }
        }

        // 求解模型
        model.update();
        model.optimize();

        // 输出结果
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
            std::cout << std::endl << "Optimal solution found:" << std::endl;
            
            std::cout << "\ntruck routes:" << std::endl;

            std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2);

            std::vector<std::vector<char>> routes(K);
            for (int k = 0; k < K; ++k) {
                routes[k].resize(MLk[k]);
                for (int n = 0; n < MLk[k]; ++n) {
                    int point = 0;
                    for (int m = 0; m < M; ++m) point += m * x[k][m][n].get(GRB_DoubleAttr_X);
                    if((n + 1) % 2 == 1) routes[k][n] = 'A' + point;
                    else routes[k][n] = 'a' + point - I;

                    double time = t[k][n].get(GRB_DoubleAttr_X);
                    std::cout << time << "(" << routes[k][n] << ") ";
                }
                std::cout << std::endl;
            }

            std::cout << "\nwait time:" << std::endl;
            for (int k = 0; k < K; ++k) {
                for (int n = 0; n < MLk[k]; ++n) {
                    double time = w[k][n].get(GRB_DoubleAttr_X);
                    std::cout << time << " ";
                }
                std::cout << std::endl;
            }

            std::cout << "\nroute length:" << std::endl;
            for (int k = 0; k < K; ++k) {
                int length = rl[k].get(GRB_DoubleAttr_X);
                std::cout << length << " ";
            }
            std::cout << std::endl;

        } else {
            std::cout << "No optimal solution found" << std::endl;
        }
    } catch(GRBException e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    } catch(...) {
        std::cout << "Error during optimization" << std::endl;
    }

    return 0;
}

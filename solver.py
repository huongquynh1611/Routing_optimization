import cplex
import pandas as pd
import numpy as np
import json
from os.path import dirname, abspath

url_logistic_route_navigation = dirname(dirname(dirname(dirname(abspath(__file__)))))


def setupproblem(source_file):
    print(source_file.upper())
    with open(str(source_file)) as json_file:
        data = json.load(json_file)

    C_ij = data['C_ij']
    C_w = data['C_w']
    CLT = data['C_LT']
    CPDm = data['C_PD_m']
    CPDp = data['C_PD_p']
    t_ij = data['t_ij']
    ET_i = data['ET_i']
    LT_i = data['LT_i']
    UDO_i = data['UDP']
    LDO_i = data['LDO']
    M = data['M']
    q = len(C_ij) - 1
    p = 0
    c = cplex.Cplex()
    x_ij = []
    name_xij = []
    obj_xij = []
    for i in range(len(C_ij)):
        x_ij.append([])
        for j in range(len(C_ij)):
            obj_xij.append(C_ij[i][j])
            varname = "x_" + str(i) + "_" + str(j)
            name_xij.append(varname)
            x_ij[i].append(varname)
    c.variables.add(names=name_xij, lb=[0] * len(name_xij),
                    ub=[1] * len(name_xij), types=["B"] * len(name_xij), obj=obj_xij)
    S_i = []
    for i in range(len(C_ij)):
        varname = "S_" + str(i)
        S_i.append(varname)
    c.variables.add(names=S_i, lb=ET_i, types=["C"] * len(S_i))
    D_i = []
    for i in range(len(C_ij)):
        varname = "D_" + str(i)
        D_i.append(varname)
    c.variables.add(names=D_i, lb=[1] * len(D_i), ub=[len(D_i)] * len(D_i), types=["I"] * len(D_i))
    W_i = []
    for i in range(len(C_ij)):
        varname = "W_" + str(i)
        W_i.append(varname)
    c.variables.add(names=W_i, lb=[0] * len(W_i), types=["C"] * len(W_i), obj=[C_w] * len(W_i))
    OT_i = []
    for i in range(len(C_ij)):
        varname = "OT_" + str(i)
        OT_i.append(varname)
    c.variables.add(names=OT_i, lb=[0] * len(OT_i), types=["C"] * len(OT_i), obj=CLT)
    PD_m_i = []
    for i in range(len(C_ij)):
        varname = "PD_m_" + str(i)
        PD_m_i.append(varname)
    c.variables.add(names=PD_m_i, lb=[0] * len(PD_m_i), types=["I"] * len(PD_m_i), obj=CPDm)
    PD_p_i = []
    for i in range(len(C_ij)):
        varname = "PD_p_" + str(i)
        PD_p_i.append(varname)
    c.variables.add(names=PD_p_i, lb=[0] * len(PD_p_i), types=["I"] * len(PD_p_i), obj=CPDp)
    for i in range(len(x_ij)):
        if i != q:
            thevars = []
            for j in range(len(x_ij)):
                if j != p:
                    thevars.append(x_ij[i][j])
            c.linear_constraints.add(
                lin_expr=[cplex.SparsePair(thevars, [1] * len(thevars))],
                senses=["E"],
                rhs=[1]
            )
    for j in range(len(x_ij)):
        if j != p:
            thevars = []
            for i in range(len(x_ij)):
                if i != q:
                    thevars.append(x_ij[i][j])
            c.linear_constraints.add(
                lin_expr=[cplex.SparsePair(thevars, [1] * len(thevars))],
                senses=["E"],
                rhs=[1]
            )
    for i in range(len(x_ij)):
        for j in range(len(x_ij)):
            if i != j:
                c.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([S_i[i], S_i[j], W_i[j], x_ij[i][j]], [1, -1, 1, M])],
                    senses=["L"],
                    rhs=[M - t_ij[i][j]]
                )
            else:
                c.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([W_i[j], x_ij[i][j]], [1, M])],
                    senses=["L"],
                    rhs=[M - t_ij[i][j]]
                )


    for i in range(len(S_i)):
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair([S_i[i], OT_i[i]], [1, -1])],
            senses=["L"],
            rhs=[LT_i[i]]
        )

    for i in range(len(x_ij)):
        for j in range(len(x_ij)):
            if i != j:
                c.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([D_i[i], x_ij[i][j], D_i[j]], [1, M, -1])],
                    senses=["L"],
                    rhs=[-1 + M]
                )
            else:
                c.linear_constraints.add(
                    lin_expr=[cplex.SparsePair([x_ij[i][j]], [M])],
                    senses=["L"],
                    rhs=[-1 + M]
                )
    c.linear_constraints.add(
        lin_expr=[cplex.SparsePair([D_i[p]], [1])],
        senses=["E"],
        rhs=[1]
    )

    c.linear_constraints.add(
        lin_expr=[cplex.SparsePair([D_i[q]], [1])],
        senses=["E"],
        rhs=[len(D_i)]
    )
    for i in range(len(D_i)):
        if (i != p) & (i != q):
            c.linear_constraints.add(
                lin_expr=[cplex.SparsePair([D_i[i], PD_m_i[i]], [1, 1])],
                senses=["G"],
                rhs=[LDO_i[i]]
            )
            c.linear_constraints.add(
                lin_expr=[cplex.SparsePair([D_i[i], PD_p_i[i]], [1, -1])],
                senses=["L"],
                rhs=[UDO_i[i]]
            )
    return c
def cplex_init(source_file):
    c = setupproblem(source_file)
    c.solve()
    sol = c.solution
    print(sol)
    print("Solution status = ", sol.get_status(), ":", end=' ')
    print(sol.status[sol.get_status()])
    if sol.is_primal_feasible():
        print("Solution value  = ", sol.get_objective_value())
        print(sol.get_values())
    else:
        print("No solution available.")
if __name__ == '__main__':
    print('main file')
    cplex_init('data_10_1.txt')

import doctest
import networkx as nx
import numpy as np
from fairpy.rent.Calculation_Assistance import *
from fairpy.agentlist import AgentList
from fairpy.rent.Algorithms import optimal_envy_free as oef
import logging
import time
from numba import jit

"""
    "Fair Rent Division on a Budget"
    
    Based on:
    "Fair Rent Division on a Budget" by Procaccia, A., Velez, R., & Yu, D. (2018), https://doi.org/10.1609/aaai.v32i1.11465
    The algorithm calculates Maximum-rent envy-free allocation in a fully connected economy, or in simple words,
    calculates a fair rent division under budget constraints.
    
    Programmers: Daniel Sela and Asif Rot
    Date: 27-12-2022
"""

LOG_FORMAT = "%(levelname)s, time: %(asctime)s, line: %(lineno)d - %(message)s"
logging.basicConfig(filename='algorithms_logging.log', level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()


def optimal_envy_free(agentsList: AgentList, rent: float, budget: dict) -> (dict, dict):
    """
        This function implements Algorithm 2 from the article.
        :param agentsList: Evaluation for each room by agent
        :param rent: Total rent
        :budget: Each agent's budget
        :return:  Optimal envy-free allocation subject to budget constraints.
                 µ: N -> A , p : is vector of prices for each room
        >>> agentList1 = AgentList({'Alice': {'2ndFloor': 250, 'Basement': 250, 'MasterBedroom': 500},'Bob': {'2ndFloor': 250, 'Basement': 250, 'MasterBedroom': 500},'Clair': {'2ndFloor': 250, 'Basement': 500, 'MasterBedroom': 250}})
        >>> optimal_envy_free(agentList1, 1000, {'Alice': 200,'Bob': 100,'Clair': 200,})
        --------THE RESULT--------
        'no solution'
        >>> agentList2 = AgentList({'P1': {'Ra': 600, 'Rb': 100, 'Rc': 150},'P2': {'Ra': 250, 'Rb': 250, 'Rc': 250},'P3': {'Ra': 100, 'Rb': 400, 'Rc': 250}})
        >>> optimal_envy_free(agentList2, 1000, {'P1': 600,'P2': 400,'P3': 400})
        ([('P1', 'Ra'), ('P2', 'Rc'), ('P3', 'Rb')], [('Rc', 166.67), ('Rb', 316.67), ('Ra', 516.67)])

        >>> ex3 = AgentList({"Alice":{'1' : 250, '2' : 750}, "Bob": {'1': 250, '2' : 750}})
        >>> optimal_envy_free(ex3, 1000, {'Alice': 600, 'Bob': 500})
        --------THE RESULT--------
        'no solution'
        """
    logger.info(f'optimal_envy_free({agentsList}, {rent}, {budget})')
    # line 48-55 : Taking the AgentList type and splitting it to lists and dictionary
    N = list([i for i in agentsList.agent_names()])
    A = list([i for i in agentsList.all_items()])

    val = {i: {j: agentsList[N.index(i)].value(j) for j in A} for i in N}
    logger.debug("done initializing the first variables")
    # compute (σ,p) ∈ F(N,A,v,r)
    sigma = {}
    p = {}
    sigma, p = spliddit(agentsList, rent)  # compute (σ,p) ∈ F(N,A,v,r)
    logger.debug("done spliddit")
    # Γ(σ, p) : weak envy graph by Γ(σ,p) ≡ (N,E), where (i, j) ∈ E if and only if viσ(i) − pσ(i) = viσ(j) − pσ(j)
    # graph = Γ(σ, p)
    graph = build_weak_envy_graph(sigma, p, agentsList)
    logger.debug("done build_weak_envy_graph")
    # SCC = C , C ← strongly connected components of Γ(σ, p)
    SCC = nx.strongly_connected_components(graph)
    logger.debug("done strongly_connected_components")

    ans_µc_pc = []
    """
    for each c ∈ C do:
        (μ_c,p_c) ← output of Algorithm 1 on C, A(C), vC,and bC
    """
    lst_SCC = list(SCC)
    for c in lst_SCC:
        # N_c = let N_c be the set of agent from strongly connected components
        N_c = list([i for i in N if i in c])
        # A_c = let A(C) be the set of rooms received by agents in C
        A_c = list([sigma[i] for i in sigma.keys() if i in c])
        # let vC and B_c be the restrictions of the two vectors to C
        V_c = {i: {j: agentsList[N_c.index(i)].value(j) for j in A_c} for i in c}
        B_c = {i: budget[i] for i in sigma.keys() if i in c}
        # New rent for calculation on c
        new_rent = sum(p[sigma[i]] for i in c)
        tempAgentC = {i: V_c[i] for i in c}
        # (μC,pC) ← output of Algorithm 1 on C, A(C), V_c, and B_c
        ans_µc_pc.append(maximum_rent_envy_free(AgentList(tempAgentC), new_rent, B_c))
        logger.debug("done appending ans_µc_pc with maximum_rent_envy_free")

    µ = {}

    # let μ:N →A s.t. for all i ∈ N , μ(i) = μ_C(i) for c ∈ C s.t. i ∈ C
    for i in agentsList.agent_names():
        for j in ans_µc_pc:
            µc = dict(j[1][0])
            if i in µc:
                µ[i] = µc[i]

    logger.debug("done comparing µ[i] = µc[i]")

    # if LP(1) for μ is feasible , ans_LP1 = (µ,p)
    ans_LP1 = LP1(µ, rent, val, budget)
    logger.debug("done LP1")
    if ans_LP1 != "no solution":
        # p ← solution of LP (1) for μ return (μ, p)
        # print(f"--------THE RESULT-------- \n µ : {ans_LP1[0]} , p : {ans_LP1[1]}")
        µ = sorted(ans_LP1[0].items(), key=lambda x: x[0])
        p = sorted(ans_LP1[1].items(), key=lambda x: x[1])
        logger.debug("done function (ans_LP1 != 'no solution')")
        return µ, p
    else:
        logger.debug("done function ('no solution')")
        print(f"--------THE RESULT--------")
        return "no solution"


def maximum_rent_envy_free(agentsList: AgentList, rent: float, budget: dict) -> (int, dict):
    """
    This function implements Algorithm 1 from the article.
    :param agentsList: Evaluation for each room by agent
    :param rent: Total rent
    :budget: Each agent's budget
    :return: Maximum-rent envy-free allocation in a fully connected economy.
             sigma: N -> A , p : is vector of prices for each room
    >>> ex1 = AgentList({"Alice":{'1' : 250, '2' : 250, '3' : 500}, "Bob": {'1': 250, '2' : 250, '3' :500}, "Clair": {'1' :250, '2' : 500, '3' : 250}})
    >>> maximum_rent_envy_free(ex1, 1000, {'Alice': 250, 'Bob': 320, 'Clair': 430})
    (1249.99, ([('Alice', '1'), ('Clair', '2'), ('Bob', '3')], [('1', 250.0), ('2', 500.0), ('3', 500.0)]))
    >>> ex2 = AgentList({"Alice":{'1' : 250, '2' : 750}, "Bob": {'1': 250, '2' : 750}})
    >>> maximum_rent_envy_free(ex2, 1000, {'Alice': 600, 'Bob': 500})
    (1700.0, ([('Alice', '1'), ('Bob', '2')], [('1', 600.0), ('2', 1100.0)]))
    >>> ex3 = AgentList({"Alice":{'1' : 400, '2' : 600}, "Bob": {'1': 300, '2' : 700}})
    >>> maximum_rent_envy_free(ex3, 1000, {'Alice': 450, 'Bob': 550})
    (1200.0, ([('Alice', '1'), ('Bob', '2')], [('1', 450.0), ('2', 750.0)]))
    """
    logger.info(f'maximum_rent_envy_free({agentsList}, {rent}, {budget})')
    N = list([i for i in agentsList.agent_names()])
    logger.debug("done initializing the first variables")
    sigma = {}
    p = {}
    sigma, p = spliddit(agentsList, rent)
    logger.debug("done spliddit")

    # It is for calculating the ∆, let Δ ∈ R such that
    # (σ,(p_a −Δ)_(a∈A)) ∈ F_b(N,C,v,r−nΔ) and there is i∈N such that p_(σ(i)) = b_i

    delta = min([p[sigma[i]] - budget[i] for i in sigma.keys()])
    logger.debug("done calculating delta")

    # p ← (p_(σ(i)) − Δ)_(i∈N)
    p = {i: round(v - delta, 3) for i, v in p.items()}
    logger.debug("done calculating p")
    # # r ← r − nΔ
    rent -= len(agentsList) * delta
    logger.debug("done calculating rent")

    # budget_graph
    # ---> Γ_b(σ,p) ≡ (N,E), where E = {(i, j) : v_(iσ(i)) − p_(σ(i)) = v_(iσ(j)) −p_(σ(i)) and p_(σ(j)) < b_i}
    budget_graph = build_budget_aware_graph(sigma, p, budget, agentsList)
    logger.debug("done build_budget_aware_graph")
    while not check_while(budget_graph, budget, p, sigma):
        # Case1
        if not case_1(budget, p, sigma):
            logger.debug("entering case_1")
            # Δ ← min i∈N (b_i − p_(σ(i)))
            delta = min([budget[i] - p[sigma[i]] for i in sigma.keys()])
            logger.debug("done min delta")
            # p ← (p_a + Δ)_(a∈A)
            p = {i: v + delta for i, v in p.items()}
            logger.debug("done calculating new p")
            # r ← r + nΔ
            rent = rent + len(N) * delta
            logger.debug("done calculating new rent")
        # Case2
        else:
            logger.debug("entering case_2")
            sigma = case_2(sigma, p, budget, agentsList)
            logger.debug("done case_2 function")
        # Update the new budget-aware graph
        budget_graph = build_budget_aware_graph(sigma, p, budget, agentsList)
        logger.debug("done build_budget_aware_graph function")

    logger.debug("done conditions before sorting")
    sigma = sorted(sigma.items(), key=lambda x: x[1])
    p = sorted(p.items(), key=lambda x: x[0])
    logger.debug("done function after sorting")
    return rent, (sigma, p)


def check_while(budget_graph, budget: dict, p: dict, sigma: dict):
    """
        Exist i∈N s.t. p_(σ(i)) = b_i and i is not a vertex of a cycle of Γ_b(σ, p)
        :return: if exist i, return true
    """

    @jit
    def optimize_cycle(cycle1, j):
        for c in cycle1:
            if j in c:
                return True

    for i in sigma.keys():
        if budget[i] == p[sigma[i]]:
            cycle = nx.simple_cycles(budget_graph)
            x = list(cycle)
            if x == []:
                return True
            else:
                optimize_cycle(cycle, i)
    return False


def case_1(budget: dict, p: dict, sigma: dict):
    """
     exist i∈N s.t. p_(σ(i)) = b_i
     :return: if exist i return true
    """
    for i in sigma.keys():
        if budget[i] == p[sigma[i]]:
            return True
    return False


def case_2(sigma: dict, p: dict, budget: dict, agentsList: AgentList):
    """
        For reshuffle of sigma along cycle C of budget aware graph.
        find i∈N s.t.p_(σ(i)) = b_i and i is a vertex of a cycle C of Γ_b(σ, p)
        σ ← reshuffle of σ along C
    :return: new sigma
    """
    budget_graph = build_budget_aware_graph(sigma, p, budget, agentsList)
    for i in sigma.keys():
        if p[sigma[i]] == budget[i]:
            # find cycle in graph
            try:
                cycle = nx.find_cycle(budget_graph, i)
                if cycle is not None:
                    temp = {j: sigma[j] for j in cycle}
                    # For reshuffle the rooms
                    values = list(temp.values())
                    # Find the position of the first element in temp
                    pos = values.index(temp[0])
                    # Shift the elements to the right
                    values = values[pos:] + values[:pos]
                    # Create a new shuffled dictionary using the original keys and the shuffled values
                    shuffled_temp = {key: value for key, value in zip(temp.keys(), values)}
                    # updating new room for agent
                    sigma = {j: shuffled_temp[j] for j in shuffled_temp.keys()}
                    return sigma
            except:
                print("yess")
                return sigma
    return sigma


def optimal_envy_free1(agentsList: AgentList, rent: float, budget: dict) -> (dict, dict):
    """
        This function implements Algorithm 2 from the article.
        :param agentsList: Evaluation for each room by agent
        :param rent: Total rent
        :budget: Each agent's budget
        :return:  Optimal envy-free allocation subject to budget constraints.
                 µ: N -> A , p : is vector of prices for each room
        >>> agentList1 = AgentList({'Alice': {'2ndFloor': 250, 'Basement': 250, 'MasterBedroom': 500},'Bob': {'2ndFloor': 250, 'Basement': 250, 'MasterBedroom': 500},'Clair': {'2ndFloor': 250, 'Basement': 500, 'MasterBedroom': 250}})
        >>> optimal_envy_free(agentList1, 1000, {'Alice': 200,'Bob': 100,'Clair': 200,})
        --------THE RESULT--------
        'no solution'
        >>> agentList2 = AgentList({'P1': {'Ra': 600, 'Rb': 100, 'Rc': 150},'P2': {'Ra': 250, 'Rb': 250, 'Rc': 250},'P3': {'Ra': 100, 'Rb': 400, 'Rc': 250}})
        >>> optimal_envy_free(agentList2, 1000, {'P1': 600,'P2': 400,'P3': 400})
        ([('P1', 'Ra'), ('P2', 'Rc'), ('P3', 'Rb')], [('Rc', 166.67), ('Rb', 316.67), ('Ra', 516.67)])

        >>> ex3 = AgentList({"Alice":{'1' : 250, '2' : 750}, "Bob": {'1': 250, '2' : 750}})
        >>> optimal_envy_free(ex3, 1000, {'Alice': 600, 'Bob': 500})
        --------THE RESULT--------
        'no solution'
        """
    logger.info(f'optimal_envy_free({agentsList}, {rent}, {budget})')
    # line 48-55 : Taking the AgentList type and splitting it to lists and dictionary
    N = list([i for i in agentsList.agent_names()])
    A = list([i for i in agentsList.all_items()])

    val = {i: {j: agentsList[N.index(i)].value(j) for j in A} for i in N}
    logger.debug("done initializing the first variables")
    # compute (σ,p) ∈ F(N,A,v,r)
    sigma = {}
    p = {}
    sigma, p = spliddit(agentsList, rent)  # compute (σ,p) ∈ F(N,A,v,r)
    logger.debug("done spliddit")
    # Γ(σ, p) : weak envy graph by Γ(σ,p) ≡ (N,E), where (i, j) ∈ E if and only if viσ(i) − pσ(i) = viσ(j) − pσ(j)
    # graph = Γ(σ, p)
    graph = build_weak_envy_graph(sigma, p, agentsList)
    logger.debug("done build_weak_envy_graph")
    # SCC = C , C ← strongly connected components of Γ(σ, p)
    SCC = nx.strongly_connected_components(graph)
    logger.debug("done strongly_connected_components")

    ans_µc_pc = []
    """
    for each c ∈ C do:
        (μ_c,p_c) ← output of Algorithm 1 on C, A(C), vC,and bC
    """
    lst_SCC = list(SCC)
    for c in lst_SCC:
        # N_c = let N_c be the set of agent from strongly connected components
        N_c = list([i for i in N if i in c])
        # A_c = let A(C) be the set of rooms received by agents in C
        A_c = list([sigma[i] for i in sigma.keys() if i in c])
        # let vC and B_c be the restrictions of the two vectors to C
        V_c = {i: {j: agentsList[N_c.index(i)].value(j) for j in A_c} for i in c}
        B_c = {i: budget[i] for i in sigma.keys() if i in c}
        # New rent for calculation on c
        new_rent = sum(p[sigma[i]] for i in c)
        tempAgentC = {i: V_c[i] for i in c}
        # (μC,pC) ← output of Algorithm 1 on C, A(C), V_c, and B_c
        ans_µc_pc.append(maximum_rent_envy_free1(AgentList(tempAgentC), new_rent, B_c))
        logger.debug("done appending ans_µc_pc with maximum_rent_envy_free")

    µ = {}

    # let μ:N →A s.t. for all i ∈ N , μ(i) = μ_C(i) for c ∈ C s.t. i ∈ C
    for i in agentsList.agent_names():
        for j in ans_µc_pc:
            µc = dict(j[1][0])
            if i in µc:
                µ[i] = µc[i]

    logger.debug("done comparing µ[i] = µc[i]")

    # if LP(1) for μ is feasible , ans_LP1 = (µ,p)
    ans_LP1 = LP1(µ, rent, val, budget)
    logger.debug("done LP1")
    if ans_LP1 != "no solution":
        # p ← solution of LP (1) for μ return (μ, p)
        # print(f"--------THE RESULT-------- \n µ : {ans_LP1[0]} , p : {ans_LP1[1]}")
        µ = sorted(ans_LP1[0].items(), key=lambda x: x[0])
        p = sorted(ans_LP1[1].items(), key=lambda x: x[1])
        logger.debug("done function (ans_LP1 != 'no solution')")
        return µ, p
    else:
        logger.debug("done function ('no solution')")
        print(f"--------THE RESULT--------")
        return "no solution"


def maximum_rent_envy_free1(agentsList: AgentList, rent: float, budget: dict) -> (int, dict):
    """
    This function implements Algorithm 1 from the article.
    :param agentsList: Evaluation for each room by agent
    :param rent: Total rent
    :budget: Each agent's budget
    :return: Maximum-rent envy-free allocation in a fully connected economy.
             sigma: N -> A , p : is vector of prices for each room
    >>> ex1 = AgentList({"Alice":{'1' : 250, '2' : 250, '3' : 500}, "Bob": {'1': 250, '2' : 250, '3' :500}, "Clair": {'1' :250, '2' : 500, '3' : 250}})
    >>> maximum_rent_envy_free(ex1, 1000, {'Alice': 250, 'Bob': 320, 'Clair': 430})
    (1249.99, ([('Alice', '1'), ('Clair', '2'), ('Bob', '3')], [('1', 250.0), ('2', 500.0), ('3', 500.0)]))
    >>> ex2 = AgentList({"Alice":{'1' : 250, '2' : 750}, "Bob": {'1': 250, '2' : 750}})
    >>> maximum_rent_envy_free(ex2, 1000, {'Alice': 600, 'Bob': 500})
    (1700.0, ([('Alice', '1'), ('Bob', '2')], [('1', 600.0), ('2', 1100.0)]))
    >>> ex3 = AgentList({"Alice":{'1' : 400, '2' : 600}, "Bob": {'1': 300, '2' : 700}})
    >>> maximum_rent_envy_free(ex3, 1000, {'Alice': 450, 'Bob': 550})
    (1200.0, ([('Alice', '1'), ('Bob', '2')], [('1', 450.0), ('2', 750.0)]))
    """
    logger.info(f'maximum_rent_envy_free({agentsList}, {rent}, {budget})')
    N = list([i for i in agentsList.agent_names()])
    logger.debug("done initializing the first variables")
    sigma = {}
    p = {}
    sigma, p = spliddit(agentsList, rent)
    logger.debug("done spliddit")

    # It is for calculating the ∆, let Δ ∈ R such that
    # (σ,(p_a −Δ)_(a∈A)) ∈ F_b(N,C,v,r−nΔ) and there is i∈N such that p_(σ(i)) = b_i

    delta = min([p[sigma[i]] - budget[i] for i in sigma.keys()])
    logger.debug("done calculating delta")

    # p ← (p_(σ(i)) − Δ)_(i∈N)
    p = {i: round(v - delta, 3) for i, v in p.items()}
    logger.debug("done calculating p")

    # # r ← r − nΔ
    listlen = len(agentsList)

    @jit
    def optimize_rent(n: int, rentt: float, delta1):
        rentt -= n * delta1

    optimize_rent(listlen, rent, delta)
    logger.debug("done calculating rent")

    # budget_graph
    # ---> Γ_b(σ,p) ≡ (N,E), where E = {(i, j) : v_(iσ(i)) − p_(σ(i)) = v_(iσ(j)) −p_(σ(i)) and p_(σ(j)) < b_i}
    budget_graph = build_budget_aware_graph(sigma, p, budget, agentsList)
    logger.debug("done build_budget_aware_graph")
    while not check_while1(budget_graph, budget, p, sigma):
        # Case1
        if not case_1(budget, p, sigma):
            logger.debug("entering case_1")
            # Δ ← min i∈N (b_i − p_(σ(i)))
            delta = min([budget[i] - p[sigma[i]] for i in sigma.keys()])
            logger.debug("done min delta")
            # p ← (p_a + Δ)_(a∈A)
            p = {i: v + delta for i, v in p.items()}
            logger.debug("done calculating new p")
            # r ← r + nΔ
            rent = rent + len(N) * delta
            logger.debug("done calculating new rent")
        # Case2
        else:
            logger.debug("entering case_2")
            sigma = case_2(sigma, p, budget, agentsList)
            logger.debug("done case_2 function")
        # Update the new budget-aware graph
        budget_graph = build_budget_aware_graph(sigma, p, budget, agentsList)
        logger.debug("done build_budget_aware_graph function")

    logger.debug("done conditions before sorting")
    sigma = sorted(sigma.items(), key=lambda x: x[1])
    p = sorted(p.items(), key=lambda x: x[0])
    logger.debug("done function after sorting")
    return rent, (sigma, p)


def check_while1(budget_graph, budget: dict, p: dict, sigma: dict):
    """
        Exist i∈N s.t. p_(σ(i)) = b_i and i is not a vertex of a cycle of Γ_b(σ, p)
        :return: if exist i, return true
    """
    for i in sigma.keys():
        if budget[i] == p[sigma[i]]:
            cycle = nx.simple_cycles(budget_graph)
            x = list(cycle)
            if x == []:
                return True
            else:
                for c in cycle:
                    if i in c:
                        return True
    return False


if __name__ == '__main__':
    doctest.testmod()
    agentList1 = AgentList({'Alice': {'2ndFloor': 250, 'Basement': 250, 'MasterBedroom': 500},
                            'Bob': {'2ndFloor': 250, 'Basement': 250, 'MasterBedroom': 500},
                            'Clair': {'2ndFloor': 250, 'Basement': 500, 'MasterBedroom': 250}})

    budget1 = {
        'Alice': 200,
        'Bob': 100,
        'Clair': 200,
    }
    rent1 = 1000

    agentList2 = AgentList({
        'P1': {'Ra': 600, 'Rb': 100, 'Rc': 150, 'Rd': 150},
        'P2': {'Ra': 250, 'Rb': 250, 'Rc': 250, 'Rd': 250},
        'P3': {'Ra': 100, 'Rb': 400, 'Rc': 250, 'Rd': 250},
        'P4': {'Ra': 100, 'Rb': 200, 'Rc': 350, 'Rd': 350}
    })
    budget2 = {
        'P1': 600,
        'P2': 400,
        'P3': 400,
        'P4': 300
    }
    rent2 = 1000

    ex8 = AgentList(
        {"a": {'1': 100, '2': 200, '3': 300, '4': 400, '5': 500, '6': 600, '7': 700, '8': 800, '9': 900,
               '10': 1000},
         "b": {'1': 1000, '2': 900, '3': 800, '4': 700, '5': 600, '6': 500, '7': 400, '8': 300, '9': 200,
               '10': 100},
         "c": {'1': 100, '2': 1000, '3': 200, '4': 300, '5': 400, '6': 500, '7': 600, '8': 700, '9': 800,
               '10': 900},
         "d": {'1': 200, '2': 100, '3': 1000, '4': 400, '5': 300, '6': 500, '7': 600, '8': 700, '9': 800,
               '10': 900},
         "e": {'1': 300, '2': 200, '3': 400, '4': 1000, '5': 500, '6': 600, '7': 700, '8': 800, '9': 900,
               '10': 100},
         "f": {'1': 400, '2': 300, '3': 200, '4': 100, '5': 1000, '6': 800, '7': 600, '8': 500, '9': 700,
               '10': 900},
         "g": {'1': 500, '2': 900, '3': 800, '4': 700, '5': 100, '6': 1000, '7': 200, '8': 300, '9': 600,
               '10': 400},
         "h": {'1': 600, '2': 200, '3': 300, '4': 400, '5': 500, '6': 100, '7': 1000, '8': 800, '9': 700,
               '10': 900},
         "i": {'1': 700, '2': 900, '3': 800, '4': 200, '5': 300, '6': 400, '7': 100, '8': 1000, '9': 500,
               '10': 600},
         "j": {'1': 800, '2': 900, '3': 100, '4': 200, '5': 300, '6': 400, '7': 500, '8': 600, '9': 1000,
               '10': 700},
         })

    budgetex8 = {
        'a': 1000,
        'b': 1000,
        'c': 1000,
        'd': 1000,
        'e': 1000,
        'f': 1000,
        'g': 1000,
        'h': 1000,
        'i': 1000,
        'j': 1000
    }
    optimal_envy_free(ex8, 5500, budgetex8)

    num_agents = 10
    num_items = 10
    agents = {f"Agent{i}": {f"Item{j}": (i + j) * 10 for j in range(num_items)} for i in range(num_agents)}
    rent = num_agents * num_items * 10
    budgets = {f"Agent{i}": (i + 1) * 100 for i in range(num_agents)}
    ex10 = AgentList(agents)

    print("running just for start")
    print(optimal_envy_free(agentList1, rent1, budget1))

    print("solution agentList1")
    start_time = time.perf_counter()
    print(optimal_envy_free1(ex10, rent, budgets))
    end_time = time.perf_counter()
    print(f'Execution time without numba: {end_time - start_time:0.6f} seconds')
    start_time = time.perf_counter()
    print(optimal_envy_free(ex10, rent, budgets))
    end_time = time.perf_counter()
    print(f'Execution time with numba: {end_time - start_time:0.6f} seconds')
    print()
    print("solution agentList2")
    print(optimal_envy_free(agentList2, rent2, budget2))

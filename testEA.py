from EA import myEA
from spoc_delivery_scheduling_evaluate_code import trappist_schedule


if __name__ == "__main__":
    udp = trappist_schedule()
    your_decision_vector = myEA.main()
    fitness_values = udp.fitness(your_decision_vector)
    print(fitness_values)

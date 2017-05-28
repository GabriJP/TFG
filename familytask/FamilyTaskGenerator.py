from random import randint, sample, shuffle

bag_of_names = [
    "Benjamin",
    "Vicente",
    "Martin",
    "Matias",
    "Joaquin",
    "Agustin",
    "Cristobal",
    "Maximiliano",
    "Sebastian",
    "Tomas",
    "Diego",
    "Jose",
    "Nicolas",
    "Felipe",
    "Lucas",
    "Alonso",
    "Bastian",
    "Juan",
    "Gabriel",
    "Ignacio",
    "Francisco",
    "Renato",
    "Maximo",
    "Mateo",
    "Javier",
    "Daniel",
    "Luis",
    "Gaspar",
    "Angel",
    "Fernando",
    "Carlos",
    "Emilio",
    "Franco",
    "Cristian",
    "Pablo",
    "Santiago",
    "Esteban",
    "David",
    "Damian",
    "Jorge",
    "Camilo",
    "Alexander",
    "Rodrigo",
    "Amaro",
    "Luciano",
    "Bruno",
    "Alexis",
    "Victor",
    "Thomas",
    "Julian"
]

"""
Schema for this family:
                0
              /   \
            1       2
          /   \     |
        3       4   5

"""
family_nodes = {0, 1, 2, 3, 4, 5}
family_relations = {
    (0, 1): "padre",
    (0, 2): "padre",
    (0, 3): "abuelo",
    (0, 4): "abuelo",
    (0, 5): "abuelo",
    (1, 0): "hijo",
    (1, 2): "hermano",
    (1, 3): "padre",
    (1, 4): "padre",
    (1, 5): "tío",
    (2, 0): "hijo",
    (2, 1): "hermano",
    (2, 3): "tío",
    (2, 4): "tío",
    (2, 5): "padre",
    (3, 0): "nieto",
    (3, 1): "hijo",
    (3, 2): "sobrino",
    (3, 4): "hermano",
    (3, 5): "primo",
    (4, 0): "nieto",
    (4, 1): "hijo",
    (4, 2): "sobrino",
    (4, 3): "hermano",
    (4, 5): "primo",
    (5, 0): "nieto",
    (5, 1): "sobrino",
    (5, 2): "hijo",
    (5, 3): "primo",
    (5, 4): "primo",
}

n_histories = 10000
history_length = (7, 13)
n_hops = (1, 4)

for hops in range(*n_hops):
    for filename in ("qa{}_train.txt", "qa{}_test.txt"):
        with open(filename.format(hops), "w") as file:
            for historia in range(n_histories):
                nombre_a_usar = sample(bag_of_names, 6)
                elementos_a_usar = sample(family_nodes, hops + 1)
                pregunta = "¿Qué es %s de %s?" % (nombre_a_usar[elementos_a_usar[-1]], nombre_a_usar[elementos_a_usar[0]])
                respuesta = family_relations[elementos_a_usar[-1], (elementos_a_usar[0])]

                hechos = []

                for hecho in range(hops):
                    hechos.append("%s es %s de %s\n" %
                                  (
                                      nombre_a_usar[elementos_a_usar[hecho]],
                                      family_relations[(elementos_a_usar[hecho], elementos_a_usar[hecho + 1])],
                                      nombre_a_usar[elementos_a_usar[hecho + 1]])
                                  )

                resto_elementos = family_nodes.difference(elementos_a_usar)
                for hecho in range(randint(*history_length) - hops):
                    elementos_a_usar = sample(resto_elementos, 2)
                    hechos.append("%s es %s de %s\n" %
                                  (
                                      nombre_a_usar[elementos_a_usar[0]],
                                      family_relations[(elementos_a_usar[0], elementos_a_usar[1])],
                                      nombre_a_usar[elementos_a_usar[1]])
                                  )

                shuffle(hechos)

                hechos = ["%d %s" % (idx + 1, hecho) for idx, hecho in enumerate(hechos + ["%s\t%s\n" % (pregunta, respuesta)])]

                file.writelines(hechos)

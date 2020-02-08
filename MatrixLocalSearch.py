import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import copy

# creation of the initial image
operator_dim = 2
image_dim = operator_dim * 200
min_color = 50
max_color = 250
random_colored_image = np.array(min_color + (max_color - min_color) * np.random.random((image_dim, image_dim, 3)),
                                dtype='uint8')

R = random_colored_image[:, :, 0]
G = random_colored_image[:, :, 1]
B = random_colored_image[:, :, 2]

R[R <= G] = 0
G[G <= R] = 0
B[B <= G] = 0
R[R <= B] = 0
G[G <= B] = 0
B[B <= R] = 0

print_colors = False
if print_colors:
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(R, 'Reds')
    plt.subplot(1, 3, 2)
    plt.imshow(G, 'Greens')
    plt.subplot(1, 3, 3)
    plt.imshow(B, 'Blues')
    plt.show()


# function to optimize

# the variation of color between neighbours


# division of the image in the 3 colors
def fitness(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # kernels
    kernels = []
    # forward mean
    #     kernels.append(np.array([[-1,1],
    #                        [-2,2],
    #                        [-1,1]])/4)
    #     kernels.append(np.array([[-1,-2,-1],
    #                        [1,2,1]])/4)
    # central mean
    #     kernels.append(np.array([[-1,0,1],
    #                        [-2,0,2],
    #                        [-1,0,1]])/4)
    #     kernels.append(np.array([[-1,-2,-1],
    #                        [0,0,0],
    #                        [1,2,1]])/4)
    # forward
    kernels.append(np.array([[-1, 1]]))
    kernels.append(np.array([[-1],
                             [1]]))

    def slider(operator, image):
        i = 0
        value = 0
        i_operator, j_operator = operator.shape
        image_x, image_y = image.shape[0:2]
        for i in range(image_x - i_operator + 1):
            for j in range(image_y - j_operator + 1):
                tmp_part_of_image = R[i:i + i_operator, j:j + j_operator]
                tmp_result = abs(np.sum(tmp_part_of_image * operator))
                value += tmp_result
                tmp_part_of_image = G[i:i + i_operator, j:j + j_operator]
                tmp_result = abs(np.sum(tmp_part_of_image * operator))
                value += tmp_result
                tmp_part_of_image = B[i:i + i_operator, j:j + j_operator]
                tmp_result = abs(np.sum(tmp_part_of_image * operator))
                value += tmp_result
                j += 1
            i += 1
        return value

    fitness_value = 0
    for kernel in kernels:
        fitness_value += slider(kernel, image)
    return fitness_value


# utilities
def saturation(value, high, low):
    if value > high:
        return high
    if value < low:
        return low
    return value


# creation of neighbourhood

# permutation
def image_permutation(permutation):
    return np.random.permutation(permutation)


# k-exchange
def k_exchange(k, image_dim):
    movements = set()
    for i in range(k):
        coordinates_to_exchange = np.array(image_dim * np.random.random((2, 2)), dtype=int)
        movements.add((coordinates_to_exchange[0, 0], coordinates_to_exchange[0, 1], coordinates_to_exchange[1, 0],
                       coordinates_to_exchange[1, 1]))
    return movements


# manhattan distance movement
def manhattan_distance_movement(matrix_dim, length_movement=1):
    movements = set()
    for i in range(matrix_dim - length_movement):
        for j in range(matrix_dim - length_movement):
            for k in range(1, length_movement + 1):
                movements.add((i, j, i, j + k))
                movements.add((i, j, i + k, j))
    return movements


# exchange of two pixels with difference in fitness(based on operator size(2))
def exchange(image, coordinates):
    x1, y1, x2, y2 = coordinates
    low = 0
    high = image.shape[0]
    # (3,3) slices

    old_fitness = fitness(image[saturation(x1 - 1, high, low):saturation(x1 + 2, high, low),
                          saturation(y1 - 1, high, low):saturation(y1 + 2, high, low)]) + fitness(
        image[saturation(x2 - 1, high, low):saturation(x2 + 2, high, low),
        saturation(y2 - 1, high, low):saturation(y2 + 2, high, low)])
    a = copy.copy(image[x1, y1])
    image[x1, y1] = image[x2, y2]
    image[x2, y2] = a
    new_fitness = fitness(image[saturation(x1 - 1, high, low):saturation(x1 + 2, high, low),
                          saturation(y1 - 1, high, low):saturation(y1 + 2, high, low)]) + fitness(
        image[saturation(x2 - 1, high, low):saturation(x2 + 2, high, low),
        saturation(y2 - 1, high, low):saturation(y2 + 2, high, low)])
    return new_fitness - old_fitness


# hill climbing for k_exchange
def hill_climbing_k_exchange(image, max_iterations, good_value, neighbour_dimension, k):

    video = [Image.fromarray(image)]
    i = 0
    h = 0
    stop = False
    best = fitness(image)
    best_image = image
    while best >= good_value and i < max_iterations and not stop:
        for j in range(neighbour_dimension):
            stop = True
            image = np.copy(best_image)
            moves = k_exchange(k, image_dim)
            dfitness = 0
            for move in moves:
                dfitness += exchange(image, move)
            candidate = best + dfitness
            i += 1
            if not i % (max_iterations / 10):
                print(10 - i / max_iterations * 10, sep=' ')
            if candidate < best:
                h += 1
                best = candidate
                best_image = np.copy(image)
                if h % int(max_iterations/2000) == 0:
                    video.append(Image.fromarray(best_image))
                stop = False
                break
    if best < good_value:
        print('better than good value')
    elif i >= max_iterations:
        print('max iteration')
    else:
        print('local minimum')
    print('')
    print(best)
    return best_image, video, best


# hill climbing for manhattan distance
def hill_climbing_manhattan(max_iterations, good_value, distance, image):
    i = 0
    stop = False
    best = fitness(image)
    best_image = image
    moves = manhattan_distance_movement(image.shape[0], distance)
    while best >= good_value and i < max_iterations and not stop:
        for move in moves:
            stop = True
            image = np.copy(best_image)
            exchange(image, move)
            candidate = fitness(image)
            i += 1
            if not i % (max_iterations / 10):
                print(10 - i / max_iterations * 10, sep=' ')
            if candidate < best:
                best = candidate
                best_image = np.copy(image)
                stop = False
                break

    if best < good_value:
        print('better than good value')
    elif i >= max_iterations:
        print('max iteration')
    else:
        print('local minimum')

    return best_image


def reduce_array(array, length):
    if len(array) <= length:
        return array
    division = int(len(array)/length)
    i = 0
    for a in array:
        if i % division:
            array.remove(a)
        i += 1
    return array


best_image, video, best = hill_climbing_k_exchange(random_colored_image,
                                                   max_iterations=5000000,
                                                   good_value=100,
                                                   neighbour_dimension=500000,
                                                   k=2)
video = reduce_array(video, 600)
print(len(video))
video[0].save('video.gif', save_all=True, append_images=video[1:], optimize=False, duration=50)

plt.figure(figsize=(20, 20))
plt.imshow(best_image)
plt.show()
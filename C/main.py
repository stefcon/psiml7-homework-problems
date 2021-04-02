import numpy as np


def calc_probability(p, num_col):
    prob = 0
    if p == 0:
        # if the ball hits the wall, it certainly disappears
        return num_col.count(0)
    for col in num_col:
        prob += p**col
    return prob


def count_collisions(point_vector, n, k, t, s):
    """
    As the name says, count the number of wall collisions of n ball in t seconds in the box of dimensions
    2s x 2s. To avoid division by zero, vx and vy should be checked if they are zero.
    Function basically counts the collisions for vx and vy separately, as all collisions are elastic
    """
    num_of_collisions = 0
    point_col = []
    for i in range(n):
        curr_col = 0
        time_x, time_y = t, t
        x, y, vx, vy = point_vector[i]
        if vx != 0:
            if vx > 0:
                line_len = s - x
                time_x -= line_len / abs(vx)
            else:
                line_len = s + x
                time_x -= line_len / abs(vx)
            if time_x > 0:
                curr_col += 1 + (abs(vx) * time_x) // (2 * s)

        if vy != 0:
            if vy > 0:
                line_len = s - y
                time_y -= line_len / abs(vy)
            else:
                line_len = s + y
                time_y -= line_len / abs(vy)
            if time_y > 0:
                curr_col += 1 + (abs(vy) * time_y) // (2 * s)

        point_col.append(curr_col)
        num_of_collisions += curr_col

    return num_of_collisions, point_col


def calculate_the_beginning(point_vector, n):
    """
    Calculates "the beginning of time" by getting the average of time it takes for all
    points to travel back to their origin and rounding it up
    """
    avg_time = 0
    for i in range(n):
        s = np.sqrt(point_vector[i, 0] ** 2 + point_vector[i, 1] ** 2)
        v = np.sqrt(point_vector[i, 2] ** 2 + point_vector[i, 3] ** 2)
        avg_time += s / v

    avg_time /= n

    return round(avg_time)


def input_points():
    """
    Function for reading the problem format
    """
    line = input()
    line = line.split()
    n = int(line[0])
    s = int(line[1])
    t = int(line[2])
    p = float(line[3])
    # point_vector columns: Px Py Vx Vy
    point_vector = np.zeros((n, 4))
    for i in range(n):
        line = input()
        line = line.split()
        point_vector[i, 0] = float(line[0])
        point_vector[i, 1] = float(line[1])
        point_vector[i, 2] = float(line[2])
        point_vector[i, 3] = float(line[3])

    return n, s, t, p, point_vector


def main():
    n, s, t, p, point_vector = input_points()

    k = calculate_the_beginning(point_vector, n)
    print(k, end=" ")
    num_of_collisions, point_col = count_collisions(point_vector, n, k, t, s)
    print(num_of_collisions, end=" ")
    prob = calc_probability(p, point_col)
    print(prob)


if __name__ == "__main__":
    main()

import sys
import random

if __name__ ==  "__main__":

    input_file = sys.argv[1]
    N = 100
    n_walls = 3000
    v_max = 10
    random.seed(42)

    with open(input_file, 'w') as f:
        f.write('%i %i %i\n' % (N, n_walls, v_max))
        f.write('%i %i 50 50\n' % (N-1, N-1))
        for t in range(n_walls):
            x = random.randint(1, N-2)
            y = random.randint(1, N-2)
            # k = random.randint(0, 4)
            k = random.randint(0, 3)

            if k == 0:
                x1 = x - 1
                y1 = y

            if k == 1:
                x1 = x + 1
                y1 = y

            if k == 2:
                x1 = x
                y1 = y - 1

            if k == 3:
                x1 = x
                y1 = y + 1

            f.write('%i %i %i %i\n' % (x, y, x1, y1))


#%%
from queue import Queue
from bisect import bisect_left, bisect_right
from functools import partial
from math import sqrt
import heapq
from PIL import Image, ImageDraw
import time
import sys

#%%
class Environment:
    directions = ('N', 'E', 'S', 'W')
    
    def is_inside(self, x, y):
        return (0 <= x) and (x < self.N) and (0 <= y) and (y < self.N)
    
    def _line2ints(self, line):
        return [int(n) for n in line.split()]

    def _is_hozirontal_wall(self, wall):
        if not(wall[1] == wall[3] or wall[0] == wall[2]):
            raise ValueError # đánh dấu dòng bị lỗi 
        return wall[1] == wall[3] # có y0 = y1

    def read_map(self, input_file):
        with open(input_file, 'r') as f:
            lines = [self._line2ints(line) for line in f.readlines()]
            (self.N, self.n_walls, self.vmax) = lines[0]
            xs, ys, xg, yg = lines[1]
            
            # Map image
            self.image = Image.new('RGB', (600, 600), color='white')
            self.scale_factor = 600 / (self.N - 0.5)
            draw = ImageDraw.Draw(self.image)
            
            self.start = (xs, ys)
            self.goal = (xg, yg)
            self.v_walls = []
            self.h_walls = []
            for wall in lines[2: 2 + self.n_walls]:
                x0, y0, x1, y1 = wall
                try:
                    if self._is_hozirontal_wall(wall):
                        self.h_walls.append(tuple([*sorted((x0, x1)), y0]))
                    else:
                        self.v_walls.append(tuple([*sorted((y0, y1)), x0]))
                    # Draw wall to map image
                    draw.line([(x0 * self.scale_factor, y0 * self.scale_factor), 
                                (x1 * self.scale_factor, y1 * self.scale_factor)], 
                                fill="red", width=int(max(self.scale_factor / 2 , 1)))
                except ValueError:
                    print('Error wall:', wall)
            self.v_walls = sorted(self.v_walls, key= lambda x: x[-1])
            self.h_walls = sorted(self.h_walls, key= lambda x: x[-1])
            
            
    # Check colision
    def _is_collided(self, walls, fix, s, e):
        """
        s < e
        """
        tmp = [p[-1] for p in walls]
        s_ind = bisect_left(tmp, s) # any bisect is fine
        if s < e:
            end_ind = bisect_right(tmp, e) # bound == case
        else:
            end_ind = bisect_left(tmp, e) # bound == case
        s_ind, end_ind = sorted((s_ind, end_ind))
        for w in walls[s_ind:end_ind + 1]:
            if w[0] <= fix <= w[1]:
                return True
        return False
        
    # Chỉ kiểm tra các tường ngang dọc tuỳ theo hướng di chuyển 
    def _next_position(self, x, y, d, v):
        """
        :param x0:
        :param y0:
        :param d: moving direction, ['N', 'S', 'E', 'W']
        :param v: speed
        :param list_of_walls: list of walls has to check collision
        :return:
            x1, y1:
            if invalid move (collision) return -1, -1
        """
        if v == 0:
            return x, y
        
        if d % 2 == 0: # verical direction
            v = (1 - d)*v
            if self.is_inside(x, y + v) and not self._is_collided(self.h_walls, x, y, y+v):
                return x, y+v
        else: # hozi direction
            v = (2 - d)*v
            if self.is_inside(x + v, y) and not self._is_collided(self.v_walls, y, x, x+v):
                return x + v, y
        return -1, -1
    
    # Actions
    # Trả về vị trí, d, v tương ứng với action
    # return x, y, d, v, cost of action
    def turn_left(self, x, y, d, v):
        d += 1
        d %= 4
        x, y = self._next_position(x, y, d, v)
        return x, y, d, v, 2 + int(sqrt(v))
    
    def turn_right(self, x, y, d, v):
        d -= 1
        d %= 4
        x, y = self._next_position(x, y, d, v)
        return x, y, d, v, 2 + int(sqrt(v))
    
    def speed_up(self, x, y, d, v):
        v = min(v + 1, self.vmax)
        x, y = self._next_position(x, y, d, v)
        return x, y, d, v,  2 + int(sqrt(v))
    
    def slow_down(self, x, y, d, v):
        v = max(v - 1, 0)
        x, y = self._next_position(x, y, d, v)
        return x, y, d, v,  2 + int(sqrt(v))

    def no_action(self, x, y, d, v):
        x, y = self._next_position(x, y, d, v)
        return x, y, d, v,  1 + int(sqrt(v))


class Solver:
    def __init__(self,env: Environment) -> None:
        self.env = env
        self.found = None

    def act_and_add_state(self, act, prev_state, cur_cost, is_ucf=False):
        x, y, d, v, step_cost = act(*prev_state)
        if cur_cost == -1:
            prev_state = None # Init state
        if x < 0 or y < 0: # Invalid action
            return None
        cur_cost += step_cost if is_ucf else 1
        s = (x, y, d, v)
        # Fast stop
        if self.env.goal == (x, y) and v == 0:
            self.path[s] = prev_state
            self.found = cur_cost, s
            return None
        
        if s in self.seen:
            return None

        if is_ucf:
            # Not sure min path to s
            prev_min_cost = self.mins.get(s, None)
            if prev_min_cost is None or prev_min_cost > cur_cost:
                self.mins[s] = cur_cost
                self.path[s] = prev_state
                return *s, cur_cost
            else:
                return None
        else:
            # Sure min path to s
            p = self.path.get(s, None)
            if p is not None: 
                return None
            self.path[s] = prev_state
        return x, y, d, v, cur_cost

    def explore(self, s, cost, is_ucf):
        if s in self.seen:
            return -1, None
        self.seen.add(s)

        if self.found is not None:  # Fast stop
            return self.found
        # print("step:", num_step, s)
        step_fn = self.step_ucf if is_ucf else self.step_bfs
        # Action
        v = s[-1]
        if v == 0:
            step_fn(self.env.turn_left, s, cost)
            step_fn(self.env.turn_right, s, cost)
            step_fn(self.env.speed_up, s, cost)
        else:
            step_fn(self.env.no_action, s, cost)
            step_fn(self.env.slow_down, s, cost)
            if v < self.env.vmax:
                step_fn(self.env.speed_up, s, cost)
        return -1, None
    
    def step_bfs(self, act, s, total_cost):
        ret = self.act_and_add_state(act, s, total_cost, is_ucf=False)
        if ret is None:
            return # Already explored or invalid action
        x, y, d, v, cur_cost = ret
        new_s = (x, y, d, v)
        self.Q.put(partial(self.explore, new_s, cur_cost, False))

    def step_ucf(self, act, s, total_cost):
        ret = self.act_and_add_state(act, s, total_cost, is_ucf=True)
        if ret is None:
            return # Already explored or invalid action
        x, y, d, v, cur_cost = ret
        new_s = (x, y, d, v)
        heapq.heappush(self.Q, (cur_cost, new_s, partial(self.explore, new_s, cur_cost, True)))

    def solve_bfs(self):
        self.Q = Queue()
        self.seen = set()
        self.path = dict()
        # self.mins = dict()
        x0, y0 = self.env.start
        self.step_bfs(self.env.no_action, (x0, y0, 0, 0), -1)
        ans = -1
        last_state = None
        while ans == -1 and not self.Q.empty():
            ans, last_state = self.Q.get()()
        return ans, last_state
    
    def solve_ucf(self):
        self.Q = []
        self.seen = set()
        self.path = dict()
        self.mins = dict()
        x0, y0 = self.env.start
        self.step_ucf(self.env.no_action,(x0, y0, 0, 0), -1)
        ans = -1
        last_state = None
        while ans == -1 and len(self.Q) > 0:
            ans, last_state = heapq.heappop(self.Q)[-1]()
        return ans, last_state
#%%

def trace_back(solver, x, y, d, v, image):
    path = [(x, y, d, v)]
    draw = ImageDraw.Draw(image)
    scale_factor = solver.env.scale_factor
    prev_state = solver.path[path[-1]]
    while prev_state != None:
        path.append(prev_state)
        draw.line([(x * scale_factor, y * scale_factor), (prev_state[0] * scale_factor, prev_state[1] * scale_factor)], fill='black', width=int(max(scale_factor / 2 , 1)))
        x, y, d, v = prev_state
        prev_state = solver.path[prev_state]
    return path

#%%
if __name__ ==  "__main__":
    method, inputmap = sys.argv[1:3]
    method = method.lower()
    # method = 'ucf'
    assert method in ('ucf', 'bfs')
    # inputmap='car_large'
    env = Environment()
    env.read_map(inputmap)
    env.image.show()
    env.image.save('images/env_' + inputmap.split('/')[-1].split('.')[0] + '.png')
    solver = Solver(env)

    print("solving ...")
    start = time.time()
    ans, last_state = solver.solve_ucf() if method == 'ucf' else solver.solve_bfs()
    print('Min cost', ans, 'Done in', time.time() - start)
    print('Tracing...')
    path = trace_back(solver, *last_state, env.image)
    print('Num step:', len(path) - 1)
    print('Found path:')
    for s in reversed(path):
        print('->',s)
    print('Showing map and path') 
    env.image.show()
    env.image.save(f'images/{method}_' + inputmap.split('/')[-1].split('.')[0] + '.png')
# %%

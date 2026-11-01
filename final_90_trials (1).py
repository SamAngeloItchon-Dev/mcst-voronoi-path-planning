"""
final_90_trials.py
Runs the final 90 trials (10 per projectile condition) with the safety_heavy
weight combination and saves final_results.csv NEXT TO THIS SCRIPT.

Just press Run. No settings or arguments needed.

  Weights (goal, risk, ttc, smooth, backtrack) = (1, 50, 50, 0.5, 1)
  Conditions: projectile count 10/20/30 x projectile speed 1/2/3
  10 trials per condition = 90 trials
  Seeds 1001-1010 (fresh seeds, not the tuning seeds 1-50)

The simulation logic is identical to the original test (2).py.
"""
import os
import csv
import time

import numpy as np
from scipy.spatial import Voronoi

# ----------------------------------------------------------------------------
# SELECTED WEIGHTS  (goal, risk, ttc, smooth, backtrack)
# ----------------------------------------------------------------------------
COMBO_NAME = "safety_heavy"
WEIGHTS = (1.0, 50, 50, 0.5, 1.0)
TRIALS_PER_CONDITION = 10
SEED_START = 1001

# ----------------------------------------------------------------------------
# CONFIG (same values as the original script)
# ----------------------------------------------------------------------------
NUM_POINTS = 40
MAP_SIZE = 100
ROBOT_SPEED = 1.0
COLLISION_THRESHOLD = 2.0
PREDICTION_FACTOR = 0.0
ANGLE_OFFSET_DEG = 2
MAX_STEPS = 2000

PROJECTILE_COUNTS = [10, 20, 30]
PROJECTILE_SPEEDS = [1, 2, 3]

# ----------------------------------------------------------------------------
# Original functions (unchanged logic)
# ----------------------------------------------------------------------------
def build_graph(vor):
    graph = {}
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:
            v1, v2 = ridge
            if all(0 <= vor.vertices[v][0] <= MAP_SIZE and 0 <= vor.vertices[v][1] <= MAP_SIZE for v in (v1, v2)):
                graph.setdefault(v1, []).append(v2)
                graph.setdefault(v2, []).append(v1)
    return graph


def closest_vertex(point, vertices):
    valid_indices = [i for i, v in enumerate(vertices) if 0 <= v[0] <= MAP_SIZE and 0 <= v[1] <= MAP_SIZE]
    dists = [np.linalg.norm(vertices[i] - point) for i in valid_indices]
    return valid_indices[np.argmin(dists)]


def distance_point_to_segment(p, a, b):
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0, min(1, t))
    closest_point = a + t * ab
    return np.linalg.norm(p - closest_point)


def project_point_to_segment(p, a, b):
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0, min(1, t))
    return a + t * ab


def rotate_vector(vec, angle_deg):
    angle_rad = np.radians(angle_deg)
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                           [np.sin(angle_rad), np.cos(angle_rad)]])
    return np.dot(rot_matrix, vec)


def safe_greedy_path(graph, vertices, start_v, goal_v, projectiles, weights, projectile_speed):
    w_goal, w_risk, w_ttc, w_smooth, w_backtrack = weights
    path = [start_v]
    visited = set([start_v])
    current = start_v
    current_goal_dist = np.linalg.norm(vertices[current] - vertices[goal_v])

    while current != goal_v:
        neighbors = [v for v in graph.get(current, []) if v not in visited]
        if not neighbors:
            break

        scored_neighbors = []
        for v in neighbors:
            point = vertices[v]

            # S - path smoothness
            goal_vector = vertices[goal_v] - vertices[current]
            next_vector = point - vertices[current]
            S = 0
            if np.linalg.norm(goal_vector) > 0 and np.linalg.norm(next_vector) > 0:
                goal_vector = goal_vector / np.linalg.norm(goal_vector)
                next_vector = next_vector / np.linalg.norm(next_vector)
                cos_theta = np.clip(np.dot(goal_vector, next_vector), -1, 1)
                S = np.arccos(cos_theta)

            # D - distance to goal
            D = np.linalg.norm(point - vertices[goal_v])

            # R - projectile risk
            R = 0
            for proj in projectiles:
                dist_to_edge = distance_point_to_segment(proj, vertices[current], point)
                R += 1 / (dist_to_edge + 1)

            # T - time-to-collision
            T = 0
            for proj in projectiles:
                closest = project_point_to_segment(proj, vertices[current], point)
                dist = np.linalg.norm(proj - closest)
                ttc = dist / (projectile_speed + 1e-6)
                T += 1 / (ttc + 1)

            # B - backtracking
            B = max(0, D - current_goal_dist)

            total_cost = (w_goal * D + w_risk * R + w_ttc * T +
                          w_smooth * S + w_backtrack * B)
            scored_neighbors.append((v, total_cost))

        next_v = min(scored_neighbors, key=lambda x: x[1])[0]
        path.append(next_v)
        visited.add(next_v)
        current_goal_dist = np.linalg.norm(vertices[next_v] - vertices[goal_v])
        current = next_v
    return path


def spawn_guaranteed_projectiles(robot_pos, goal, num_projectiles):
    projectiles = []
    dirs = []
    spawn_positions = []

    for _ in range(num_projectiles):
        side = np.random.randint(4)
        if side == 0:
            pos = np.array([np.random.uniform(0, MAP_SIZE), MAP_SIZE])
        elif side == 1:
            pos = np.array([np.random.uniform(0, MAP_SIZE), 0])
        elif side == 2:
            pos = np.array([0, np.random.uniform(0, MAP_SIZE)])
        else:
            pos = np.array([MAP_SIZE, np.random.uniform(0, MAP_SIZE)])
        spawn_positions.append(pos)

    for pos in spawn_positions:
        predicted_point = robot_pos + PREDICTION_FACTOR * (goal - robot_pos)
        direction = predicted_point - pos
        direction /= np.linalg.norm(direction)
        angle_offset = np.random.uniform(-ANGLE_OFFSET_DEG, ANGLE_OFFSET_DEG)
        direction = rotate_vector(direction, angle_offset)
        direction /= np.linalg.norm(direction)
        projectiles.append(pos)
        dirs.append(direction)

    return np.array(projectiles), np.array(dirs)


# ----------------------------------------------------------------------------
# One trial (same loop as the original, minus plotting)
# ----------------------------------------------------------------------------
def run_trial(job):
    combo_name, weights, num_projectiles, projectile_speed, seed = job
    np.random.seed(seed)  # same seed -> same environment for every weight combo

    points = np.random.rand(NUM_POINTS, 2) * MAP_SIZE
    while True:
        start = np.random.rand(2) * MAP_SIZE
        goal = np.random.rand(2) * MAP_SIZE
        if np.linalg.norm(start - goal) > MAP_SIZE / 3:
            break

    robot_pos = start.copy()
    projectiles, proj_dirs = spawn_guaranteed_projectiles(robot_pos, goal, num_projectiles)

    steps = 0
    path_trace = []
    collision = False
    current_edge = None
    edge_target = None
    outcome = None

    try:
        while np.linalg.norm(robot_pos - goal) > 2 and steps < MAX_STEPS:
            steps += 1

            projectiles += proj_dirs * projectile_speed
            in_bounds = ((projectiles[:, 0] >= 0) & (projectiles[:, 0] <= MAP_SIZE) &
                         (projectiles[:, 1] >= 0) & (projectiles[:, 1] <= MAP_SIZE))
            projectiles = projectiles[in_bounds]
            proj_dirs = proj_dirs[in_bounds]

            for proj in projectiles:
                if np.linalg.norm(robot_pos - proj) <= COLLISION_THRESHOLD:
                    collision = True
                    break
            if collision:
                break

            if current_edge is None or np.allclose(robot_pos, edge_target, atol=0.5):
                dynamic_points = np.vstack((points, projectiles)) if len(projectiles) > 0 else points
                vor = Voronoi(dynamic_points)
                graph = build_graph(vor)
                robot_v = closest_vertex(robot_pos, vor.vertices)
                goal_v = closest_vertex(goal, vor.vertices)
                path_vertices = safe_greedy_path(graph, vor.vertices, robot_v, goal_v,
                                                 projectiles, weights, projectile_speed)
                if len(path_vertices) < 2:
                    edge_target = goal
                    current_edge = (robot_pos, goal)
                else:
                    v_start = vor.vertices[path_vertices[0]]
                    v_next = vor.vertices[path_vertices[1]]
                    robot_pos = project_point_to_segment(robot_pos, v_start, v_next)
                    current_edge = (v_start, v_next)
                    edge_target = v_next

            direction = edge_target - robot_pos
            dist = np.linalg.norm(direction)
            if dist <= ROBOT_SPEED:
                robot_pos = edge_target
            else:
                robot_pos += (direction / dist) * ROBOT_SPEED
            robot_pos = np.clip(robot_pos, 0, MAP_SIZE)
            path_trace.append(robot_pos.copy())
    except Exception as exc:  # a crashed trial is kept and counted as a failure, never dropped
        outcome = "error"

    goal_reached = np.linalg.norm(robot_pos - goal) <= 2
    success = goal_reached and not collision
    if outcome is None:
        outcome = "success" if success else ("collision" if collision else "timeout")

    path_length = 0.0
    for i in range(1, len(path_trace)):
        path_length += np.linalg.norm(path_trace[i] - path_trace[i - 1])

    return {
        "Combo": combo_name,
        "w_goal": weights[0], "w_risk": weights[1], "w_ttc": weights[2],
        "w_smooth": weights[3], "w_backtrack": weights[4],
        "Projectile Count": num_projectiles,
        "Projectile Speed": projectile_speed,
        "Seed": seed,
        "Success": int(success),
        "Outcome": outcome,
        "Steps": steps,
        "Path Length": round(path_length, 2),
    }



# ----------------------------------------------------------------------------
# Run the 90 trials
# ----------------------------------------------------------------------------
def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_results.csv")
    rows = []
    t0 = time.time()
    print(f"Running 90 trials with {COMBO_NAME} weights {WEIGHTS} ...")

    for n_proj in PROJECTILE_COUNTS:
        for spd in PROJECTILE_SPEEDS:
            for k in range(TRIALS_PER_CONDITION):
                r = run_trial((COMBO_NAME, WEIGHTS, n_proj, spd, SEED_START + k))
                r["Trial"] = k + 1
                rows.append(r)
            done = sum(1 for x in rows if x["Projectile Count"] == n_proj and x["Projectile Speed"] == spd)
            print(f"  count {n_proj}, speed {spd}: {done} trials done ({time.time() - t0:.0f}s)")

    cols = ["Trial", "Projectile Count", "Projectile Speed", "Success",
            "Steps", "Path Length", "Outcome", "Seed"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])

    total = sum(r["Success"] for r in rows)
    print(f"\nOverall: {total} of {len(rows)} trials successful ({total / len(rows):.1%})")
    print("\nSuccess rate by condition (rows = projectile count, columns = speed)")
    print(f"{'Count':<8}" + "".join(f"{'Speed ' + str(s):>10}" for s in PROJECTILE_SPEEDS))
    for c in PROJECTILE_COUNTS:
        vals = ""
        for s in PROJECTILE_SPEEDS:
            sub = [r["Success"] for r in rows if r["Projectile Count"] == c and r["Projectile Speed"] == s]
            vals += f"{np.mean(sub):>10.2f}"
        print(f"{c:<8}{vals}")

    print("\nSaved to:", out_path)


if __name__ == "__main__":
    main()

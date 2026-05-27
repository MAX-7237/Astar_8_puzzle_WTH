import heapq
import html
import math
import time
from collections import deque
from pathlib import Path


N = 3

start = (
    1, 0, 2,
    3, 4, 5,
    6, 7, 8
)

goal = (
    1, 2, 3,
    4, 5, 6,
    7, 8, 0
)

goal_pos = {value: index for index, value in enumerate(goal)}


def state_to_lines(state):
    lines = []
    for i in range(0, N * N, N):
        row = state[i:i + N]
        line = " ".join("_" if x == 0 else str(x) for x in row)
        lines.append(line)
    return lines


def state_to_string(state):
    return "\n".join(state_to_lines(state))


def state_to_one_line(state):
    return " / ".join(state_to_lines(state))


def manhattan_distance(state):
    distance = 0

    for idx, value in enumerate(state):
        if value == 0:
            continue

        cur_r, cur_c = idx // N, idx % N
        goal_idx = goal_pos[value]
        goal_r, goal_c = goal_idx // N, goal_idx % N

        distance += abs(cur_r - goal_r) + abs(cur_c - goal_c)

    return distance


def get_neighbors(state):
    neighbors = []
    zero_idx = state.index(0)
    r, c = zero_idx // N, zero_idx % N

    directions = [
        (-1, 0, "上"),
        (1, 0, "下"),
        (0, -1, "左"),
        (0, 1, "右"),
    ]

    for dr, dc, move in directions:
        nr, nc = r + dr, c + dc

        if 0 <= nr < N and 0 <= nc < N:
            new_idx = nr * N + nc
            new_state = list(state)
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            neighbors.append((tuple(new_state), move))

    return neighbors


def rebuild_path(parent, end_state):
    path = []
    state = end_state

    while state is not None:
        prev_state, move, g, h = parent[state]
        path.append((state, move, g, h, g + h))
        state = prev_state

    path.reverse()
    return path


def bfs(start_state, goal_state):
    queue = deque([start_state])

    h0 = manhattan_distance(start_state)
    parent = {
        start_state: (None, None, 0, h0)
    }

    visited = {start_state}
    expanded_nodes = 0

    while queue:
        current_state = queue.popleft()
        expanded_nodes += 1

        if current_state == goal_state:
            return rebuild_path(parent, current_state), expanded_nodes

        current_g = parent[current_state][2]

        for next_state, move in get_neighbors(current_state):
            if next_state not in visited:
                visited.add(next_state)
                new_g = current_g + 1
                new_h = manhattan_distance(next_state)
                parent[next_state] = (current_state, move, new_g, new_h)
                queue.append(next_state)

    return None, expanded_nodes


def greedy_best_first(start_state, goal_state):
    open_list = []

    h0 = manhattan_distance(start_state)
    heapq.heappush(open_list, (h0, start_state))

    parent = {
        start_state: (None, None, 0, h0)
    }

    visited = set()
    expanded_nodes = 0

    while open_list:
        h, current_state = heapq.heappop(open_list)

        if current_state in visited:
            continue

        visited.add(current_state)
        expanded_nodes += 1

        if current_state == goal_state:
            return rebuild_path(parent, current_state), expanded_nodes

        current_g = parent[current_state][2]

        for next_state, move in get_neighbors(current_state):
            if next_state not in visited and next_state not in parent:
                new_g = current_g + 1
                new_h = manhattan_distance(next_state)
                parent[next_state] = (current_state, move, new_g, new_h)
                heapq.heappush(open_list, (new_h, next_state))

    return None, expanded_nodes


def astar(start_state, goal_state):
    open_list = []

    h0 = manhattan_distance(start_state)
    heapq.heappush(open_list, (h0, h0, 0, start_state))

    parent = {
        start_state: (None, None, 0, h0)
    }

    g_score = {
        start_state: 0
    }

    closed = set()
    expanded_nodes = 0

    while open_list:
        f, h, g, current_state = heapq.heappop(open_list)

        if current_state in closed:
            continue

        closed.add(current_state)
        expanded_nodes += 1

        if current_state == goal_state:
            return rebuild_path(parent, current_state), expanded_nodes

        for next_state, move in get_neighbors(current_state):
            if next_state in closed:
                continue

            new_g = g + 1
            new_h = manhattan_distance(next_state)
            new_f = new_g + new_h

            if next_state not in g_score or new_g < g_score[next_state]:
                g_score[next_state] = new_g
                parent[next_state] = (current_state, move, new_g, new_h)
                heapq.heappush(open_list, (new_f, new_h, new_g, next_state))

    return None, expanded_nodes


def safe_filename(name):
    name = name.replace(" ", "_")
    name = name.replace("*", "star")
    name = name.replace("/", "_")
    name = name.replace("：", "_")
    return name


def build_result_text(results):
    lines = []

    lines.append("实验 4：不同搜索算法对比")
    lines.append("=" * 70)
    lines.append("")

    lines.append("初始状态：")
    lines.append(state_to_string(start))
    lines.append("")

    lines.append("目标状态：")
    lines.append(state_to_string(goal))
    lines.append("")

    lines.append("一、算法对比结果")
    lines.append("-" * 70)
    lines.append("| 算法 | 是否找到解 | 解的步数 | 扩展节点数 | 运行时间/s | 是否保证最优 |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")

    for result in results:
        name = result["name"]
        path = result["path"]
        expanded = result["expanded"]
        elapsed = result["elapsed"]
        optimal = result["optimal"]

        if path is None:
            lines.append(f"| {name} | 否 | - | {expanded} | {elapsed:.6f} | {optimal} |")
        else:
            lines.append(f"| {name} | 是 | {len(path) - 1} | {expanded} | {elapsed:.6f} | {optimal} |")

    lines.append("")
    lines.append("二、各算法详细求解过程")
    lines.append("=" * 70)

    for result in results:
        name = result["name"]
        path = result["path"]
        expanded = result["expanded"]
        elapsed = result["elapsed"]

        lines.append("")
        lines.append(f"算法：{name}")
        lines.append("-" * 70)
        lines.append(f"扩展节点数：{expanded}")
        lines.append(f"运行时间：{elapsed:.6f} 秒")

        if path is None:
            lines.append("该算法未找到解。")
            continue

        lines.append(f"解的步数：{len(path) - 1}")
        moves = [step[1] for step in path[1:]]
        lines.append("移动序列：")
        lines.append(" -> ".join(moves))
        lines.append("")

        for i, (state, move, g, h, f) in enumerate(path):
            if i == 0:
                lines.append(f"Step {i}: 初始状态, g={g}, h={h}, f={f}")
            else:
                lines.append(f"Step {i}: 空格向{move}移动, g={g}, h={h}, f={f}")

            lines.append(state_to_string(state))
            lines.append("")

        lines.append("表格形式：")
        lines.append("| 步数 | 移动 | g | h | f | 状态 |")
        lines.append("| -: | -- | -: | -: | -: | ----------------------- |")

        for i, (state, move, g, h, f) in enumerate(path):
            move_text = "初始" if i == 0 else move
            lines.append(
                f"| {i} | {move_text} | {g} | {h} | {f} | `{state_to_one_line(state)}` |"
            )

    lines.append("")
    lines.append("三、实验分析")
    lines.append("=" * 70)
    lines.append("BFS 按层扩展节点，可以保证找到最短路径，但不利用启发信息，扩展节点较多。")
    lines.append("Greedy 贪心最佳优先搜索只根据 h(n) 选择当前最接近目标的状态，速度可能较快，但不保证最优解。")
    lines.append("A* 同时考虑已走代价 g(n) 和估计代价 h(n)，使用 f(n)=g(n)+h(n) 进行排序，在启发函数可采纳时可以保证最优。")
    lines.append("本程序同时生成 exp4_algorithm_comparison.svg 对比图，以及每种算法对应的贪吃蛇路径可视化 SVG 文件。")

    return "\n".join(lines)


def svg_text(x, y, text, size=16, weight="normal", anchor="middle", color="#0b3558"):
    text = html.escape(str(text))
    return (
        f'<text x="{x}" y="{y}" '
        f'font-family="Consolas, Microsoft YaHei, monospace" '
        f'font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{color}">{text}</text>'
    )


def draw_board_svg(step_index, state, move, g, h, f, x, y, board_w, board_h, cell):
    parts = []

    parts.append(
        f'<rect x="{x}" y="{y}" width="{board_w}" height="{board_h}" '
        f'rx="12" ry="12" fill="#f8fbff" stroke="#7aa7d9" stroke-width="2"/>'
    )

    title = "Step 0 初始" if step_index == 0 else f"Step {step_index} 空格向{move}"

    parts.append(svg_text(x + board_w / 2, y + 25, title, size=14, weight="bold"))
    parts.append(svg_text(x + board_w / 2, y + 48, f"g={g}, h={h}, f={f}", size=12))

    grid_size = cell * N
    grid_x = x + (board_w - grid_size) / 2
    grid_y = y + 65

    for r in range(N):
        for c in range(N):
            idx = r * N + c
            value = state[idx]

            cx = grid_x + c * cell
            cy = grid_y + r * cell

            fill = "#ffffff" if value != 0 else "#dbe9f8"

            parts.append(
                f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" '
                f'fill="{fill}" stroke="#2c5e91" stroke-width="1.5"/>'
            )

            text = "_" if value == 0 else str(value)
            color = "#5d7fa3" if value == 0 else "#0b3558"
            parts.append(svg_text(cx + cell / 2, cy + cell / 2 + 7, text, size=20, weight="bold", color=color))

    return "\n".join(parts)


def get_snake_position(index, cols, board_w, board_h, h_gap, v_gap, margin):
    row = index // cols
    pos_in_row = index % cols

    if row % 2 == 0:
        col = pos_in_row
    else:
        col = cols - 1 - pos_in_row

    x = margin + col * (board_w + h_gap)
    y = margin + row * (board_h + v_gap)

    return x, y, row, col


def draw_arrow_between(pos1, pos2, board_w, board_h):
    x1, y1, row1, col1 = pos1
    x2, y2, row2, col2 = pos2

    if row1 == row2:
        cy = y1 + board_h / 2

        if x2 > x1:
            start_x = x1 + board_w + 6
            end_x = x2 - 6
        else:
            start_x = x1 - 6
            end_x = x2 + board_w + 6

        return (
            f'<line x1="{start_x}" y1="{cy}" x2="{end_x}" y2="{cy}" '
            f'stroke="#1f6fd1" stroke-width="4" marker-end="url(#arrow)"/>'
        )

    start_x = x1 + board_w / 2
    start_y = y1 + board_h + 6
    end_x = x2 + board_w / 2
    end_y = y2 - 6
    mid_y = (start_y + end_y) / 2

    points = f"{start_x},{start_y} {start_x},{mid_y} {end_x},{mid_y} {end_x},{end_y}"

    return (
        f'<polyline points="{points}" fill="none" stroke="#1f6fd1" stroke-width="4" '
        f'stroke-linejoin="round" stroke-linecap="round" marker-end="url(#arrow)"/>'
    )


def create_path_svg(path, output_svg_path, title, cols=6):
    if path is None:
        return

    board_w = 170
    board_h = 190
    cell = 34
    h_gap = 70
    v_gap = 75
    margin = 45

    total_steps = len(path)
    rows = math.ceil(total_steps / cols)

    width = margin * 2 + cols * board_w + (cols - 1) * h_gap
    height = margin * 2 + rows * board_h + (rows - 1) * v_gap + 60

    svg_parts = []
    svg_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg_parts.append(
        """
<defs>
    <marker id="arrow" markerWidth="12" markerHeight="12"
            refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
        <path d="M 0 0 L 12 6 L 0 12 z" fill="#1f6fd1"/>
    </marker>
</defs>
"""
    )

    svg_parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#eef5ff"/>')
    svg_parts.append(svg_text(width / 2, 28, title, size=20, weight="bold"))

    positions = []

    for i in range(total_steps):
        x, y, row, col = get_snake_position(i, cols, board_w, board_h, h_gap, v_gap, margin)
        y += 30
        positions.append((x, y, row, col))

    for i in range(total_steps - 1):
        svg_parts.append(draw_arrow_between(positions[i], positions[i + 1], board_w, board_h))

    for i, (state, move, g, h, f) in enumerate(path):
        x, y, row, col = positions[i]
        svg_parts.append(draw_board_svg(i, state, move, g, h, f, x, y, board_w, board_h, cell))

    svg_parts.append("</svg>")

    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))


def create_comparison_svg(results, output_svg_path):
    width = 1000
    height = 560
    margin_left = 100
    chart_top = 110
    chart_height = 320
    bar_width = 70

    max_expanded = max(result["expanded"] for result in results)
    max_time = max(result["elapsed"] for result in results)
    max_steps = max((len(result["path"]) - 1) if result["path"] else 0 for result in results)

    svg_parts = []
    svg_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg_parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#eef5ff"/>')
    svg_parts.append(svg_text(width / 2, 40, "实验 4：不同搜索算法效果对比", size=22, weight="bold"))

    svg_parts.append(svg_text(240, 85, "解的步数", size=15, weight="bold"))
    svg_parts.append(svg_text(500, 85, "扩展节点数", size=15, weight="bold"))
    svg_parts.append(svg_text(760, 85, "运行时间 ×100000", size=15, weight="bold"))

    metrics = [
        ("steps", 240, max_steps),
        ("expanded", 500, max_expanded),
        ("time", 760, max_time * 100000),
    ]

    for metric_name, center_x, max_value in metrics:
        svg_parts.append(
            f'<line x1="{center_x - 100}" y1="{chart_top + chart_height}" '
            f'x2="{center_x + 100}" y2="{chart_top + chart_height}" '
            f'stroke="#24496b" stroke-width="2"/>'
        )

        for i, result in enumerate(results):
            if metric_name == "steps":
                value = (len(result["path"]) - 1) if result["path"] else 0
            elif metric_name == "expanded":
                value = result["expanded"]
            else:
                value = result["elapsed"] * 100000

            bar_h = 0 if max_value == 0 else value / max_value * chart_height

            x = center_x - 90 + i * 65
            y = chart_top + chart_height - bar_h

            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" '
                f'fill="#6ca6df" stroke="#2c5e91" stroke-width="1.5"/>'
            )

            text_value = f"{value:.3f}" if metric_name == "time" else str(int(value))
            svg_parts.append(svg_text(x + bar_width / 2, y - 8, text_value, size=12))
            svg_parts.append(svg_text(x + bar_width / 2, chart_top + chart_height + 22, str(i + 1), size=13, weight="bold"))

    legend_y = 480
    for i, result in enumerate(results):
        svg_parts.append(svg_text(170 + i * 280, legend_y, f"{i + 1}. {result['name']}", size=14, anchor="start"))

    svg_parts.append("</svg>")

    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))


def main():
    algorithms = [
        ("BFS 广度优先搜索", bfs, "是"),
        ("Greedy 贪心最佳优先搜索", greedy_best_first, "否"),
        ("A* 启发式搜索", astar, "是"),
    ]

    results = []

    for name, func, optimal in algorithms:
        t0 = time.time()
        path, expanded = func(start, goal)
        t1 = time.time()

        results.append({
            "name": name,
            "path": path,
            "expanded": expanded,
            "elapsed": t1 - t0,
            "optimal": optimal,
        })

    output_text = build_result_text(results)

    print(output_text)

    current_dir = Path(__file__).resolve().parent

    txt_path = current_dir / "exp4_algorithm_compare_result.txt"
    comparison_svg_path = current_dir / "exp4_algorithm_comparison.svg"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    create_comparison_svg(results, comparison_svg_path)

    for result in results:
        name = safe_filename(result["name"])
        path_svg_path = current_dir / f"exp4_path_{name}.svg"
        create_path_svg(
            result["path"],
            path_svg_path,
            f"实验 4：{result['name']} 的求解路径",
            cols=6,
        )

    print()
    print(f"文字结果已保存到：{txt_path}")
    print(f"对比可视化已保存到：{comparison_svg_path}")
    print("每种搜索算法的贪吃蛇路径图也已生成到当前目录。")


if __name__ == "__main__":
    main()
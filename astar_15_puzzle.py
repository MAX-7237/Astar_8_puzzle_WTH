import heapq
import html
import math
import time
from pathlib import Path


N = 4

# 初始状态：0 表示空格
# 这个状态比之前更复杂，最优解大约需要 20 步
start = (
    5, 1, 2, 3,
    4, 7, 10, 8,
    9, 6, 11, 12,
    13, 14, 15, 0
)

# 目标状态
goal = (
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 0
)

goal_pos = {value: index for index, value in enumerate(goal)}


def state_to_lines(state):
    lines = []
    for i in range(0, N * N, N):
        row = state[i:i + N]
        line = " ".join("_" if x == 0 else f"{x:2d}" for x in row)
        lines.append(line)
    return lines


def state_to_string(state):
    return "\n".join(state_to_lines(state))


def state_to_one_line(state):
    return " / ".join(state_to_lines(state))


def print_state(state):
    print(state_to_string(state))
    print()


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
    zero_r, zero_c = zero_idx // N, zero_idx % N

    directions = [
        (-1, 0, "上"),
        (1, 0, "下"),
        (0, -1, "左"),
        (0, 1, "右"),
    ]

    for dr, dc, move_name in directions:
        new_r = zero_r + dr
        new_c = zero_c + dc

        if 0 <= new_r < N and 0 <= new_c < N:
            new_idx = new_r * N + new_c

            new_state = list(state)
            new_state[zero_idx], new_state[new_idx] = (
                new_state[new_idx],
                new_state[zero_idx],
            )

            neighbors.append((tuple(new_state), move_name))

    return neighbors


def inversion_count(state):
    nums = [x for x in state if x != 0]
    count = 0

    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] > nums[j]:
                count += 1

    return count


def blank_row_from_bottom(state):
    zero_idx = state.index(0)
    row_from_top = zero_idx // N
    return N - row_from_top


def is_solvable(start_state, goal_state):
    """
    判断 N 数码是否有解。
    当 N 为奇数时，逆序数奇偶性相同则有解。
    当 N 为偶数时，需要同时考虑空格所在行距离底部的行号。
    """
    inv_start = inversion_count(start_state)
    inv_goal = inversion_count(goal_state)

    if N % 2 == 1:
        return inv_start % 2 == inv_goal % 2

    start_blank_row = blank_row_from_bottom(start_state)
    goal_blank_row = blank_row_from_bottom(goal_state)

    return (inv_start + start_blank_row) % 2 == (inv_goal + goal_blank_row) % 2


def astar(start_state, goal_state):
    if not is_solvable(start_state, goal_state):
        return None, 0

    open_list = []

    h0 = manhattan_distance(start_state)
    heapq.heappush(open_list, (h0, h0, 0, start_state))

    parent = {
        start_state: (None, None, 0, h0)
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
            path = []
            state = current_state

            while state is not None:
                prev_state, move, g_value, h_value = parent[state]
                path.append((state, move, g_value, h_value, g_value + h_value))
                state = prev_state

            path.reverse()
            return path, expanded_nodes

        for next_state, move in get_neighbors(current_state):
            if next_state in closed:
                continue

            new_g = g + 1
            new_h = manhattan_distance(next_state)
            new_f = new_g + new_h

            if next_state not in parent or new_g < parent[next_state][2]:
                parent[next_state] = (current_state, move, new_g, new_h)
                heapq.heappush(open_list, (new_f, new_h, new_g, next_state))

    return None, expanded_nodes


def build_output_text(path, expanded_nodes, elapsed_time):
    lines = []

    lines.append("4x4 十五数码 A* 算法求解结果")
    lines.append("=" * 70)
    lines.append("")

    lines.append("初始状态：")
    lines.append(state_to_string(start))
    lines.append("")

    lines.append("目标状态：")
    lines.append(state_to_string(goal))
    lines.append("")

    if path is None:
        lines.append("该十五数码问题无解。")
        return "\n".join(lines)

    lines.append("找到解！")
    lines.append(f"最少移动步数：{len(path) - 1}")
    lines.append(f"扩展节点数：{expanded_nodes}")
    lines.append(f"运行时间：{elapsed_time:.6f} 秒")
    lines.append("")

    moves = [step[1] for step in path[1:]]
    lines.append("移动序列：")
    lines.append(" -> ".join(moves))
    lines.append("")

    lines.append("详细求解过程：")
    lines.append("=" * 70)
    lines.append("")

    for i, (state, move, g, h, f) in enumerate(path):
        if i == 0:
            lines.append(f"Step {i}: 初始状态, g={g}, h={h}, f={f}")
        else:
            lines.append(f"Step {i}: 空格向{move}移动, g={g}, h={h}, f={f}")

        lines.append(state_to_string(state))
        lines.append("")

    lines.append("表格形式：")
    lines.append("=" * 70)
    lines.append("")
    lines.append("| 步数 | 移动 | g | h | f | 状态 |")
    lines.append("| -: | -- | -: | -: | -: | ------------------------------- |")

    for i, (state, move, g, h, f) in enumerate(path):
        move_text = "初始" if i == 0 else move
        lines.append(
            f"| {i} | {move_text} | {g} | {h} | {f} | `{state_to_one_line(state)}` |"
        )

    lines.append("")
    lines.append("说明：")
    lines.append("移动方向表示空格的移动方向。")
    lines.append("例如“空格向上移动”表示空格与其上方数字交换位置。")
    lines.append("程序同时生成 astar_15puzzle_path.svg，用贪吃蛇式箭头大图展示每一步状态。")

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

    if step_index == 0:
        title = "Step 0  初始"
    else:
        title = f"Step {step_index}  空格向{move}"

    parts.append(svg_text(x + board_w / 2, y + 25, title, size=14, weight="bold"))
    parts.append(
        svg_text(
            x + board_w / 2,
            y + 48,
            f"g={g}, h={h}, f={f}",
            size=12,
            color="#294b6b",
        )
    )

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
                f'fill="{fill}" stroke="#2c5e91" stroke-width="1.4"/>'
            )

            if value != 0:
                parts.append(
                    svg_text(
                        cx + cell / 2,
                        cy + cell / 2 + 6,
                        value,
                        size=16,
                        weight="bold",
                        color="#0b3558",
                    )
                )
            else:
                parts.append(
                    svg_text(
                        cx + cell / 2,
                        cy + cell / 2 + 6,
                        "_",
                        size=16,
                        weight="bold",
                        color="#5d7fa3",
                    )
                )

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

    if abs(start_x - end_x) < 1e-6:
        points = f"{start_x},{start_y} {end_x},{end_y}"
    else:
        mid_y = (start_y + end_y) / 2
        points = (
            f"{start_x},{start_y} "
            f"{start_x},{mid_y} "
            f"{end_x},{mid_y} "
            f"{end_x},{end_y}"
        )

    return (
        f'<polyline points="{points}" fill="none" '
        f'stroke="#1f6fd1" stroke-width="4" '
        f'stroke-linejoin="round" stroke-linecap="round" '
        f'marker-end="url(#arrow)"/>'
    )


def create_solution_svg(path, output_svg_path, cols=5):
    if path is None:
        return

    board_w = 190
    board_h = 215
    cell = 32

    h_gap = 65
    v_gap = 80
    margin = 45

    total_steps = len(path)
    rows = math.ceil(total_steps / cols)

    width = margin * 2 + cols * board_w + (cols - 1) * h_gap
    height = margin * 2 + rows * board_h + (rows - 1) * v_gap + 60

    svg_parts = []

    svg_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
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

    svg_parts.append(
        svg_text(
            width / 2,
            28,
            "4x4 十五数码 A* 求解路径：每个小图为一步状态，箭头表示空格移动顺序",
            size=20,
            weight="bold",
            color="#0b3558",
        )
    )

    positions = []

    for i in range(total_steps):
        x, y, row, col = get_snake_position(
            i,
            cols,
            board_w,
            board_h,
            h_gap,
            v_gap,
            margin,
        )
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


def main():
    print("4x4 十五数码 A* 求解")
    print("=" * 50)

    print("初始状态：")
    print_state(start)

    print("目标状态：")
    print_state(goal)

    begin_time = time.time()
    path, expanded_nodes = astar(start, goal)
    end_time = time.time()

    elapsed_time = end_time - begin_time

    output_text = build_output_text(path, expanded_nodes, elapsed_time)

    print(output_text)

    current_dir = Path(__file__).resolve().parent

    txt_path = current_dir / "astar_15puzzle_result.txt"
    svg_path = current_dir / "astar_15puzzle_path.svg"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    create_solution_svg(path, svg_path, cols=5)

    print()
    print(f"文字结果已保存到：{txt_path}")
    print(f"贪吃蛇箭头大图已保存到：{svg_path}")


if __name__ == "__main__":
    main()
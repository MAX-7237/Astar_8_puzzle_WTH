import heapq
import html
import math
from pathlib import Path
N = 3
# 初始状态：0 表示空格
start = (
    1, 0, 2,
    3, 4, 5,
    6, 7, 8
)
# 目标状态
goal = (
    1, 2, 3,
    4, 5, 6,
    7, 8, 0
)
def state_to_lines(state):
    lines = []
    for i in range(0, 9, 3):
        row = state[i:i + 3]
        line = " ".join("_" if x == 0 else str(x) for x in row)
        lines.append(line)
    return lines
def state_to_string(state):
    return "\n".join(state_to_lines(state))
def state_to_one_line(state):
    lines = state_to_lines(state)
    return " / ".join(lines)
def manhattan_distance(state):
    distance = 0
    for index, value in enumerate(state):
        if value == 0:
            continue
        current_row = index // N
        current_col = index % N
        goal_index = goal.index(value)
        goal_row = goal_index // N
        goal_col = goal_index % N
        distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance
def get_neighbors(state):
    neighbors = []
    zero_index = state.index(0)
    zero_row = zero_index // N
    zero_col = zero_index % N
    directions = [
        (-1, 0, "上"),
        (1, 0, "下"),
        (0, -1, "左"),
        (0, 1, "右"),
    ]
    for dr, dc, move_name in directions:
        new_row = zero_row + dr
        new_col = zero_col + dc
        if 0 <= new_row < N and 0 <= new_col < N:
            new_index = new_row * N + new_col
            new_state = list(state)
            new_state[zero_index], new_state[new_index] = (
                new_state[new_index],
                new_state[zero_index],
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
def is_solvable(start_state, goal_state):
    return inversion_count(start_state) % 2 == inversion_count(goal_state) % 2
def astar(start_state, goal_state):
    if not is_solvable(start_state, goal_state):
        return None
    open_list = []
    h0 = manhattan_distance(start_state)
    heapq.heappush(open_list, (h0, h0, 0, start_state))
    parent = {
        start_state: (None, None, 0, h0)
    }
    closed_set = set()
    while open_list:
        f, h, g, current_state = heapq.heappop(open_list)
        if current_state in closed_set:
            continue
        closed_set.add(current_state)
        if current_state == goal_state:
            path = []
            state = current_state
            while state is not None:
                prev_state, move, g_value, h_value = parent[state]
                path.append((state, move, g_value, h_value, g_value + h_value))
                state = prev_state
            path.reverse()
            return path
        for next_state, move in get_neighbors(current_state):
            if next_state in closed_set:
                continue
            new_g = g + 1
            new_h = manhattan_distance(next_state)
            new_f = new_g + new_h
            if next_state not in parent or new_g < parent[next_state][2]:
                parent[next_state] = (current_state, move, new_g, new_h)
                heapq.heappush(open_list, (new_f, new_h, new_g, next_state))
    return None
def build_output_text(path):
    lines = []
    lines.append("八数码问题 A* 算法求解结果")
    lines.append("=" * 60)
    lines.append("")
    lines.append("初始状态：")
    lines.append(state_to_string(start))
    lines.append("")
    lines.append("目标状态：")
    lines.append(state_to_string(goal))
    lines.append("")
    if path is None:
        lines.append("该八数码问题无解。")
        return "\n".join(lines)
    lines.append("找到解！")
    lines.append(f"最少移动步数：{len(path) - 1}")
    lines.append("")
    moves = [step[1] for step in path[1:]]
    lines.append("移动序列：")
    lines.append(" -> ".join(moves))
    lines.append("")
    lines.append("详细求解过程：")
    lines.append("=" * 60)
    lines.append("")
    for i, (state, move, g, h, f) in enumerate(path):
        if i == 0:
            lines.append(f"Step {i}: 初始状态, g={g}, h={h}, f={f}")
        else:
            lines.append(f"Step {i}: 空格向{move}移动, g={g}, h={h}, f={f}")
        lines.append(state_to_string(state))
        lines.append("")
    lines.append("表格形式：")
    lines.append("=" * 60)
    lines.append("")
    lines.append("| 步数 | 移动 | g | h | f | 状态 |")
    lines.append("| -: | -- | -: | -: | -: | ----------------------- |")
    for i, (state, move, g, h, f) in enumerate(path):
        move_text = "初始" if i == 0 else move
        lines.append(
            f"| {i} | {move_text} | {g} | {h} | {f} | `{state_to_one_line(state)}` |"
        )
    lines.append("")
    lines.append("说明：")
    lines.append("移动方向表示空格的移动方向。例如“空格向下移动”表示空格与其下方数字交换位置。")
    lines.append("同时程序已生成 astar_8puzzle_path.svg，用贪吃蛇式箭头大图展示每一步状态。")
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
    # 外框
    parts.append(
        f'<rect x="{x}" y="{y}" width="{board_w}" height="{board_h}" '
        f'rx="12" ry="12" fill="#f8fbff" stroke="#7aa7d9" stroke-width="2"/>'
    )
    # 标题
    if step_index == 0:
        title = "Step 0  初始"
    else:
        title = f"Step {step_index}  空格向{move}"
    parts.append(svg_text(x + board_w / 2, y + 25, title, size=15, weight="bold"))
    # g h f
    parts.append(
        svg_text(
            x + board_w / 2,
            y + 48,
            f"g={g}, h={h}, f={f}",
            size=13,
            color="#294b6b",
        )
    )
    grid_size = cell * 3
    grid_x = x + (board_w - grid_size) / 2
    grid_y = y + 65
    for r in range(3):
        for c in range(3):
            idx = r * 3 + c
            value = state[idx]
            cx = grid_x + c * cell
            cy = grid_y + r * cell
            fill = "#ffffff" if value != 0 else "#dbe9f8"
            parts.append(
                f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" '
                f'fill="{fill}" stroke="#2c5e91" stroke-width="1.5"/>'
            )
            if value != 0:
                parts.append(
                    svg_text(
                        cx + cell / 2,
                        cy + cell / 2 + 7,
                        value,
                        size=22,
                        weight="bold",
                        color="#0b3558",
                    )
                )
            else:
                parts.append(
                    svg_text(
                        cx + cell / 2,
                        cy + cell / 2 + 7,
                        "_",
                        size=22,
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
    parts = []
    if row1 == row2:
        # 同一行：左右直线箭头
        cy = y1 + board_h / 2
        if x2 > x1:
            start_x = x1 + board_w + 6
            end_x = x2 - 6
        else:
            start_x = x1 - 6
            end_x = x2 + board_w + 6
        parts.append(
            f'<line x1="{start_x}" y1="{cy}" x2="{end_x}" y2="{cy}" '
            f'stroke="#1f6fd1" stroke-width="4" marker-end="url(#arrow)"/>'
        )
    else:
        # 换行：竖直或折线箭头，形成贪吃蛇式连接
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
        parts.append(
            f'<polyline points="{points}" fill="none" '
            f'stroke="#1f6fd1" stroke-width="4" '
            f'stroke-linejoin="round" stroke-linecap="round" '
            f'marker-end="url(#arrow)"/>'
        )
    return "\n".join(parts)
def create_solution_svg(path, output_svg_path, cols=6):
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
            "八数码 A* 算法求解路径：每个小图为一步状态，箭头表示空格移动顺序",
            size=20,
            weight="bold",
            color="#0b3558",
        )
    )
    positions = []
    # 先计算每一步在大图中的位置
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
    # 先画箭头，避免箭头压在棋盘上
    for i in range(total_steps - 1):
        svg_parts.append(draw_arrow_between(positions[i], positions[i + 1], board_w, board_h))
    # 再画每个棋盘
    for i, (state, move, g, h, f) in enumerate(path):
        x, y, row, col = positions[i]
        svg_parts.append(draw_board_svg(i, state, move, g, h, f, x, y, board_w, board_h, cell))
    svg_parts.append("</svg>")
    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
def main():
    path = astar(start, goal)
    output_text = build_output_text(path)
    print(output_text)
    current_dir = Path(__file__).resolve().parent
    txt_path = current_dir / "astar_8puzzle_result.txt"
    svg_path = current_dir / "astar_8puzzle_path.svg"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    create_solution_svg(path, svg_path, cols=6)
    print()
    print(f"文字结果已保存到：{txt_path}")
    print(f"贪吃蛇箭头大图已保存到：{svg_path}")
if __name__ == "__main__":
    main()

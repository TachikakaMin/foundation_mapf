import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os  # 确保导入os模块
from .utils import read_map, parse_file_name
from matplotlib.widgets import Slider  # 添加在文件开头的import部分

def revert_xy(paths):
    if not isinstance(paths[0][0], np.ndarray):
        return [[y, x] for x, y in paths]
    return [[[y, x] for x, y in path] for path in paths]


def visualize_path(all_paths, goal_locations, file_name, video_path=None, show=False):
    all_paths = revert_xy(all_paths)
    goal_locations = revert_xy(goal_locations)
    map_name, agent_num, path_name = parse_file_name(file_name)
    map_data = read_map(map_name)
    height, width = map_data.shape
    steps = len(all_paths)
    
    # Create figure with constrained layout
    fig, (ax, slider_ax) = plt.subplots(2, 1, figsize=(8, 9), 
                                       gridspec_kw={'height_ratios': [20, 1]},
                                       constrained_layout=True)
    
    # Draw map (obstacles in black, free space in white)
    ax.imshow(map_data, cmap="binary")
    # Add grid
    ax.grid(True, which="major", color="gray", linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, width, 1))
    ax.set_yticks(np.arange(-0.5, height, 1))
    ax.set_xticks(np.arange(0, width, 1), minor=True)
    ax.set_yticks(np.arange(0, height, 1), minor=True)
    ax.tick_params(which="minor", length=0)

    # Set labels on minor ticks
    ax.set_xticklabels([], minor=False)
    ax.set_yticklabels([], minor=False)
    ax.set_xticklabels(range(width), minor=True)
    ax.set_yticklabels(range(height), minor=True)

    # Colors for agents
    colors = ["r", "g", "b", "y", "m", "c"]
    agents_scatter = {}
    agents_lines = {}
    agents_annotations = {}
    agents_goal_lines = {}  # 新增：存储目标连线

    # Initialize scatter plots and lines for each agent
    for agent_id in range(agent_num):
        color = colors[agent_id % len(colors)]
        goal_pos = goal_locations[agent_id]
        # Draw goal (star) markers
        ax.scatter(
            goal_pos[0],
            goal_pos[1],
            edgecolor=color,
            facecolor="none",
            marker="*",
            s=100,
        )
        ax.annotate(
            str(agent_id),
            xy=(goal_pos[0], goal_pos[1]),
            xytext=(0, -5),
            textcoords="offset points",
            color="black",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        # Initialize current position scatter
        scatter = ax.scatter([], [], c=color, s=50)
        agents_scatter[agent_id] = scatter

        # Initialize path line
        (line,) = ax.plot([], [], c=color, alpha=0.3)
        agents_lines[agent_id] = line

        # Initialize agent ID annotation
        annotation = ax.annotate(
            str(agent_id),
            xy=(0, 0),
            xytext=(0, 0),
            bbox=dict(boxstyle="circle", fc="white", ec=color),
            ha="center",
            va="center",
            fontsize=6,
        )
        annotation.set_visible(False)
        agents_annotations[agent_id] = annotation

        # 新增：初始化目标连线
        (goal_line,) = ax.plot([], [], c=color, linestyle='--', alpha=0.5)
        agents_goal_lines[agent_id] = goal_line

    current_frame = 0
    is_playing = False  # 添加播放状态控制
    frame_text = ax.text(
        0.02, 1.0, f"Frame: {current_frame}", transform=ax.transAxes, fontsize=12
    )

    # 创建滑块
    frame_slider = Slider(
        ax=slider_ax,
        label='Frame',
        valmin=0,
        valmax=steps-1,
        valinit=0,
        valstep=1
    )

    def update(frame, update_slider=True):
        frame_text.set_text(f"Frame: {frame}")
        if update_slider:
            frame_slider.set_val(frame)  # 只在需要时更新滑块位置
        artists = []
        for agent_id in range(agent_num):
            current_pos = all_paths[frame][agent_id]
            goal_pos = goal_locations[agent_id]  # 目标位置

            # Update agent position
            agents_scatter[agent_id].set_offsets([[current_pos[0], current_pos[1]]])
            artists.append(agents_scatter[agent_id])

            # Update agent ID annotation
            agents_annotations[agent_id].set_position((current_pos[0], current_pos[1]))
            agents_annotations[agent_id].set_visible(True)
            artists.append(agents_annotations[agent_id])

            # Update path line
            path_until_now = [pos[agent_id] for pos in all_paths[: frame + 1]]
            x_coords = [p[0] for p in path_until_now]
            y_coords = [p[1] for p in path_until_now]
            agents_lines[agent_id].set_data(x_coords, y_coords)
            artists.append(agents_lines[agent_id])

            # 新增：更新目标连线
            agents_goal_lines[agent_id].set_data(
                [current_pos[0], goal_pos[0]], 
                [current_pos[1], goal_pos[1]]
            )
            artists.append(agents_goal_lines[agent_id])

        artists.append(frame_text)
        plt.draw()
        return artists
    


    def slider_update(val):
        nonlocal current_frame
        current_frame = int(val)
        update(current_frame, update_slider=False)  # 滑块更新时不要再次更新滑块位置

    frame_slider.on_changed(slider_update)

    def update_frame():
        nonlocal current_frame
        if is_playing:
            current_frame = (current_frame + 1) % steps
            update(current_frame)
            # 设置下一帧的定时器
            fig.canvas.start_event_loop(0.01)  # 10ms 延迟
            if is_playing:  # 再次检查，避免在暂停后继续
                fig.canvas.draw_idle()
                update_frame()

    def on_key(event):
        nonlocal current_frame, is_playing
        if event.key == "right":
            current_frame = min(current_frame + 1, steps - 1)
            update(current_frame)
        elif event.key == "left":
            current_frame = max(current_frame - 1, 0)
            update(current_frame)
        elif event.key == " ":  # 空格键控制播放/暂停
            is_playing = not is_playing
            if is_playing:
                update_frame()
        elif event.key == "v":
            save_video()

    def save_video():
        if video_path is not None:
            print("start saving video")
            anim = animation.FuncAnimation(
                fig, update, frames=steps, interval=200, blit=True
            )
            os.makedirs(video_path, exist_ok=True)
            writer = animation.FFMpegWriter(
                fps=5, 
                metadata=dict(artist='Me'),
                bitrate=1500,
                codec='h264',
                extra_args=['-preset', 'ultrafast', '-crf', '23']
            )
            anim.save(video_path + "/" + path_name + ".mp4", writer=writer)
            print("video saved to ", video_path + "/" + path_name + ".mp4")
    # Bind key events
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initialize first frame
    update(0)

    plt.title(f"Agent Paths: {path_name}\nLeft/Right: prev/next frame, Space: play/pause, V: save video")
    if show:
        plt.show()
    else:
        save_video()
    plt.close()

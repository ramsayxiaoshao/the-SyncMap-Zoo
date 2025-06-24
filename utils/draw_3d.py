import os
import re
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter


def labels2colors(labels):
    labels = np.array(labels)
    labels = labels - np.min(labels)
    # if len(labels) <= 10:
    #     colorbar = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'pink', 'brown', 'gray']
    # else:
    if len(np.unique(labels)) <= 20:
        colorbar = sns.color_palette("tab20", len(np.unique(labels)))  # hls
    else:
        colorbar = sns.color_palette("gist_ncar", len(np.unique(labels)))  # hls
    # Convert RGB tuples to Plotly's "rgb(r, g, b)" format
    colorbar = [f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})' for r, g, b in colorbar]
    colorbar = np.random.choice(colorbar, len(colorbar), replace=False)
    return [colorbar[label] for label in labels]


def rgb_to_hex(rgb):
    # Extract the integers using regex
    rgb_values = re.findall(r'\d+', rgb)
    # Convert the integers to hex and format them properly
    return "#{:02x}{:02x}{:02x}".format(int(rgb_values[0]), int(rgb_values[1]), int(rgb_values[2]))


def convert_rgb_list_to_hex(rgb_list):
    return [rgb_to_hex(rgb) for rgb in rgb_list]


def save_frame_3d(frame, colors, temp_dir, i, dpi=80, iter_multiplier=1000, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                  marker_size=100):
    '''
    Save a single frame of a 3D scatter plot.

    Parameters:
        frame (np.ndarray): The frame data with shape (t, 3).
        colors (list): List of colors for each point.
        temp_dir (str): Directory to save the temporary frame image.
        i (int): The index of the frame.
        dpi (int): Resolution of the output image.
        iter_multiplier (int): Multiplier for iteration count in titles.
        xlim (tuple): Limits for the x-axis.
        ylim (tuple): Limits for the y-axis.
        zlim (tuple): Limits for the z-axis.
    '''
    sns.set_theme()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    for j in range(frame.shape[0]):
        ax.scatter(frame[j, 0], frame[j, 1], frame[j, 2], color=colors[j], s=marker_size)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_title(f"iter {i * iter_multiplier}")

    filename = os.path.join(temp_dir, f'iter_{i}.png')
    plt.savefig(filename, dpi=dpi)
    plt.close()
    return filename


# Not saving images in disk
def save_frame_3d_in_memory(frame, colors, i, dpi=80, iter_multiplier=1000,
                            xlim=(-5, 5), ylim=(-5, 5), zlim=(-5, 5),
                            marker_size=100):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    # for j in range(frame.shape[0]):
    #     ax.scatter(frame[j, 0], frame[j, 1], frame[j, 2], c=colors[j], s=marker_size)

    ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c=colors, s=marker_size)

    margin = 0.5  # 额外空出一点边缘

    x_min, x_max = frame[:, 0].min(), frame[:, 0].max()
    y_min, y_max = frame[:, 1].min(), frame[:, 1].max()
    z_min, z_max = frame[:, 2].min(), frame[:, 2].max()

    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def create_scatter_gif_3d(ndarray, colors, gif_path='./results/output.gif', dpi=80, duration=0.5, n_jobs=-1, iter_multiplier=1000,
                          xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), marker_size=1000):
    '''
    Create a GIF from a 3D scatter plot animation.

    Parameters:
        ndarray (np.ndarray): The input data with shape (l, t, 3).
        colors (list): List of colors for each point.
        gif_path (str): Path to save the output GIF.
        dpi (int): Resolution of the output images.
        duration (float): Duration for each frame in the GIF.
        n_jobs (int): Number of parallel jobs for frame generation.
        iter_multiplier (int): Multiplier for iteration count in titles.
        xlim (tuple): Limits for the x-axis.
        ylim (tuple): Limits for the y-axis.
        zlim (tuple): Limits for the z-axis.
    '''
    # Ensure the ndarray has the correct shape for 3D data
    assert ndarray.shape[2] == 3, "ndarray must have shape (l, t, 3) for 3D data"

    temp_dir = './results/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # print("colors:", len(colors))#9
    frames = []
    for i, frame in tqdm(enumerate(ndarray), total=ndarray.shape[0], desc='Creating GIF frames'):
        # print(f"frame {i} shape:", frame)
        # print("frame.shape:", frame.shape)#(9, 3)
        # print("x range:", frame[:, 0].min(), frame[:, 0].max())
        # print("y range:", frame[:, 1].min(), frame[:, 1].max())
        # print("z range:", frame[:, 2].min(), frame[:, 2].max())
        frames.append(save_frame_3d_in_memory(frame, colors, i, dpi, iter_multiplier, xlim, ylim, zlim, marker_size))

    # frames = [save_frame_3d_in_memory(frame, colors, i, dpi, iter_multiplier, xlim, ylim, zlim, marker_size)
    #           for i, frame in tqdm(enumerate(ndarray), total=ndarray.shape[0], desc='Creating GIF frames')]

    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
    # # Create frames in parallel
    # filenames = Parallel(n_jobs=n_jobs)(
    #     delayed(save_frame_3d_in_memory)(frame, colors, temp_dir, i, dpi, iter_multiplier, xlim, ylim, zlim, marker_size)
    #     for i, frame in tqdm(enumerate(ndarray), total=ndarray.shape[0], desc='Creating frames')
    # )
    #
    # # Create GIF
    # frames = [Image.open(filename) for filename in filenames]
    # frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
    # for filename in filenames:
    #     os.remove(filename)


def animate_3d_coords(coords, colors, gif_path='./results/output.gif', interval=20, marker_size=50, stride=100):
    """
    coords: (T, N, 3) numpy array
    colors: list of hex or RGB tuples, length=N
    """
    assert coords.ndim == 3 and coords.shape[2] == 3
    T, N, _ = coords.shape
    assert len(colors) == N, f"colors must match number of nodes (N={N})"

    sampled_coords = coords[::stride]  # shape = (T//stride, N, 3)
    T_sampled = sampled_coords.shape[0]

    # 全局坐标范围
    margin = 0.5
    x_min, x_max = coords[:, :, 0].min(), coords[:, :, 0].max()
    y_min, y_max = coords[:, :, 1].min(), coords[:, :, 1].max()
    z_min, z_max = coords[:, :, 2].min(), coords[:, :, 2].max()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)

    # 初始化散点（只用第一帧）
    scat = ax.scatter(coords[0, :, 0], coords[0, :, 1], coords[0, :, 2], c=colors, s=marker_size)

    def update(frame_idx):
        pos = coords[frame_idx]
        scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        ax.set_title(f"Frame {frame_idx * stride}", fontsize=12)
        return scat,

    ani = FuncAnimation(fig, update, frames=T_sampled, interval=interval, blit=False)
    ani.save(gif_path, dpi=80, writer=PillowWriter(fps=30))
    plt.close()

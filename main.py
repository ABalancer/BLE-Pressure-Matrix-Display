import tkinter as tk
import numpy as np
import random
import time
from tkinter import ttk
from multiprocessing_functions import *

'''
TODO: 
make sure each corner is used for base of support placement.
generate a target that randomly appears within base of support.
check when user meets target.
move target and count the completion_time it took for target to be met.

If tracking circle outside of base of support, regenerate tracking circle.

Targets caught:
Average completion_time to catch target:

Tracking accuracy:
'''

UUID_CHARACTERISTIC = "2156AF88-FBD2-4B37-BFD9-5C2C7C293D5A"
GRID_SIZE = 16
GRID_HEIGHT = 500
GRID_WIDTH = 500


colour_interpolation_values = [
    (13, 22, 135), (45, 25, 148), (66, 29, 158), (90, 32, 165), (112, 34, 168),
    (130, 35, 167), (148, 35, 161), (167, 36, 151), (182, 48, 139), (196, 63, 127),
    (208, 77, 115), (220, 93, 102), (231, 109, 92), (239, 126, 79), (247, 143, 68),
    (250, 160, 58), (254, 181, 44), (253, 202, 40), (247, 226, 37), (240, 249, 32)
]


def interpolate_colours(value):
    if value < 4095:
        colour_steps = len(colour_interpolation_values) - 1
        step = 4095 / colour_steps
        start_step = int(value // step)
        end_step = min(start_step + 1, colour_steps)

        start_color = colour_interpolation_values[start_step]
        end_color = colour_interpolation_values[end_step]

        start_r, start_g, start_b = start_color
        end_r, end_g, end_b = end_color

        start_value = start_step * step
        end_value = end_step * step

        ratio = (value - start_value) / (end_value - start_value)
        red = int(start_r + (end_r - start_r) * ratio)
        green = int(start_g + (end_g - start_g) * ratio)
        blue = int(start_b + (end_b - start_b) * ratio)
    else:
        red, green, blue = colour_interpolation_values[-1]
    return f'#{red:02x}{green:02x}{blue:02x}'  # Convert RGB values to hexadecimal color code


def remap_matrix(matrix, threshold):
    # Convert the matrix to a NumPy array
    np_matrix = np.array(matrix)
    np_matrix -= threshold
    remapped_matrix = 2*np.where(np_matrix < 0, 0, np_matrix)
    return np.fliplr(remapped_matrix)
    #return remapped_matrix


def create_colourmap():
    colour_array = []
    for i in range(0, 4096):
        colour_array.append(interpolate_colours(i))
    return colour_array


def create_widget(parent, widget_type, *args, **kwargs):
    widget = widget_type(parent, *args, **kwargs)

    widget.config(background="#2b2b2b", borderwidth=0, relief=tk.FLAT)
    # Apply the styling based on the current mode (light/dark)
    if widget_type is tk.Canvas:
        widget.config(highlightthickness=0)
    if widget_type is tk.Label or widget_type is tk.Listbox or widget_type is tk.Button:
        widget.config(foreground="#a8b5c4", font=("Helvetica", 13))
    if widget_type is tk.Button:
        widget.config(highlightbackground="#2b2b2b", activebackground="#485254",
                      activeforeground="#a8b5c4", background="#3c3f41", width=17, padx=2, pady=2)
    if widget_type is tk.Listbox:
        widget.config(exportselection=False, background="#3c3f41", height=5)
    return widget


def scale_tuple(input_tuple, x_scale, y_scale, total_rows, total_columns):
    output_tuple = (round((input_tuple[0]) * x_scale / total_columns),
                    round((input_tuple[1]) * y_scale / total_rows))
    return output_tuple


class Matrix:
    def __init__(self, canvas, rows, columns):
        self.canvas = canvas
        self.rows = rows
        self.columns = columns
        self.canvas_width = canvas.winfo_reqwidth()
        self.canvas_height = canvas.winfo_reqheight()
        self.pc_x_pos = self.canvas_width / 2
        self.pc_y_pos = self.canvas_height / 2
        self.cell_width = self.canvas_width // columns
        self.cell_height = self.canvas_height // rows
        self.rectangles = []
        self.colour_map = create_colourmap()
        self.base_of_support_lines = None
        self.target_circle = None
        self.pressure_circle = None
        self.draw()

    def draw(self):
        for row in range(self.rows):
            for col in range(self.columns):
                x1 = col * self.cell_width + 1
                y1 = row * self.cell_height + 1
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height

                rectangle = self.canvas.create_rectangle(x1, y1, x2, y2, outline="#777777")
                self.rectangles.append(rectangle)

        self.pressure_circle = self.canvas.create_oval(self.canvas_width / 2 - 5, self.canvas_height / 2 - 5,
                                                       self.canvas_width / 2 + 5, self.canvas_height / 2 + 5,
                                                       fill='white', outline='', state='hidden', tag='pressure_circle')

    def edit_rectangle(self, row, col, color):
        index = row * self.columns + col
        if 0 <= index < len(self.rectangles):
            self.canvas.itemconfig(self.rectangles[index], fill=color)

    def match_colours(self, matrix_data):
        # Map each value in the matrix to a color
        if self._check_matrix_size(matrix_data):
            try:
                colour_matrix = [[self.colour_map[value] for value in row] for row in matrix_data]
                return colour_matrix
            except IndexError:
                return None
        else:
            return None

    def update_matrix(self, colour_matrix):
        if colour_matrix:
            for row in range(0, self.rows):
                for column in range(0, self.columns):
                    self.edit_rectangle(row, column, colour_matrix[row][column])

    def _check_matrix_size(self, matrix):
        if len(matrix) == self.rows:
            if len(matrix[15]) == self.columns:
                return True
        print('Matrix data did not match with the expected size')
        return False

    def plot_centre_of_pressure(self, matrix_data):
        # Create coordinate matrices for X and Y
        x, y = np.meshgrid(np.arange(matrix_data.shape[1]), np.arange(matrix_data.shape[0]))
        # Calculate total pressure and centroid coordinates
        total_pressure = np.sum(matrix_data)
        if total_pressure > 0:
            centre_x = np.sum(x * matrix_data) / total_pressure
            centre_y = np.sum(y * matrix_data) / total_pressure
            # print("X: {}, Y: {}".format(centre_x, centre_y))
            new_centre_x = self.canvas_width * centre_x / (self.rows - 1)
            new_centre_y = self.canvas_height * centre_y / (self.columns - 1)
            centre_dx = new_centre_x - self.pc_x_pos
            centre_dy = new_centre_y - self.pc_y_pos
            self.pc_x_pos = new_centre_x
            self.pc_y_pos = new_centre_y
            self.canvas.move('pressure_circle', centre_dx, centre_dy)
            self.canvas.itemconfigure('pressure_circle', state='normal')
        else:
            self.canvas.itemconfigure('pressure_circle', state='hidden')

    def find_base_of_support(self, matrix_data):
        min_row = self.rows + 1
        max_row = -1
        min_col = self.columns + 1
        max_col = -1

        for row in range(self.rows):
            for column in range(self.columns):
                if matrix_data[row][column] != 0:
                    min_row = min(min_row, row)
                    max_row = max(max_row, row)
                    min_col = min(min_col, column)
                    max_col = max(max_col, column)

        if min_row == self.rows + 1 or max_row == -1 or min_col == self.columns + 1 or max_col == -1:
            return None, None, None, None
        else:
            min_col += 0.5
            max_col += 0.5
            min_row += 0.5
            max_row += 0.5
            return [(min_col, min_row), (min_col, max_row), (max_col, min_row), (max_col, max_row)]

    def draw_base_of_support(self, top_left, top_right, bottom_left, bottom_right):
        if self.base_of_support_lines is not None:
            self.canvas.delete(self.base_of_support_lines)

        if top_left is not None and top_right is not None and bottom_left is not None and bottom_right is not None:
            top_left = scale_tuple(top_left, self.canvas_width, self.canvas_height, self.rows, self.columns)
            top_right = scale_tuple(top_right, self.canvas_width, self.canvas_height, self.rows, self.columns)
            bottom_left = scale_tuple(bottom_left, self.canvas_width, self.canvas_height, self.rows, self.columns)
            bottom_right = scale_tuple(bottom_right, self.canvas_width, self.canvas_height, self.rows, self.columns)

            self.base_of_support_lines = self.canvas.create_line(top_left, top_right,
                                                                 bottom_right, bottom_left,
                                                                 top_left,
                                                                 width=3, fill="white")

    def generate_target(self, top_left, bottom_right):
        self.remove_target()
        if top_left is not None and bottom_right is not None:
            radius = 20
            scaled_top_left = scale_tuple(top_left, self.canvas_width,
                                          self.canvas_height, self.rows, self.columns)
            scaled_bottom_right = scale_tuple(bottom_right, self.canvas_width,
                                              self.canvas_height, self.rows, self.columns)
            target_location = (random.uniform(scaled_top_left[0] + radius, scaled_bottom_right[0] - radius),
                               random.uniform(scaled_bottom_right[1] - radius, scaled_top_left[1] + radius))
            self.target_circle = self.canvas.create_oval(target_location[0] - radius, target_location[1] - radius,
                                                         target_location[0] + radius, target_location[1] + radius,
                                                         fill="", outline="lightgreen", width=4)

    def remove_target(self):
        if self.target_circle:
            self.canvas.delete(self.target_circle)
            self.target_circle = None

    def check_pressure_target_overlap(self):
        overlap = False
        if self.target_circle is not None:
            # Get coordinates and radii of the circles
            x1, y1, x2, y2 = self.canvas.coords(self.pressure_circle)
            pressure_circle_radius = (x2 - x1) / 2
            pressure_x_centre = x1 + pressure_circle_radius
            pressure_y_centre = y1 + pressure_circle_radius

            x1, y1, x2, y2 = self.canvas.coords(self.target_circle)
            target_circle_radius = (x2 - x1) / 2
            target_x_centre = x1 + target_circle_radius
            target_y_centre = y1 + target_circle_radius

            # Calculate distance between centers
            distance = np.sqrt((target_x_centre - pressure_x_centre) ** 2 +
                               (target_y_centre - pressure_y_centre) ** 2)

            # print(pressure_circle_radius, target_circle_radius, distance)
            if distance + pressure_circle_radius < target_circle_radius:
                overlap = True
        return overlap


class App:
    def __init__(self, name):
        # Multiprocessing
        self.timer = [0, 0]
        self.is_data_available = multiprocessing.Value('i', 0)
        self.connected = multiprocessing.Value('i', 0)
        self.lock = multiprocessing.Lock()

        # Bluetooth connection and tasks
        self.selected_device = None
        self.devices = []
        self.task_start_time = 0
        self.random_target_task_score = 0
        self.random_target_task = None
        self.tracking_task = None
        self.top_left = None
        self.bottom_right = None

        # Tkinter
        self.root = tk.Tk()
        self.root.config(background="#2b2b2b")
        self.root.title(name)
        self.root.resizable(False, False)
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.root.iconbitmap("icon.ico")
        self.root.protocol("WM_DELETE_WINDOW", self._exit)

        # Canvas matrix grid
        self.matrix_canvas = create_widget(self.root, tk.Canvas, width=GRID_WIDTH, height=GRID_HEIGHT, borderwidth=0)
        self.matrix_canvas.grid(row=0, column=0)

        self.grid = Matrix(self.matrix_canvas, rows=GRID_SIZE, columns=GRID_SIZE)
        self.grid.draw()

        self.heat_canvas = create_widget(self.root, tk.Canvas, width=GRID_WIDTH, height=25)
        self.heat_canvas.grid(row=1, column=0)
        self.create_heatmap_scale(GRID_WIDTH, 25, self.grid.colour_map)

        # Activities
        self.activities_frame = create_widget(self.root, tk.Frame)
        self.activities_frame.grid(row=2, column=0, sticky="we")
        self.activities_frame.columnconfigure((0, 1, 2), weight=1)

        self.activity_label = create_widget(self.activities_frame, tk.Label, text="Activities:")
        self.activity_label.grid(row=0, column=0, columnspan=3, sticky="w")

        self.start_random_target_task_button = create_widget(self.activities_frame, tk.Button, text="Random Targets",
                                                             command=self.start_random_target_task)
        self.start_random_target_task_button.grid(row=1, column=0)

        self.start_tracking_task_button = create_widget(self.activities_frame, tk.Button, text="Tracking Task",
                                                        command=self.start_tracking_task)
        self.start_tracking_task_button.grid(row=1, column=1)

        self.end_task_button = create_widget(self.activities_frame, tk.Button, text="End Task",
                                             command=self.end_task)
        self.end_task_button.grid(row=1, column=2)

        self.score_label = create_widget(self.activities_frame, tk.Label, text=" ")
        self.score_label.grid(row=2, column=0, sticky="w")
        self.time_label = create_widget(self.activities_frame, tk.Label, text=" ")
        self.time_label.grid(row=2, column=1, columnspan=2, sticky="w")

        # BLE Box
        self.ble_frame = create_widget(self.root, tk.Frame)
        self.ble_frame.grid(row=3, column=0, sticky="we")
        self.ble_frame.columnconfigure((0, 1, 2), weight=1)

        self.ble_label = create_widget(self.ble_frame, tk.Label, text="BLE Devices:")
        self.ble_label.grid(row=0, column=0, columnspan=3, sticky="w")

        self.devices_listbox = create_widget(self.ble_frame, tk.Listbox)
        self.devices_listbox.grid(row=1, column=0, columnspan=3, stick="nsew")

        self.search_button = create_widget(self.ble_frame, tk.Button, text="Search", command=self.update_devices_list)
        self.search_button.grid(row=2, column=0)

        self.connect_button = create_widget(self.ble_frame, tk.Button, text="Connect", command=self.connect_to_device)
        self.connect_button.grid(row=2, column=1)

        self.disconnect_button = create_widget(self.ble_frame, tk.Button, text="Disconnect",
                                               command=self.disconnect_from_device)
        self.disconnect_button.grid(row=2, column=2)
        self.root.columnconfigure(0, weight=1)

        # Initialise certain states
        self.connect_disconnect_buttons_state(False)
        self.end_task_button.config(state=tk.DISABLED)

    def run(self):
        self.root.mainloop()

    def _exit(self):
        if self.is_data_available:
            self.disconnect_from_device()
        self.root.destroy()

    def create_heatmap_scale(self, width, height, colour_map):
        for x in range(width):
            increment = 4095 * x / width
            colour = colour_map[round(increment)]
            self.heat_canvas.create_line(x, 0, x, height, fill=colour, width=1)

    # Function to update the listbox with detected devices
    def update_devices_list(self):
        self.search_button.config(state=tk.DISABLED)
        queue = multiprocessing.Queue()
        process = process_handler(target=device_scanner, args=(queue, self.lock, self.is_data_available))
        self.root.after(50, self._update_devices_status, queue, process)

    def _update_devices_status(self, queue, process):
        if self.is_data_available.value == 1:
            self.lock.acquire()
            self.is_data_available.value = 0
            self.lock.release()
            self.devices = queue.get()
            self.devices_listbox.delete(0, tk.END)
            for device in self.devices:
                self.devices_listbox.insert(tk.END, "{}: {}".format(device[0], device[1]))
            self.search_button.config(state=tk.NORMAL)
        elif self.is_data_available.value == 0:
            self.root.after(50, self._update_devices_status, queue, process)
        else:
            self.lock.acquire()
            self.is_data_available.value = 0
            self.lock.release()
            self.search_button.config(state=tk.NORMAL)

    # toggles the Connect, Disconnect and Search buttons
    def connect_disconnect_buttons_state(self, state):  # if true turn connect button off, disconnect on
        self.connect_button.config(state=tk.DISABLED if state else tk.NORMAL)
        self.search_button.config(state=tk.DISABLED if state else tk.NORMAL)
        self.disconnect_button.config(state=tk.NORMAL if state else tk.DISABLED)
        self.start_random_target_task_button.config(state=tk.NORMAL if state else tk.DISABLED)
        self.start_tracking_task_button.config(state=tk.NORMAL if state else tk.DISABLED)

    def start_random_target_task(self):
        if self.random_target_task is None:
            self.random_target_task = self.root.after(1000, self._target_task)
            self.random_target_task = True
            self.task_start_time = time.time()
            self.random_target_task_score = 0
            self.set_activity_score_labels(self.random_target_task_score, 0.0)
        self.start_random_target_task_button.config(state=tk.DISABLED)
        self.end_task_button.config(state=tk.NORMAL)

    def start_tracking_task(self):
        print("TODO")

    def _end_tracking_task(self):
        self.root.after_cancel(self.tracking_task)
        self.tracking_task = None

    def _end_random_target_task(self):
        self.root.after_cancel(self.random_target_task)
        self.random_target_task = None
        self.grid.remove_target()

    def end_task(self):
        if self.random_target_task is not None:
            self._end_random_target_task()
            self.start_random_target_task_button.config(state=tk.NORMAL)
        if self.tracking_task is not None:
            self._end_tracking_task()
        self.end_task_button.config(state=tk.DISABLED)

    def set_activity_score_labels(self, score, completion_time):
        self.score_label.config(text="Score: %d" % score)
        self.time_label.config(text="Average Time: %.2f seconds" % completion_time)

    def reset_activity_score_labels(self):
        self.score_label.config(text=" ")
        self.time_label.config(text=" ")

    # Function to connect to device
    def connect_to_device(self):
        if self.devices_listbox.size() > 0:
            self.connect_disconnect_buttons_state(True)
            queue = multiprocessing.Queue()
            lock = multiprocessing.Lock()
            self.selected_device = self.devices[self.devices_listbox.curselection()[0]][0]
            process = process_handler(target=connect, args=(queue, lock, self.connected, self.is_data_available,
                                                            self.selected_device, UUID_CHARACTERISTIC))
            self.root.after(5, self._map_updater_task, queue, process)

    # Recursive Loop for updating the matrix when connected to device
    def _map_updater_task(self, queue, process):
        if process.is_alive():
            if self.is_data_available.value:
                self.lock.acquire()
                self.is_data_available.value = 0
                self.lock.release()
                matrix_data = queue.get()
                if matrix_data:
                    matrix_data = remap_matrix(matrix_data, 2048)
                    matrix_colours = self.grid.match_colours(matrix_data)
                    self.grid.update_matrix(matrix_colours)
                    self.grid.plot_centre_of_pressure(matrix_data)
                    self.top_left, top_right, \
                        bottom_left, self.bottom_right = self.grid.find_base_of_support(matrix_data)
                    self.grid.draw_base_of_support(self.top_left, top_right, bottom_left, self.bottom_right)
                    if self.random_target_task is not None:  # Pressure circle within target
                        if self.grid.check_pressure_target_overlap():
                            self._end_random_target_task()
                            self.random_target_task_score += 1
                            average_target_time = (time.time() - self.task_start_time) / self.random_target_task_score
                            self._target_task()
                            self.set_activity_score_labels(self.random_target_task_score, average_target_time)

            self.root.after(5, self._map_updater_task, queue, process)
        else:
            self.end_task()
            self.connect_disconnect_buttons_state(False)

    # Recursive loop for changing the target location if the user takes too long to reach it.
    def _target_task(self):
        reset_time = 15
        self.grid.generate_target(self.top_left, self.bottom_right)
        self.random_target_task = self.root.after(int(reset_time * 1000), self._target_task)

    # Function to disconnect from the connected device
    def disconnect_from_device(self):
        self.lock.acquire()
        self.is_data_available.value = 0
        self.connected.value = 0
        self.lock.release()


if __name__ == "__main__":
    program = App("BLE Pressure Mat")
    program.run()

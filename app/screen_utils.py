# Dynamic screen dimension resizing

# contains the function get_scaled_dimensions that returns the screen dimensions scaled to a percentage.

import tkinter as tk

def get_scaled_dimensions(scale=0.8):
    """Get screen dimensions scaled to a percentage."""
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()  # Close the Tkinter root window
    return int(screen_width * scale), int(screen_height * scale)
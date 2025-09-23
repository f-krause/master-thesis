import os
import platform


def set_output_dir():
    if platform.node() == "TGA-NB-060":
        OUTPUT_DIR = r"C:\Users\felix.krause\code\uni\master-thesis-text\assets\data_viz"
    elif platform.node() == "Felix-PC":
        OUTPUT_DIR = r"C:\Users\Felix\code\uni\UniVie\master-thesis-text\assets\data_viz"
    else:
        raise ValueError("Unknown platform")

    os.environ["OUTPUT_DIR"] = OUTPUT_DIR


set_output_dir()

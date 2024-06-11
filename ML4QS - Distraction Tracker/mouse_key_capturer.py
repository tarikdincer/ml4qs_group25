import logging
from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Key
import os

base_dir = './data-1/'
logging.basicConfig(filename=(os.path.join(base_dir, 'log.csv')), level=logging.DEBUG, format='%(asctime)s, %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def on_press(key):
    try:
        logging.info(f'Key Press,{key.char},,')
    except AttributeError:
        logging.info(f'Key Press,{key},,')

def on_move(x, y):
    logging.info(f'Mouse Move,,,{x},{y}')

def on_click(x, y, button, pressed):
    if pressed:
        logging.info(f'Mouse Click,{button},{x},{y}')

def on_scroll(x, y, dx, dy):
    logging.info(f'Mouse Scroll,,,{x},{y}')

with open("log.csv", "w") as file:
    file.write("Time,Event,Key/Button,PositionX,PositionY\n")

with MouseListener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as mouse_listener:
    with KeyboardListener(on_press=on_press) as keyboard_listener:
        keyboard_listener.join()
        mouse_listener.join()

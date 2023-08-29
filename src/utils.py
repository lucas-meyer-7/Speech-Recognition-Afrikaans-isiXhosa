import os
import re

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

def remove_special_characters(sentence):
    return re.sub(chars_to_remove_regex, '', sentence).lower()

def change_pwd():
    """Change the cwd to the script's directory"""
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    os.chdir(script_directory)
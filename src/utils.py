import gc
import os
import re
import torch
import tarfile
import requests
from tqdm import tqdm


# ---------------------------------------
# Changing power working directory to the
# directory of the python file running
# this function.
# ---------------------------------------

SR = 16000
WRITE_ACCESS_TOKEN = "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"

def change_pwd():
    """Change the cwd to the script's directory"""
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    os.chdir(script_directory)


# ---------------------------------------
# Preprocess transcript data
# ---------------------------------------

# Reference: https://usefulwebtool.com/characters-afrikaans
chars_to_remove_regex = '[\[\]\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\|'
chars_to_remove_regex += "\u002f"  # Forward slash
chars_to_remove_regex += "\u2215"  # Divide
chars_to_remove_regex += "\u00b0"  # Angle/degree sign
chars_to_remove_regex += "\u00b2"  # Pow2 sign
chars_to_remove_regex += "\u00e7"  # "c" with elongation
chars_to_remove_regex += "\u00f5"  # "o" with tilde
chars_to_remove_regex += "\u0142"  # "l" with line through it
chars_to_remove_regex += "\u0144"  # "n" with an acute
chars_to_remove_regex += "\u0307"  # No idea, weird characters
chars_to_remove_regex += "\u0308"  # No idea, weird characters
chars_to_remove_regex += "\u0366"  # Circle thing
chars_to_remove_regex += "\u03ca"  # Greek letter
chars_to_remove_regex += "\u2013"  # Long dash
chars_to_remove_regex += "\u2019"  # Quote
chars_to_remove_regex += "]"

def remove_special_characters(sentence):
    return re.sub(chars_to_remove_regex, '', sentence).lower()

def remove_special_characters_batch(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch


# ---------------------------------------
# Downloading datasets
# ---------------------------------------

def download_file(url, file_name):
    """
    Downloads the file for the given ``url``, and names it 
    as the given ``file_name``.
    """

    # Set up for downloading blocks and set up progress bar
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 128 KB
    print(f"Downloading {os.path.basename(file_name)} ...")
    progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True)

    # Download all blocks and update progress bar
    with open(file_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    print(f"File {os.path.basename(file_name)} downloaded successfully!\n")

def extract_tar_file(tar_path, extract_path):
    """
    Uses the tarfile library to extract 
    the downloaded data sets.
    """
    
    # Extract and remove tarfile
    with tarfile.open(tar_path, "r:gz") as tar_:
        tar_.extractall(path=extract_path)
    os.remove(tar_path)

def download_high_quality_tts(language: str):
    """
    Downloads data required for this project. 
    Also checks if data is already downloaded
    """

    # Check arguments
    if not (language == "af" or language == "xh" or language == "both"):
        raise Exception("Must specify a language as either 'af'/'xh'/'both'.")

    # Make directories for datasets
    change_pwd()
    DATA_PATH = os.path.join("data", "speech_data", "downloaded", "high-quality-tts-data")
    # os.makedirs(DATA_PATH, exist_ok=True)
    af_url = "https://repo.sadilar.org/bitstream/handle/20.500.12185/527/af_za.tar.gz"
    af_file_name = os.path.join(DATA_PATH, "af_za.tar.gz")
    xh_url = "https://repo.sadilar.org/bitstream/handle/20.500.12185/527/xh_za.tar.gz"
    xh_file_name = os.path.join(DATA_PATH, "xh_za.tar.gz")

    # Download and extract the datasets
    try:
        if language == "af":
            download_file(af_url, af_file_name)
            extract_tar_file(af_file_name, DATA_PATH)
        elif language == "xh":
            download_file(xh_url, xh_file_name)
            extract_tar_file(xh_file_name, DATA_PATH)
        elif language == "both":
            download_file(af_url, af_file_name)
            download_file(xh_url, xh_file_name)
            extract_tar_file(af_file_name, DATA_PATH)
            extract_tar_file(xh_file_name, DATA_PATH)
        else:
            raise Exception("")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while handling the data: {e}")
    except Exception as e:
        print(f"Error occurred while handling the data: {e}")

def download_nchlt(language: str):
    """
    Downloads data required for this project. 
    Also checks if data is already downloaded
    """

    # Check arguments
    if not (language == "af" or language == "xh" or language == "both"):
        raise Exception("Must specify a language as either 'af'/'xh'/'both'.")

    # Make directories for datasets
    change_pwd()
    DATA_PATH = os.path.join("data", "speech_data", "downloaded")
    AF_PATH = os.path.join(DATA_PATH, "nchlt_afr")
    XH_PATH = os.path.join(DATA_PATH, "nchlt_xho")
    # if language == "af":
    #     os.makedirs(AF_PATH, exist_ok=True)
    # elif language == "xh":
    #     os.makedirs(XH_PATH, exist_ok=True)
    # elif language == "both":
    #     os.makedirs(AF_PATH, exist_ok=True)
    #     os.makedirs(XH_PATH, exist_ok=True)

    # Store URLs of datasets
    af_url = "http://ftp.internat.freebsd.org/pub/nchlt/Speech_corpora/nchlt_afr.tar.gz"
    af_file_name = os.path.join(AF_PATH, "nchlt_afr.tar.gz")
    xh_url = "http://ftp.internat.freebsd.org/pub/nchlt/Speech_corpora/nchlt_xho.tar.gz"
    xh_file_name = os.path.join(XH_PATH, "nchlt_xho.tar.gz")

    # Download and extract the datasets
    try:
        if language == "af":
            download_file(af_url, af_file_name)
            extract_tar_file(af_file_name, AF_PATH)
        elif language == "xh":
            download_file(xh_url, xh_file_name)
            extract_tar_file(xh_file_name, XH_PATH)
        elif language == "both":
            download_file(af_url, af_file_name)
            extract_tar_file(af_file_name, AF_PATH)
            download_file(xh_url, xh_file_name)
            extract_tar_file(xh_file_name, XH_PATH)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while handling the data: {e}")
    except Exception as e:
        print(f"Error occurred while handling the data: {e}")


# ---------------------------------------
# Clear cache (PyTorch)
# ---------------------------------------

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

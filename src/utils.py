import os
import re
import tarfile
import requests
from tqdm import tqdm

# ---------------------------------------
# Changing power working directory to the
# directory of the python file running
# this function.
# ---------------------------------------

def change_pwd():
    """Change the cwd to the script's directory"""
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    os.chdir(script_directory)

# ---------------------------------------
# Preprocess transcript data
# ---------------------------------------

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
def remove_special_characters(sentence):
    return re.sub(chars_to_remove_regex, '', sentence).lower()

# ---------------------------------------
# Downloading datasets
# ---------------------------------------

def download_file(url, file_name):
    """
    Downloads the file for the given ``url``, and names it 
    as the given ``file_name``.
    """
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB
    print(f"Downloading {os.path.basename(file_name)} ...")
    progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True)

    with open(file_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    print(f"File {os.path.basename(file_name)} downloaded successfully!\n")

def extract_tar_file(tar_path, extract_path):
    """Uses the tarfile library to extract the downloaded data sets."""
    with tarfile.open(tar_path, "r:gz") as tar_:
        tar_.extractall(path=extract_path)
    os.remove(tar_path)

def download_high_quality_tts():
    """
    Downloads data required for this project. 
    Also checks if data is already downloaded
    """
    change_pwd()
    DATA_PATH = os.path.join("downloaded", "high-quality-tts-data")

    af_url = "https://repo.sadilar.org/bitstream/handle/20.500.12185/527/af_za.tar.gz"
    af_file_name = os.path.join(DATA_PATH, "af_za.tar.gz")
    xh_url = "https://repo.sadilar.org/bitstream/handle/20.500.12185/527/xh_za.tar.gz"
    xh_file_name = os.path.join(DATA_PATH, "xh_za.tar.gz")

    # Download and extract the datasets
    try:
        download_file(af_url, af_file_name)
        download_file(xh_url, xh_file_name)
        extract_tar_file(af_file_name, DATA_PATH)
        extract_tar_file(xh_file_name, DATA_PATH)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while handling the data: {e}")
    except Exception as e:
        print(f"Error occurred while handling the data: {e}")

def download_nchlt():
    """
    Downloads data required for this project. 
    Also checks if data is already downloaded
    """
    change_pwd()
    DATA_PATH = os.path.join("downloaded")
    AF_PATH = os.path.join(DATA_PATH, "nchlt_afr")
    XH_PATH = os.path.join(DATA_PATH, "nchlt_xho")

    af_url = "http://ftp.internat.freebsd.org/pub/nchlt/Speech_corpora/nchlt_afr.tar.gz"
    af_file_name = os.path.join(AF_PATH, "nchlt_afr.tar.gz")
    xh_url = "http://ftp.internat.freebsd.org/pub/nchlt/Speech_corpora/nchlt_xho.tar.gz"
    xh_file_name = os.path.join(XH_PATH, "nchlt_xho.tar.gz")

    # Download and extract the datasets
    try:
        download_file(af_url, af_file_name)
        download_file(xh_url, xh_file_name)
        extract_tar_file(af_file_name, AF_PATH)
        extract_tar_file(xh_file_name, XH_PATH)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while handling the data: {e}")
    except Exception as e:
        print(f"Error occurred while handling the data: {e}")
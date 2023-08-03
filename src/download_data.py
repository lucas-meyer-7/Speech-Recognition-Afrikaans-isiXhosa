import os
import tarfile
import requests

from tqdm import tqdm
from os.path import join as p_join

def download_file(url, file_name):
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
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)


if __name__ == "__main__":
    data_path = p_join("data", "high-quality-tts-data")
    
    # Check if data is already downloaded
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    else:
        if (os.path.isdir(p_join(data_path, "af_za"))) and (os.path.isdir(p_join(data_path, "xh_za"))) :
            print("The data has already been downloaded.")
            exit(0)

    af_url = "https://repo.sadilar.org/bitstream/handle/20.500.12185/527/af_za.tar.gz"
    af_file_name = p_join(data_path, "af_za.tar.gz")
    xh_url = "https://repo.sadilar.org/bitstream/handle/20.500.12185/527/xh_za.tar.gz"
    xh_file_name = p_join(data_path, "xh_za.tar.gz")

    # Download and extract the datasets
    try:
        download_file(af_url, af_file_name)
        download_file(xh_url, xh_file_name)
        print("Extracting (un-compressing) data ...", end="")
        extract_tar_file(af_file_name, data_path)
        extract_tar_file(xh_file_name, data_path)
        print("\rData downloaded and extracted successfully!")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while handling the data: {e}")
    except Exception as e:
        print(f"Error occurred while handling the data: {e}")

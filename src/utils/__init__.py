import os
import requests
from tqdm import tqdm
import re
from urllib.parse import urlencode
from src.constants import DIRECT_URL

def download_weights(url, save_path):
    """
    Download pre-trained model weights from a Google Drive URL and save them locally.
    
    Args:
        url (str): Direct URL to download the weights from.
        save_path (str): Local path to save the weights.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"Weights already exist at {save_path}")
        return

    try:
        session = requests.Session()
        response = session.get(url, allow_redirects=True)

        if "Virus scan warning" in response.text or "Download anyway" in response.text:
            confirm_match = re.search(r'<input type="hidden" name="confirm" value="([^"]+)"', response.text)
            uuid_match = re.search(r'<input type="hidden" name="uuid" value="([^"]+)"', response.text)
            id_match = re.search(r'<input type="hidden" name="id" value="([^"]+)"', response.text)
            export_match = re.search(r'<input type="hidden" name="export" value="([^"]+)"', response.text)
            authuser_match = re.search(r'<input type="hidden" name="authuser" value="([^"]+)"', response.text)

            if confirm_match and uuid_match:
                form_data = {
                    "id": id_match.group(1) if id_match else "1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF",
                    "export": export_match.group(1) if export_match else "download",
                    "authuser": authuser_match.group(1) if authuser_match else "0",
                    "confirm": confirm_match.group(1),
                    "uuid": uuid_match.group(1)
                }
                print(f"Extracted form parameters: {form_data}")
                
                form_action = "https://drive.usercontent.google.com/download"
                download_url = f"{form_action}?{urlencode(form_data)}"

                response = session.get(download_url, stream=True, allow_redirects=True)
            else:
                print("Could not extract required form parameters.")
                print(f"Confirm: {confirm_match.group(1) if confirm_match else 'Not found'}")
                print(f"UUID: {uuid_match.group(1) if uuid_match else 'Not found'}")
                return

        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            if total_size == 0:
                print("Warning: Content-length not provided, progress bar may not be accurate.")

            with open(save_path, "wb") as file, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

            print(f"Weights downloaded successfully and saved at {save_path}")
        else:
            print(f"Failed to download weights. Status code: {response.status_code}")
            print(f"Response content (first 500 chars): {response.text[:500]}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def download_weights_from_google_drive():
    # Google Drive download URL
    direct_url = DIRECT_URL 
    # Determine the project root from src/utils/__init__.py
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))

    # Target weights directory and file path
    weights_dir = os.path.join(project_root, "weights")
    weights_filename = "inswapper_128.onnx"
    save_path = os.path.join(weights_dir, weights_filename)

    # Attempt download
    print(f"Attempting to download from: {direct_url}")
    print(f"Saving to: {save_path}")
    download_weights(direct_url, save_path)

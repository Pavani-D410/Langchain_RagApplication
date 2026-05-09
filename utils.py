import os

from datetime import datetime

def save_uploaded_file(uploaded_file):

    upload_folder = "uploads"

    os.makedirs(
        upload_folder,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d%H%M%S"
    )

    filename = (
        f"{timestamp}_{uploaded_file.name}"
    )

    file_path = os.path.join(
        upload_folder,
        filename
    )

    with open(file_path, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )

    return file_path
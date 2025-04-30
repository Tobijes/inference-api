import os
from pathlib import Path
import tempfile

import aiofiles
from fastapi import UploadFile

class Storage():
    directory: str = Path('/dev/shm/inference_uploaded_data')

    def __init__(self) -> None:
        # Create RAM disk directory
        self.directory.mkdir(exist_ok=True) 
    
    async def save_temporary_media(self, to_save: UploadFile | list[UploadFile]) -> list[Path]:
        """
        Saves (copies) input files to temporary files in RAM directory using 'tmpfs' in '/dev/shm'.

        Note: Files are not automatically deleted, but should be manually deleted after use, using delete_temporary_media()
        """
        if not isinstance(to_save, list):
            to_save = [to_save]

        file_paths = []
        for file_to_save in to_save:
            # Create temporary file
            temporary_file = tempfile.NamedTemporaryFile(delete=False, dir=self.directory)
            temporary_file_path = Path(temporary_file.name)
            temporary_file.close()

            # Write input file content to temporary file
            async with aiofiles.open(temporary_file_path, 'wb') as out_file:
                while content := await file_to_save.read(1024):  # async read chunk
                    await out_file.write(content)  # async write chunk
                file_paths.append(temporary_file_path)

        return file_paths
    
    def delete_temporary_media(self, paths: Path | list[Path]):
        """Delete specified temporary files"""
        if not isinstance(paths, list):
            paths = [paths]

        for path in paths:
            os.remove(path)
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time
import spikingjelly.datasets as sjds
from spikingjelly.datasets import np_savez
from typing import Callable, Optional, Tuple
import numpy as np
from dv import AedatFile
import os


class DVSDataset(sjds.NeuromorphicDatasetFolder):
    def __init__(
        self,
        root: str,
        train: Optional[bool] = None,
        data_type: Optional[str] = "event",
        frames_number: Optional[int] = None,
        split_by: Optional[str] = None,
        duration: Optional[int] = None,
        custom_integrate_function: Optional[Callable] = None,
        custom_integrated_frames_dir_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        assert train is not None
        super().__init__(
            root,
            train,
            data_type,
            frames_number,
            split_by,
            duration,
            custom_integrate_function,
            custom_integrated_frames_dir_name,
            transform,
            target_transform,
        )

    @staticmethod
    def resource_url_md5() -> list:
        """
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        """
        return []

    @staticmethod
    def load_event_data(file_name: str):
        """
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin binary data.
        """
        with AedatFile(file_name) as f:
            events = np.hstack([event for event in f["events"].numpy()])
        return events

    @staticmethod
    def convert_aedat_files_to_np(fname: str, aedat_file: str, output_dir: str):
        if not os.path.exists(aedat_file):
            pass
            # print(f"Error: The file {aedat_file} does not exist.")
        else:
            print(f"Start to convert [{aedat_file}] to samples.")
            events = DVSDataset.load_event_data(aedat_file)

            file_name = os.path.join(output_dir, f"{fname.split('.')[0]}.npz")
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            np_savez(file_name, t=events["timestamp"], x=events["x"], y=events["y"], p=events["polarity"])
            print(f"[{file_name}] saved.")

    @staticmethod
    def downloadable() -> bool:
        """
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        """
        return False

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        aedat_dir = extract_root
        train_dir = os.path.join(events_np_root, "train")
        test_dir = os.path.join(events_np_root, "test")
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        print(f"Mkdir [{train_dir, test_dir}].")
        # for label in range(CLASS_NUM):
        #     os.mkdir(os.path.join(train_dir, str(label)))
        #     os.mkdir(os.path.join(test_dir, str(label)))
        # print(f"Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].")
        with (
            open(os.path.join(aedat_dir, "train.txt")) as train_txt,
            open(os.path.join(aedat_dir, "test.txt")) as test_txt,
        ):
            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 16)) as tpe:
                sub_threads = []
                print(f"Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].")

                for line in train_txt.readlines():
                    fname = line.strip().split(" ")[0]
                    print(fname)
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        sub_threads.append(
                            tpe.submit(DVSDataset.convert_aedat_files_to_np, fname, aedat_file, train_dir)
                        )

                for line in test_txt.readlines():
                    fname = line.strip().split(" ")[0]
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        sub_threads.append(
                            tpe.submit(DVSDataset.convert_aedat_files_to_np, fname, aedat_file, test_dir)
                        )
                print(f"Used time = [{round(time.time() - t_ckp, 2)}s].")
            print(f"All aedat files have been split to samples and saved into [{train_dir, test_dir}].")

    @staticmethod
    def get_H_W() -> Tuple:
        """
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        """
        return 260, 346


# if __name__ == "__main__":
#     data_dir = "/home/zdm/dataset/PAF_filter/"  
#     train_set = DVSDataset(root=data_dir, train=True, data_type="frame", frames_number=16, split_by="number")
#     test_set = DVSDataset(root=data_dir, train=False, data_type="frame", frames_number=16, split_by="number")
    # print(train_set.class_to_idx)
    
import torchio as tio
import numpy as np

class BaseDataset:
    """
    Datasets must inherit from this class, which through get() returns the
    appropriate dataset split.
    """

    def __init__(self):
        pass

    def get(self, split: str) -> tio.SubjectsDataset:
        """
        Return the appropriate dataset split.

        Args:
          `split`: 'train' or 'validation'

        Returns:
          Dataset split.
        """
        # No subjects
        if not self.subjects_dict[split]:
            return []

        sd = tio.SubjectsDataset(self.subjects_dict[split],
                transform=self.transforms_dict[split])

        if split == "train":
            # UnformSampler: Randomly extract patches from a volume
            #                with uniform probability.
            # So, my "slices" are sometimes the same, but it's fine because
            # during the entire training, all slices will likely be seen.
            if not hasattr(self, "sampler"):
                self.sampler = tio.data.UniformSampler(
                        patch_size=sd[0].info["patch_size"])

            # Volumes are random because of the UniformSampler
            # Subjects are random because I randomize it
            queue = tio.Queue(sd, max_length=50, shuffle_patches=True,
                    samples_per_volume=[x.info["slices"] for x in sd],
                    sampler=self.sampler, num_workers=6, shuffle_subjects=True)
            queue.dataset = self
            return queue
        elif split == "validation" or split == "test":
            sd.dataset = self
            return sd
        else:
            raise ValueError(f"Unknown split `{split}`")

    @staticmethod
    def pre_verify(inputFolder: str) -> bool:
        raise NotImplementedError("")

    @staticmethod
    def pre_process(inputFolder: str, outputFolder: str, _=None) -> None:
        raise NotImplementedError("")

    @staticmethod
    def post_verify(inputFolder: str) -> bool:
        raise NotImplementedError("")

    @staticmethod
    def post_process(inputFolder: str, outputFolder: str, original: str) -> None:
        raise NotImplementedError("")


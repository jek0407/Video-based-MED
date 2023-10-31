import cv2
import numpy as np

from pyturbo import Stage, Task


class SIFTFeature(Stage):

    """
    Input: batch of frames [B x H x W x C]
    Output: yield SIFT features of each frame, each as [N x D]
    """

    def allocate_resource(self, resources, *, num_features=32):
        self.num_features = num_features
        self.sift = None
        return [resources]

    def reset(self):
        if self.sift is None:
            self.sift = cv2.SIFT_create(self.num_features)

    def extract_sift_feature(self, frame: np.ndarray) -> np.ndarray:
        """
        frame: [H x W x C]

        Return: Feature for N key points, [N x 128]
        """
        # TODO: Extract SIFT feature for the current frame
        # Use self.sift.detectAndCompute
        # Remember to handle when it returns None
            # Ensure the frame is in grayscale as SIFT requires it
        if frame.shape[-1] == 3:  # Check if the image has 3 channels
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame

        # Use self.sift.detectAndCompute
        keypoints, descriptors = self.sift.detectAndCompute(gray_frame, None)

        # Remember to handle when it returns None
        if descriptors is None:
            # Handle case with no descriptors, you might adapt this part as per your use-case
            descriptors = np.empty((0, 128), dtype=np.float32)
        elif descriptors.shape[0] > self.num_features:
            # Optionally, down-sample descriptors if they exceed num_features
            chosen_indices = np.random.choice(
                descriptors.shape[0], self.num_features, replace=False)
            descriptors = descriptors[chosen_indices]

        return descriptors
        # raise NotImplementedError

    def process(self, task):
        task.start(self)
        frames = task.content
        frame_ids = task.meta['frame_ids']
        for frame_id, frame in zip(frame_ids, frames):
            sub_task = Task(meta={'sequence_id': frame_id},
                            parent_task=task).start(self)
            feature = self.extract_sift_feature(frame.numpy())
            assert feature is not None and isinstance(feature, np.ndarray)
            assert feature.shape[1] == 128
            yield sub_task.finish(feature)
        task.finish()

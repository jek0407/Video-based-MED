import numpy as np
import pickle
from pyturbo import Stage
from scipy.spatial.distance import cdist

class BagOfWords(Stage):

    """
    Input: features [N x D]
    Output: bag-of-words [W]
    """

    def allocate_resource(self, resources, *, weight_path):
        self.weight_path = weight_path
        self.weight = None
        return [resources]

    def reset(self):
        if self.weight is None:
            with open(self.weight_path, 'rb') as f:
                self.weight = pickle.load(f)

    def get_bag_of_words(self, features: np.ndarray) -> np.ndarray:
        """
        features: [N x D]

        Return: count of each word, [W]
        """
        # TODO: Generate bag of words
        # Calculate pairwise distance between each feature and each cluster,
        # assign each feature to the nearest cluster, and count
        # Calculate pairwise distances between features and cluster centers (words).
        distances = cdist(features, self.weight)
        
        # Assign each feature to the closest cluster.
        word_assignments = np.argmin(distances, axis=1)
        
        # Count the number of assignments to each cluster.
        bag_of_words = np.bincount(word_assignments, minlength=self.weight.shape[0])
        
        return bag_of_words
    def get_video_feature(self, bags: np.ndarray) -> np.ndarray:
        """
        bags: [B x W]

        Return: pooled vector, [W]
        """
        # TODO: Aggregate frame-level bags into a video-level feature.
        # 1. Sum-pooling: Sum the BoW histograms across all frames to get a video-level histogram.
        video_bag = np.sum(bags, axis=0)

        return video_bag

    def process(self, task):
        features = task.content
        bags = []
        for frame_features in features:
            bag = self.get_bag_of_words(frame_features)
            assert isinstance(bag, np.ndarray)
            assert bag.shape == self.weight.shape[:1]
            bags.append(bag)
        bags = np.stack(bags)
        video_bag = self.get_video_feature(bags)
        assert isinstance(video_bag, np.ndarray)
        assert video_bag.shape == self.weight.shape[:1]
        return task.finish(video_bag)

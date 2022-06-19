import unittest





class TestDepthInference(unittest.TestCase):

    def test_models(self):
        """
        Check if the pretrained models ('nyu.h5','kitti.h5') are in the project(DenseDepth) folder.
        """
        #list_files = os.listdir()
        kitty_file_name = 'kitti1.h5'
        nyu_file_name = 'nyu.h5'
        self.assertIn(kitty_file_name,"hello")
        self.assertIn(nyu_file_name,"hello")

    def test_resize32(self):
        """

        Returns:

        """


# if __name__ == '__main__':
#     unittest.main()
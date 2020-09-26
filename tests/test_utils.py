from numpy import array
from pandas import DataFrame
from pandas.testing import assert_frame_equal
import pytest

from dvt.aggregate.length import ShotLengthAggregator
from dvt.core import DataExtraction, FrameInput
from dvt.utils import process_output_values, sub_image


class TestSubImage:

    def test_sub_image(self, test_img):

        simg = sub_image(test_img, 0, 128, 64, 0)
        assert simg.shape == (64, 128, 3)

    def test_sub_image_fct(self, test_img):

        simg = sub_image(test_img, 0, 128, 64, 0, fct=1.5)
        assert simg.shape == (80, 160, 3)

    def test_sub_image_reshape(self, test_img):

        simg = sub_image(test_img, 32, 64, 64, 32, output_shape=(100, 100))
        assert simg.shape == (100, 100, 3)


class TestProcessOutput:

    def test_dataframe_input(self):
        input = DataFrame({'a': [1, 2, 3]})
        output = process_output_values(input)

        assert type(output) == list
        assert len(output) == 1
        assert output == [input]

    def test_none_input(self):
        input = None
        output = process_output_values(input)

        assert output == input

    def test_list_input(self):
        input = [1, 2, 3]
        output = process_output_values(input)

        assert type(output) == list
        assert len(output) == 3
        assert output == input

    def test_dict_input(self):
        input = {'a': [1, 2, 3], 'b': [3, 4, 5]}
        output = process_output_values(input)

        assert type(output) == list
        assert len(output) == 1
        assert_frame_equal(output[0], DataFrame(input))

    def test_np_1d_input(self):
        input = {'a': [1, 2, 3], 'b': array([3, 4, 5])}
        output = process_output_values(input)

        assert type(output) == list
        assert len(output) == 1
        assert_frame_equal(output[0], DataFrame(input))

    def test_np_2d_input(self):
        input = {'a': [1, 2, 3], 'b': array([[3, 4], [5, 5], [7, 6]])}
        output = process_output_values(input)

        expected_output = DataFrame({
            'a': [1, 2, 3],
            'b': [array([3, 4]), array([5, 5]), array([7, 6])]
        })

        assert type(output) == list
        assert len(output) == 1
        assert_frame_equal(output[0], expected_output)

    def test_1row_input(self):
        input = {'a': 1, 'b': 1}
        output = process_output_values(input)

        assert type(output) == list
        assert len(output) == 1
        assert_frame_equal(output[0], DataFrame(input, index=[0]))


class TestMisc:

    def test_bad_key(self):

        de = DataExtraction(FrameInput(
            input_path="test-data/video-clip.mp4", bsize=256
        ))

        with pytest.raises(KeyError) as e_info:
            de.run_aggregator(ShotLengthAggregator())


if __name__ == "__main__":
    pytest.main([__file__])

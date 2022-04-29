import os
import pytest
import freesolo.unsupervised_learning

@pytest.fixture
def raw_test_data():
	test_df = pandas.DataFrame(columns=['count_example',
										'z_score_example'],
							   data = [])
	yield data
	os.remove(file_path)

def test_for_zscore(raw_test_data):

	usl = unsupervised_learning()
	assert generate_standard_vars()

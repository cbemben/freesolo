import os
import pandas
import pytest
from freesolo import unsupervised_learning as us

@pytest.fixture
def raw_test_data():
	data = pandas.DataFrame({'count_example': [4,5,3,7,5,12],
								'z_score_example': [-0.6201736729, -0.3100868365,
								                    -0.9302605094, 0.3100868365,
								                    -0.3100868365, 1.860521019]})
	yield data

def test_standardize_vars(raw_test_data):
	usl = us.UnsupervisedLearning(data=raw_test_data,
		                        target_features=['count_example'])
	usl.generate_standard_vars()
	output = round(usl._ads_data['count_example_zscore'],5).tolist()
	expected = round(raw_test_data['z_score_example'],5).tolist()
	assert output == expected

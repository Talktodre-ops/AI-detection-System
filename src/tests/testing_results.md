(AI-Detection-311) C:\Users\VICTUS\Documents\GITHUB\AI-detection-System\src>python -m pytest tests/ --cov=src
========================================================== test session starts ==========================================================
platform win32 -- Python 3.11.8, pytest-8.3.5, pluggy-1.5.0
rootdir: C:\Users\VICTUS\Documents\GITHUB\AI-detection-System\src\tests
configfile: pytest.ini
plugins: anyio-4.9.0, cov-6.1.1
collected 19 items / 1 deselected / 18 selected

tests\test_api.py::test_flag_prediction_endpoint
------------------------------------------------------------- live log call ------------------------------------------------------------- 
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/flag_prediction/ "HTTP/1.1 200 OK"
PASSED                                                                                                                             [  5%] 
tests\test_api.py::test_root_endpoint
------------------------------------------------------------- live log call ------------------------------------------------------------- 
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/ "HTTP/1.1 200 OK"
PASSED                                                                                                                             [ 11%] 
tests\test_api.py::test_predict_endpoint
------------------------------------------------------------- live log call ------------------------------------------------------------- 
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/predict/ "HTTP/1.1 200 OK"
PASSED                                                                                                                             [ 16%] 
tests\test_data_processing.py::test_data_loading PASSED                                                                            [ 22%]
tests\test_data_processing_extended.py::test_preprocess_text PASSED                                                                [ 27%] 
tests\test_data_processing_extended.py::test_clean_dataset
------------------------------------------------------------- live log call ------------------------------------------------------------- 
INFO     root:clean_dataset.py:25 Cleaning dataset: temp_test_clean.csv
INFO     root:file_utils.py:37 Successfully loaded temp_test_clean.csv with 4 rows
INFO     root:clean_dataset.py:38 Original dataset shape: (4, 2)
INFO     root:file_utils.py:67 Successfully saved 4 rows to temp_test_clean_output.csv
INFO     root:clean_dataset.py:50 Cleaned dataset saved to: temp_test_clean_output.csv
PASSED                                                                                                                             [ 33%] 
tests\test_data_processing_extended.py::test_clean_dataset_with_missing_values
------------------------------------------------------------- live log call ------------------------------------------------------------- 
INFO     root:clean_dataset.py:25 Cleaning dataset: temp_test_missing.csv
INFO     root:file_utils.py:37 Successfully loaded temp_test_missing.csv with 4 rows
INFO     root:clean_dataset.py:38 Original dataset shape: (4, 2)
INFO     root:file_utils.py:67 Successfully saved 4 rows to temp_test_missing_output.csv
INFO     root:clean_dataset.py:50 Cleaned dataset saved to: temp_test_missing_output.csv
PASSED                                                                                                                             [ 38%]
tests\test_frontend.py::test_read_file_function PASSED                                                                             [ 44%] 
tests\test_frontend.py::test_predict_proba_function PASSED                                                                         [ 50%] 
tests\test_frontend_extended.py::test_file_processing_functions SKIPPED (Could not import file processing functions from front...) [ 55%]
tests\test_model.py::test_model_loading PASSED                                                                                     [ 61%]
tests\test_model.py::test_model_prediction PASSED                                                                                  [ 66%]
tests\test_model.py::test_model_with_various_inputs[The utilization of advanced algorithms enables the system to process vast amounts of data efficiently.-1] PASSED [ 72%]
tests\test_model.py::test_model_with_various_inputs[I'm not sure if this will work, but let's give it a try! Maybe we'll get lucky :)-1]PPASSED [ 77%]
tests\test_model.py::test_model_with_various_inputs[-0] PASSED                                                                     [ 83%]
tests\test_model.py::test_model_with_various_inputs[Lorem ipsum dolor sit amet, consectetur adipiscing elit.-0] PASSED             [ 88%]
tests\test_model_evaluation.py::test_calculate_metrics PASSED                                                                      [ 94%] 
tests\test_model_evaluation.py::test_evaluate_model_function PASSED                                                                [100%]

=========================================================== warnings summary ============================================================ 
test_frontend_extended.py::test_file_processing_functions
  C:\Users\VICTUS\AppData\Local\Programs\Python\Python311\Lib\site-packages\PyPDF2\__init__.py:21: DeprecationWarning: PyPDF2 is deprecated. Please move to the pypdf library instead.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================================================ tests coverage ============================================================= 
____________________________________________ coverage: platform win32, python 3.11.8-final-0 ____________________________________________ 

Name                                     Stmts   Miss  Cover
------------------------------------------------------------
__init__.py                                  0      0   100%
api\__init__.py                              0      0   100%
api\main.py                                 26     26     0%
data\__init__.py                             0      0   100%
data\clean_dataset.py                       36     36     0%
data\config.py                              21     21     0%
data\inspect_test_data.py                   35     35     0%
data\label_data.py                          17     17     0%
data\preprocess_data.py                     17     17     0%
data\preprocessing.py                       18     18     0%
data\split_data.py                          24     24     0%
frontend\__init__.py                         0      0   100%
frontend\ai_detector_gpu.py                103    103     0%
frontend\ai_detector_gpu.py                103    103     0%
frontend\app.py                             99     99     0%
model\__init__.py                            0      0   100%
model\analyze_misclassifications.py         63     63     0%
model\analyze_results.py                    75     75     0%
model\evaluate_model.py                     54     54     0%
model\explainability.py                      0      0   100%
model\retrain_model.py                      44     44     0%
frontend\ai_detector_gpu.py                103    103     0%
frontend\app.py                             99     99     0%
model\__init__.py                            0      0   100%
model\analyze_misclassifications.py         63     63     0%
model\analyze_results.py                    75     75     0%
model\evaluate_model.py                     54     54     0%
model\explainability.py                      0      0   100%
model\retrain_model.py                      44     44     0%
model\analyze_misclassifications.py         63     63     0%
model\analyze_results.py                    75     75     0%
model\evaluate_model.py                     54     54     0%
model\explainability.py                      0      0   100%
model\retrain_model.py                      44     44     0%
model\evaluate_model.py                     54     54     0%
model\explainability.py                      0      0   100%
model\retrain_model.py                      44     44     0%
model\explainability.py                      0      0   100%
model\retrain_model.py                      44     44     0%
model\retrain_model.py                      44     44     0%
model\roberta_model.py                      52     52     0%
tests\__init__.py                            0      0   100%
tests\conftest.py                            6      0   100%
tests\test_api.py                           25      0   100%
tests\test_data_processing.py               30      5    83%
tests\test_data_processing_extended.py      71      0   100%
tests\test_frontend.py                      32      0   100%
tests\test_frontend_extended.py             34     23    32%
tests\test_model.py                         69     15    78%
tests\test_model_evaluation.py              39      0   100%
utils\__init__.py                            0      0   100%
utils\file_utils.py                         31     31     0%
------------------------------------------------------------
TOTAL                                     1021    758    26%
======================================== 17 passed, 1 skipped, 1 deselected, 1 warning in 17.64s ========================================

(AI-Detection-311) C:\Users\VICTUS\Documents\GITHUB\AI-detection-System\src>
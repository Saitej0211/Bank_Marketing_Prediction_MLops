Folder PATH listing for volume Windows
Volume serial number is 6E73-904C
D:.
ª   .DS_Store
ª   .dvcignore
ª   .env
ª   .gitattributes
ª   .gitignore
ª   app_test.py
ª   directory_structure.md
ª   directory_structure.txt
ª   docker-compose.yaml
ª   Dockerfile
ª   filebeat.yml
ª   LICENSE
ª   metricbeat.yml
ª   mlflow.db
ª   README.md
ª   requirements.txt
ª   vm_ssh
ª   vm_ssh.pub
ª   
+---.dvc
ª       .gitignore
ª       config
ª       
+---.github
ª   +---workflows
ª           CI_CD_gcp_deploy.yml
ª           deploy.yml
ª           model_pipe.yml
ª           unittest.yml
ª           
+---assets
ª   +---images
ª   ª       airflow.jpeg
ª   ª       
ª   +---plots
ª           age_outlier_handling.png
ª           campaign_outlier_handling.png
ª           cons.conf.idx_outlier_handling.png
ª           cons.price.idx_outlier_handling.png
ª           duration_outlier_handling.png
ª           emp.var.rate_outlier_handling.png
ª           euribor3m_outlier_handling.png
ª           nr.employed_outlier_handling.png
ª           pdays_outlier_handling.png
ª           previous_outlier_handling.png
ª           
+---config
ª       config.yaml
ª       constants.py
ª       Key.json
ª       Key_old.json
ª       schema.yaml
ª       
+---dags
ª   ª   .DS_Store
ª   ª   __init__.py
ª   ª   
ª   +---logs
ª   ª       anomaly_detection.log
ª   ª       correlation_analysis.log
ª   ª       data_schema_statistics_generation.log
ª   ª       eda.log
ª   ª       encoding.log
ª   ª       load_data.log
ª   ª       process_data.log
ª   ª       smote_analysis.log
ª   ª       
ª   +---src
ª       ª   .DS_Store
ª       ª   data_pipeline.py
ª       ª   DownloadData.py
ª       ª   eda.py
ª       ª   HandlingNullValues.py
ª       ª   LoadData.py
ª       ª   model_development_pipeline.py
ª       ª   util_bank_marketing.py
ª       ª   __init__.py
ª       ª   
ª       +---bias_analysis_output
ª       ª       bias_age_accuracy.png
ª       ª       bias_marital_accuracy.png
ª       ª       
ª       +---dags
ª       ª   +---logs
ª       ª           compare_best_models.log
ª       ª           push_to_gcp.log
ª       ª           
ª       +---data_preprocessing
ª       ª   ª   correlation_analysis.py
ª       ª   ª   datatype_format.py
ª       ª   ª   encoding.py
ª       ª   ª   outlier_handing.py
ª       ª   ª   preprocessing_main.py
ª       ª   ª   removing_duplicates.py
ª       ª   ª   smote.py
ª       ª   ª   __init__.py
ª       ª   ª   
ª       ª   +---__pycache__
ª       ª           correlation_analysis.cpython-312.pyc
ª       ª           correlation_analysis.cpython-38.pyc
ª       ª           datatype_format.cpython-312.pyc
ª       ª           datatype_format.cpython-38.pyc
ª       ª           encoding.cpython-312.pyc
ª       ª           encoding.cpython-38.pyc
ª       ª           HandlingNullValues.cpython-312.pyc
ª       ª           outlier_handing.cpython-312.pyc
ª       ª           outlier_handing.cpython-38.pyc
ª       ª           preprocessing_main.cpython-312.pyc
ª       ª           preprocessing_main.cpython-38.pyc
ª       ª           smote.cpython-312.pyc
ª       ª           smote.cpython-38.pyc
ª       ª           __init__.cpython-312.pyc
ª       ª           __init__.cpython-38.pyc
ª       ª           
ª       +---Data_validation
ª       ª   ª   anomaly_detection.py
ª       ª   ª   data_schema_statistics_generation.py
ª       ª   ª   util_bank_marketing.py
ª       ª   ª   
ª       ª   +---__pycache__
ª       ª           anomaly_detection.cpython-38.pyc
ª       ª           anomaly_detection_alerts.cpython-38.pyc
ª       ª           data_schema_statistics_generation.cpython-38.pyc
ª       ª           
ª       +---final_model
ª       ª       best_model.json
ª       ª       best_model.pkl
ª       ª       random_forest_20241123-073854.pkl
ª       ª       random_forest_20241123-073854_X_test.csv
ª       ª       random_forest_20241123-073854_y_test.csv
ª       ª       random_forest_20241123-081315.pkl
ª       ª       random_forest_20241123-081315_X_test.csv
ª       ª       random_forest_20241123-081315_y_test.csv
ª       ª       random_forest_20241123-103534.pkl
ª       ª       random_forest_20241123-103534_X_test.csv
ª       ª       random_forest_20241123-103534_y_test.csv
ª       ª       random_forest_20241123-104910.pkl
ª       ª       random_forest_20241123-104910_X_test.csv
ª       ª       random_forest_20241123-104910_y_test.csv
ª       ª       results_20241123-073854.json
ª       ª       results_20241123-081315.json
ª       ª       results_20241123-103534.json
ª       ª       results_20241123-104910.json
ª       ª       
ª       +---logs
ª       ª       ml_metrics.log
ª       ª       
ª       +---Model_Pipeline
ª       ª   ª   compare_best_models.py
ª       ª   ª   model_bias_detection.py
ª       ª   ª   model_development_and_evaluation_with_mlflow.py
ª       ª   ª   push_to_gcp.py
ª       ª   ª   sensitivity_analysis.py
ª       ª   ª   
ª       ª   +---__pycache__
ª       ª           compare_best_models.cpython-38.pyc
ª       ª           model_bias_detection.cpython-38.pyc
ª       ª           model_development_and_evaluation_with_mlflow.cpython-38.pyc
ª       ª           push_to_gcp.cpython-38.pyc
ª       ª           sensitivity_analysis.cpython-38.pyc
ª       ª           
ª       +---__pycache__
ª               airflow.cpython-312.pyc
ª               airflow.cpython-38.pyc
ª               airflow_dag2_model.cpython-38.pyc
ª               data_pipeline.cpython-38.pyc
ª               DownloadData.cpython-312.pyc
ª               DownloadData.cpython-38.pyc
ª               eda.cpython-312.pyc
ª               eda.cpython-38.pyc
ª               HandlingNullValues.cpython-312.pyc
ª               HandlingNullValues.cpython-38.pyc
ª               LoadData.cpython-312.pyc
ª               LoadData.cpython-38.pyc
ª               model_development_pipeline.cpython-38.pyc
ª               model_development_with_mlflow.cpython-38.pyc
ª               __init__.cpython-312.pyc
ª               __init__.cpython-38.pyc
ª               
+---data
ª   +---processed
ª   ª   ª   contact_label_encoder.pkl
ª   ª   ª   correlation_matrix.csv
ª   ª   ª   correlation_matrix.pkl
ª   ª   ª   dataframe_description.csv
ª   ª   ª   dataframe_info.csv
ª   ª   ª   datatype_format_processed.pkl
ª   ª   ª   datatype_info_after.csv
ª   ª   ª   datatype_info_before.csv
ª   ª   ª   default_label_encoder.pkl
ª   ª   ª   education_label_encoder.pkl
ª   ª   ª   encoded_data.csv
ª   ª   ª   encoded_data.pkl
ª   ª   ª   housing_label_encoder.pkl
ª   ª   ª   job_label_encoder.pkl
ª   ª   ª   loan_label_encoder.pkl
ª   ª   ª   marital_label_encoder.pkl
ª   ª   ª   month_label_encoder.pkl
ª   ª   ª   normalizer.pkl
ª   ª   ª   outlier_handled_data.pkl
ª   ª   ª   processed_data.csv
ª   ª   ª   processed_data.pkl
ª   ª   ª   raw_data.csv
ª   ª   ª   raw_data.pkl
ª   ª   ª   scaler.pkl
ª   ª   ª   schema.pbtxt
ª   ª   ª   schema_serving.json
ª   ª   ª   schema_training.json
ª   ª   ª   smote_resampled_train_data.csv
ª   ª   ª   smote_resampled_train_data.pkl
ª   ª   ª   test_data.csv
ª   ª   ª   test_data.pkl
ª   ª   ª   validate.csv
ª   ª   ª   validate.pkl
ª   ª   ª   validate_process.csv
ª   ª   ª   validate_process.json
ª   ª   ª   validate_process.pkl
ª   ª   ª   y_label_encoder.pkl
ª   ª   ª   
ª   ª   +---eda_plots
ª   ª           age_distribution.png
ª   ª           contact_method_distribution.png
ª   ª           correlation_heatmap.png
ª   ª           default_status_distribution.png
ª   ª           deposit_distribution.png
ª   ª           education_distribution.png
ª   ª           housing_loan_distribution.png
ª   ª           job_distribution.png
ª   ª           marital_status_distribution.png
ª   ª           month_distribution.png
ª   ª           personal_loan_distribution.png
ª   ª           
ª   +---raw
ª       ª   .DS_Store
ª       ª   
ª       +---bank
ª       ª       .gitignore
ª       ª       bank-full.csv.dvc
ª       ª       bank-names.txt
ª       ª       bank.csv
ª       ª       
ª       +---bank-additional
ª               .DS_Store
ª               .Rhistory
ª               bank-additional-names.txt
ª               bank-additional.csv
ª               
+---gcpdeploy
ª   ª   app.py
ª   ª   app_test.py
ª   ª   credentials.txt
ª   ª   Dockerfile
ª   ª   load_test.py
ª   ª   requirements.txt
ª   ª   sampleCheck.py
ª   ª   samplecheck1.ipynb
ª   ª   setup-CI_CD.sh
ª   ª   setup.sh
ª   ª   startup-script-CI_CD.sh
ª   ª   startup-script.sh
ª   ª   test.sh
ª   ª   
ª   +---myenv
ª   ª   ª   pyvenv.cfg
ª   ª   ª   
ª   ª   +---Lib
ª   ª   ª   +---site-packages
ª   ª   ª       ª   easy_install.py
ª   ª   ª       ª   six.py
ª   ª   ª       ª   threadpoolctl.py
ª   ª   ª       ª   
ª   ª   ª       +---blinker
ª   ª   ª       ª       base.py
ª   ª   ª       ª       py.typed
ª   ª   ª       ª       _utilities.py
ª   ª   ª       ª       __init__.py
ª   ª   ª       ª       
ª   ª   ª       +---blinker-1.9.0.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---cachetools
ª   ª   ª       ª       func.py
ª   ª   ª       ª       keys.py
ª   ª   ª       ª       __init__.py
ª   ª   ª       ª       
ª   ª   ª       +---cachetools-5.5.0.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---certifi
ª   ª   ª       ª       cacert.pem
ª   ª   ª       ª       core.py
ª   ª   ª       ª       py.typed
ª   ª   ª       ª       __init__.py
ª   ª   ª       ª       __main__.py
ª   ª   ª       ª       
ª   ª   ª       +---certifi-2024.8.30.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---charset_normalizer
ª   ª   ª       ª   ª   api.py
ª   ª   ª       ª   ª   cd.py
ª   ª   ª       ª   ª   constant.py
ª   ª   ª       ª   ª   legacy.py
ª   ª   ª       ª   ª   md.cp39-win32.pyd
ª   ª   ª       ª   ª   md.py
ª   ª   ª       ª   ª   md__mypyc.cp39-win32.pyd
ª   ª   ª       ª   ª   models.py
ª   ª   ª       ª   ª   py.typed
ª   ª   ª       ª   ª   utils.py
ª   ª   ª       ª   ª   version.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   __main__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---cli
ª   ª   ª       ª           __init__.py
ª   ª   ª       ª           __main__.py
ª   ª   ª       ª           
ª   ª   ª       +---charset_normalizer-3.4.0.dist-info
ª   ª   ª       ª       entry_points.txt
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---click
ª   ª   ª       ª       core.py
ª   ª   ª       ª       decorators.py
ª   ª   ª       ª       exceptions.py
ª   ª   ª       ª       formatting.py
ª   ª   ª       ª       globals.py
ª   ª   ª       ª       parser.py
ª   ª   ª       ª       py.typed
ª   ª   ª       ª       shell_completion.py
ª   ª   ª       ª       termui.py
ª   ª   ª       ª       testing.py
ª   ª   ª       ª       types.py
ª   ª   ª       ª       utils.py
ª   ª   ª       ª       _compat.py
ª   ª   ª       ª       _termui_impl.py
ª   ª   ª       ª       _textwrap.py
ª   ª   ª       ª       _winconsole.py
ª   ª   ª       ª       __init__.py
ª   ª   ª       ª       
ª   ª   ª       +---click-8.1.7.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.rst
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---colorama
ª   ª   ª       ª   ª   ansi.py
ª   ª   ª       ª   ª   ansitowin32.py
ª   ª   ª       ª   ª   initialise.py
ª   ª   ª       ª   ª   win32.py
ª   ª   ª       ª   ª   winterm.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---tests
ª   ª   ª       ª           ansitowin32_test.py
ª   ª   ª       ª           ansi_test.py
ª   ª   ª       ª           initialise_test.py
ª   ª   ª       ª           isatty_test.py
ª   ª   ª       ª           utils.py
ª   ª   ª       ª           winterm_test.py
ª   ª   ª       ª           __init__.py
ª   ª   ª       ª           
ª   ª   ª       +---colorama-0.4.6.dist-info
ª   ª   ª       ª   ª   INSTALLER
ª   ª   ª       ª   ª   METADATA
ª   ª   ª       ª   ª   RECORD
ª   ª   ª       ª   ª   WHEEL
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---licenses
ª   ª   ª       ª           LICENSE.txt
ª   ª   ª       ª           
ª   ª   ª       +---dateutil
ª   ª   ª       ª   ª   easter.py
ª   ª   ª       ª   ª   relativedelta.py
ª   ª   ª       ª   ª   rrule.py
ª   ª   ª       ª   ª   tzwin.py
ª   ª   ª       ª   ª   utils.py
ª   ª   ª       ª   ª   _common.py
ª   ª   ª       ª   ª   _version.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---parser
ª   ª   ª       ª   ª       isoparser.py
ª   ª   ª       ª   ª       _parser.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---tz
ª   ª   ª       ª   ª       tz.py
ª   ª   ª       ª   ª       win.py
ª   ª   ª       ª   ª       _common.py
ª   ª   ª       ª   ª       _factories.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---zoneinfo
ª   ª   ª       ª           dateutil-zoneinfo.tar.gz
ª   ª   ª       ª           rebuild.py
ª   ª   ª       ª           __init__.py
ª   ª   ª       ª           
ª   ª   ª       +---flask
ª   ª   ª       ª   ª   app.py
ª   ª   ª       ª   ª   blueprints.py
ª   ª   ª       ª   ª   cli.py
ª   ª   ª       ª   ª   config.py
ª   ª   ª       ª   ª   ctx.py
ª   ª   ª       ª   ª   debughelpers.py
ª   ª   ª       ª   ª   globals.py
ª   ª   ª       ª   ª   helpers.py
ª   ª   ª       ª   ª   logging.py
ª   ª   ª       ª   ª   py.typed
ª   ª   ª       ª   ª   sessions.py
ª   ª   ª       ª   ª   signals.py
ª   ª   ª       ª   ª   templating.py
ª   ª   ª       ª   ª   testing.py
ª   ª   ª       ª   ª   typing.py
ª   ª   ª       ª   ª   views.py
ª   ª   ª       ª   ª   wrappers.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   __main__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---json
ª   ª   ª       ª   ª       provider.py
ª   ª   ª       ª   ª       tag.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---sansio
ª   ª   ª       ª           app.py
ª   ª   ª       ª           blueprints.py
ª   ª   ª       ª           README.md
ª   ª   ª       ª           scaffold.py
ª   ª   ª       ª           
ª   ª   ª       +---flask-3.1.0.dist-info
ª   ª   ª       ª       entry_points.txt
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       REQUESTED
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---google
ª   ª   ª       ª   +---api
ª   ª   ª       ª   ª       annotations.proto
ª   ª   ª       ª   ª       annotations_pb2.py
ª   ª   ª       ª   ª       auth.proto
ª   ª   ª       ª   ª       auth_pb2.py
ª   ª   ª       ª   ª       backend.proto
ª   ª   ª       ª   ª       backend_pb2.py
ª   ª   ª       ª   ª       billing.proto
ª   ª   ª       ª   ª       billing_pb2.py
ª   ª   ª       ª   ª       client.proto
ª   ª   ª       ª   ª       client_pb2.py
ª   ª   ª       ª   ª       config_change.proto
ª   ª   ª       ª   ª       config_change_pb2.py
ª   ª   ª       ª   ª       consumer.proto
ª   ª   ª       ª   ª       consumer_pb2.py
ª   ª   ª       ª   ª       context.proto
ª   ª   ª       ª   ª       context_pb2.py
ª   ª   ª       ª   ª       control.proto
ª   ª   ª       ª   ª       control_pb2.py
ª   ª   ª       ª   ª       distribution.proto
ª   ª   ª       ª   ª       distribution_pb2.py
ª   ª   ª       ª   ª       documentation.proto
ª   ª   ª       ª   ª       documentation_pb2.py
ª   ª   ª       ª   ª       endpoint.proto
ª   ª   ª       ª   ª       endpoint_pb2.py
ª   ª   ª       ª   ª       error_reason.proto
ª   ª   ª       ª   ª       error_reason_pb2.py
ª   ª   ª       ª   ª       field_behavior.proto
ª   ª   ª       ª   ª       field_behavior_pb2.py
ª   ª   ª       ª   ª       field_info.proto
ª   ª   ª       ª   ª       field_info_pb2.py
ª   ª   ª       ª   ª       http.proto
ª   ª   ª       ª   ª       httpbody.proto
ª   ª   ª       ª   ª       httpbody_pb2.py
ª   ª   ª       ª   ª       http_pb2.py
ª   ª   ª       ª   ª       label.proto
ª   ª   ª       ª   ª       label_pb2.py
ª   ª   ª       ª   ª       launch_stage.proto
ª   ª   ª       ª   ª       launch_stage_pb2.py
ª   ª   ª       ª   ª       log.proto
ª   ª   ª       ª   ª       logging.proto
ª   ª   ª       ª   ª       logging_pb2.py
ª   ª   ª       ª   ª       log_pb2.py
ª   ª   ª       ª   ª       metric.proto
ª   ª   ª       ª   ª       metric_pb2.py
ª   ª   ª       ª   ª       monitored_resource.proto
ª   ª   ª       ª   ª       monitored_resource_pb2.py
ª   ª   ª       ª   ª       monitoring.proto
ª   ª   ª       ª   ª       monitoring_pb2.py
ª   ª   ª       ª   ª       policy.proto
ª   ª   ª       ª   ª       policy_pb2.py
ª   ª   ª       ª   ª       quota.proto
ª   ª   ª       ª   ª       quota_pb2.py
ª   ª   ª       ª   ª       resource.proto
ª   ª   ª       ª   ª       resource_pb2.py
ª   ª   ª       ª   ª       routing.proto
ª   ª   ª       ª   ª       routing_pb2.py
ª   ª   ª       ª   ª       service.proto
ª   ª   ª       ª   ª       service_pb2.py
ª   ª   ª       ª   ª       source_info.proto
ª   ª   ª       ª   ª       source_info_pb2.py
ª   ª   ª       ª   ª       system_parameter.proto
ª   ª   ª       ª   ª       system_parameter_pb2.py
ª   ª   ª       ª   ª       usage.proto
ª   ª   ª       ª   ª       usage_pb2.py
ª   ª   ª       ª   ª       visibility.proto
ª   ª   ª       ª   ª       visibility_pb2.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---api_core
ª   ª   ª       ª   ª   ª   bidi.py
ª   ª   ª       ª   ª   ª   client_info.py
ª   ª   ª       ª   ª   ª   client_options.py
ª   ª   ª       ª   ª   ª   datetime_helpers.py
ª   ª   ª       ª   ª   ª   exceptions.py
ª   ª   ª       ª   ª   ª   extended_operation.py
ª   ª   ª       ª   ª   ª   general_helpers.py
ª   ª   ª       ª   ª   ª   grpc_helpers.py
ª   ª   ª       ª   ª   ª   grpc_helpers_async.py
ª   ª   ª       ª   ª   ª   iam.py
ª   ª   ª       ª   ª   ª   operation.py
ª   ª   ª       ª   ª   ª   operation_async.py
ª   ª   ª       ª   ª   ª   page_iterator.py
ª   ª   ª       ª   ª   ª   page_iterator_async.py
ª   ª   ª       ª   ª   ª   path_template.py
ª   ª   ª       ª   ª   ª   protobuf_helpers.py
ª   ª   ª       ª   ª   ª   py.typed
ª   ª   ª       ª   ª   ª   rest_helpers.py
ª   ª   ª       ª   ª   ª   rest_streaming.py
ª   ª   ª       ª   ª   ª   rest_streaming_async.py
ª   ª   ª       ª   ª   ª   retry_async.py
ª   ª   ª       ª   ª   ª   timeout.py
ª   ª   ª       ª   ª   ª   universe.py
ª   ª   ª       ª   ª   ª   version.py
ª   ª   ª       ª   ª   ª   version_header.py
ª   ª   ª       ª   ª   ª   _rest_streaming_base.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---future
ª   ª   ª       ª   ª   ª       async_future.py
ª   ª   ª       ª   ª   ª       base.py
ª   ª   ª       ª   ª   ª       polling.py
ª   ª   ª       ª   ª   ª       _helpers.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---gapic_v1
ª   ª   ª       ª   ª   ª       client_info.py
ª   ª   ª       ª   ª   ª       config.py
ª   ª   ª       ª   ª   ª       config_async.py
ª   ª   ª       ª   ª   ª       method.py
ª   ª   ª       ª   ª   ª       method_async.py
ª   ª   ª       ª   ª   ª       routing_header.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---operations_v1
ª   ª   ª       ª   ª   ª   ª   abstract_operations_base_client.py
ª   ª   ª       ª   ª   ª   ª   abstract_operations_client.py
ª   ª   ª       ª   ª   ª   ª   operations_async_client.py
ª   ª   ª       ª   ª   ª   ª   operations_client.py
ª   ª   ª       ª   ª   ª   ª   operations_client_config.py
ª   ª   ª       ª   ª   ª   ª   operations_rest_client_async.py
ª   ª   ª       ª   ª   ª   ª   pagers.py
ª   ª   ª       ª   ª   ª   ª   pagers_async.py
ª   ª   ª       ª   ª   ª   ª   pagers_base.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---transports
ª   ª   ª       ª   ª   ª           base.py
ª   ª   ª       ª   ª   ª           rest.py
ª   ª   ª       ª   ª   ª           rest_asyncio.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---retry
ª   ª   ª       ª   ª           retry_base.py
ª   ª   ª       ª   ª           retry_streaming.py
ª   ª   ª       ª   ª           retry_streaming_async.py
ª   ª   ª       ª   ª           retry_unary.py
ª   ª   ª       ª   ª           retry_unary_async.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---auth
ª   ª   ª       ª   ª   ª   api_key.py
ª   ª   ª       ª   ª   ª   app_engine.py
ª   ª   ª       ª   ª   ª   aws.py
ª   ª   ª       ª   ª   ª   credentials.py
ª   ª   ª       ª   ª   ª   downscoped.py
ª   ª   ª       ª   ª   ª   environment_vars.py
ª   ª   ª       ª   ª   ª   exceptions.py
ª   ª   ª       ª   ª   ª   external_account.py
ª   ª   ª       ª   ª   ª   external_account_authorized_user.py
ª   ª   ª       ª   ª   ª   iam.py
ª   ª   ª       ª   ª   ª   identity_pool.py
ª   ª   ª       ª   ª   ª   impersonated_credentials.py
ª   ª   ª       ª   ª   ª   jwt.py
ª   ª   ª       ª   ª   ª   metrics.py
ª   ª   ª       ª   ª   ª   pluggable.py
ª   ª   ª       ª   ª   ª   py.typed
ª   ª   ª       ª   ª   ª   version.py
ª   ª   ª       ª   ª   ª   _cloud_sdk.py
ª   ª   ª       ª   ª   ª   _credentials_async.py
ª   ª   ª       ª   ª   ª   _credentials_base.py
ª   ª   ª       ª   ª   ª   _default.py
ª   ª   ª       ª   ª   ª   _default_async.py
ª   ª   ª       ª   ª   ª   _exponential_backoff.py
ª   ª   ª       ª   ª   ª   _helpers.py
ª   ª   ª       ª   ª   ª   _jwt_async.py
ª   ª   ª       ª   ª   ª   _oauth2client.py
ª   ª   ª       ª   ª   ª   _refresh_worker.py
ª   ª   ª       ª   ª   ª   _service_account_info.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---aio
ª   ª   ª       ª   ª   ª   ª   credentials.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---transport
ª   ª   ª       ª   ª   ª           aiohttp.py
ª   ª   ª       ª   ª   ª           sessions.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---compute_engine
ª   ª   ª       ª   ª   ª       credentials.py
ª   ª   ª       ª   ª   ª       _metadata.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---crypt
ª   ª   ª       ª   ª   ª       base.py
ª   ª   ª       ª   ª   ª       es256.py
ª   ª   ª       ª   ª   ª       rsa.py
ª   ª   ª       ª   ª   ª       _cryptography_rsa.py
ª   ª   ª       ª   ª   ª       _helpers.py
ª   ª   ª       ª   ª   ª       _python_rsa.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---transport
ª   ª   ª       ª   ª           grpc.py
ª   ª   ª       ª   ª           mtls.py
ª   ª   ª       ª   ª           requests.py
ª   ª   ª       ª   ª           urllib3.py
ª   ª   ª       ª   ª           _aiohttp_requests.py
ª   ª   ª       ª   ª           _custom_tls_signer.py
ª   ª   ª       ª   ª           _http_client.py
ª   ª   ª       ª   ª           _mtls_helper.py
ª   ª   ª       ª   ª           _requests_base.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---cloud
ª   ª   ª       ª   ª   ª   extended_operations.proto
ª   ª   ª       ª   ª   ª   extended_operations_pb2.py
ª   ª   ª       ª   ª   ª   version.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---client
ª   ª   ª       ª   ª   ª       py.typed
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---environment_vars
ª   ª   ª       ª   ª   ª       py.typed
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---exceptions
ª   ª   ª       ª   ª   ª       py.typed
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---location
ª   ª   ª       ª   ª   ª       locations.proto
ª   ª   ª       ª   ª   ª       locations_pb2.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---obsolete
ª   ª   ª       ª   ª   ª       py.typed
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---operation
ª   ª   ª       ª   ª   ª       py.typed
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---storage
ª   ª   ª       ª   ª   ª       acl.py
ª   ª   ª       ª   ª   ª       batch.py
ª   ª   ª       ª   ª   ª       blob.py
ª   ª   ª       ª   ª   ª       bucket.py
ª   ª   ª       ª   ª   ª       client.py
ª   ª   ª       ª   ª   ª       constants.py
ª   ª   ª       ª   ª   ª       fileio.py
ª   ª   ª       ª   ª   ª       hmac_key.py
ª   ª   ª       ª   ª   ª       iam.py
ª   ª   ª       ª   ª   ª       notification.py
ª   ª   ª       ª   ª   ª       retry.py
ª   ª   ª       ª   ª   ª       transfer_manager.py
ª   ª   ª       ª   ª   ª       version.py
ª   ª   ª       ª   ª   ª       _helpers.py
ª   ª   ª       ª   ª   ª       _http.py
ª   ª   ª       ª   ª   ª       _opentelemetry_tracing.py
ª   ª   ª       ª   ª   ª       _signing.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_helpers
ª   ª   ª       ª   ª   ª       py.typed
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_http
ª   ª   ª       ª   ª   ª       py.typed
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_testing
ª   ª   ª       ª   ª           py.typed
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---gapic
ª   ª   ª       ª   ª   +---metadata
ª   ª   ª       ª   ª           gapic_metadata.proto
ª   ª   ª       ª   ª           gapic_metadata_pb2.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---logging
ª   ª   ª       ª   ª   +---type
ª   ª   ª       ª   ª           http_request.proto
ª   ª   ª       ª   ª           http_request_pb2.py
ª   ª   ª       ª   ª           log_severity.proto
ª   ª   ª       ª   ª           log_severity_pb2.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---longrunning
ª   ª   ª       ª   ª       operations.proto
ª   ª   ª       ª   ª       operations_grpc.py
ª   ª   ª       ª   ª       operations_grpc_pb2.py
ª   ª   ª       ª   ª       operations_pb2.py
ª   ª   ª       ª   ª       operations_pb2_grpc.py
ª   ª   ª       ª   ª       operations_proto.py
ª   ª   ª       ª   ª       operations_proto_pb2.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---oauth2
ª   ª   ª       ª   ª       challenges.py
ª   ª   ª       ª   ª       credentials.py
ª   ª   ª       ª   ª       gdch_credentials.py
ª   ª   ª       ª   ª       id_token.py
ª   ª   ª       ª   ª       py.typed
ª   ª   ª       ª   ª       reauth.py
ª   ª   ª       ª   ª       service_account.py
ª   ª   ª       ª   ª       sts.py
ª   ª   ª       ª   ª       utils.py
ª   ª   ª       ª   ª       webauthn_handler.py
ª   ª   ª       ª   ª       webauthn_handler_factory.py
ª   ª   ª       ª   ª       webauthn_types.py
ª   ª   ª       ª   ª       _client.py
ª   ª   ª       ª   ª       _client_async.py
ª   ª   ª       ª   ª       _credentials_async.py
ª   ª   ª       ª   ª       _id_token_async.py
ª   ª   ª       ª   ª       _reauth_async.py
ª   ª   ª       ª   ª       _service_account_async.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---protobuf
ª   ª   ª       ª   ª   ª   any_pb2.py
ª   ª   ª       ª   ª   ª   api_pb2.py
ª   ª   ª       ª   ª   ª   descriptor.py
ª   ª   ª       ª   ª   ª   descriptor_database.py
ª   ª   ª       ª   ª   ª   descriptor_pb2.py
ª   ª   ª       ª   ª   ª   descriptor_pool.py
ª   ª   ª       ª   ª   ª   duration_pb2.py
ª   ª   ª       ª   ª   ª   empty_pb2.py
ª   ª   ª       ª   ª   ª   field_mask_pb2.py
ª   ª   ª       ª   ª   ª   json_format.py
ª   ª   ª       ª   ª   ª   message.py
ª   ª   ª       ª   ª   ª   message_factory.py
ª   ª   ª       ª   ª   ª   proto.py
ª   ª   ª       ª   ª   ª   proto_builder.py
ª   ª   ª       ª   ª   ª   proto_json.py
ª   ª   ª       ª   ª   ª   reflection.py
ª   ª   ª       ª   ª   ª   runtime_version.py
ª   ª   ª       ª   ª   ª   service.py
ª   ª   ª       ª   ª   ª   service_reflection.py
ª   ª   ª       ª   ª   ª   source_context_pb2.py
ª   ª   ª       ª   ª   ª   struct_pb2.py
ª   ª   ª       ª   ª   ª   symbol_database.py
ª   ª   ª       ª   ª   ª   text_encoding.py
ª   ª   ª       ª   ª   ª   text_format.py
ª   ª   ª       ª   ª   ª   timestamp_pb2.py
ª   ª   ª       ª   ª   ª   type_pb2.py
ª   ª   ª       ª   ª   ª   unknown_fields.py
ª   ª   ª       ª   ª   ª   wrappers_pb2.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---compiler
ª   ª   ª       ª   ª   ª       plugin_pb2.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---internal
ª   ª   ª       ª   ª   ª       api_implementation.py
ª   ª   ª       ª   ª   ª       builder.py
ª   ª   ª       ª   ª   ª       containers.py
ª   ª   ª       ª   ª   ª       decoder.py
ª   ª   ª       ª   ª   ª       encoder.py
ª   ª   ª       ª   ª   ª       enum_type_wrapper.py
ª   ª   ª       ª   ª   ª       extension_dict.py
ª   ª   ª       ª   ª   ª       field_mask.py
ª   ª   ª       ª   ª   ª       message_listener.py
ª   ª   ª       ª   ª   ª       python_edition_defaults.py
ª   ª   ª       ª   ª   ª       python_message.py
ª   ª   ª       ª   ª   ª       testing_refleaks.py
ª   ª   ª       ª   ª   ª       type_checkers.py
ª   ª   ª       ª   ª   ª       well_known_types.py
ª   ª   ª       ª   ª   ª       wire_format.py
ª   ª   ª       ª   ª   ª       _parameterized.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---pyext
ª   ª   ª       ª   ª   ª       cpp_message.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---testdata
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---util
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---resumable_media
ª   ª   ª       ª   ª   ª   common.py
ª   ª   ª       ª   ª   ª   py.typed
ª   ª   ª       ª   ª   ª   _download.py
ª   ª   ª       ª   ª   ª   _helpers.py
ª   ª   ª       ª   ª   ª   _upload.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---requests
ª   ª   ª       ª   ª           download.py
ª   ª   ª       ª   ª           upload.py
ª   ª   ª       ª   ª           _request_helpers.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---rpc
ª   ª   ª       ª   ª   ª   code.proto
ª   ª   ª       ª   ª   ª   code_pb2.py
ª   ª   ª       ª   ª   ª   error_details.proto
ª   ª   ª       ª   ª   ª   error_details_pb2.py
ª   ª   ª       ª   ª   ª   http.proto
ª   ª   ª       ª   ª   ª   http_pb2.py
ª   ª   ª       ª   ª   ª   status.proto
ª   ª   ª       ª   ª   ª   status_pb2.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---context
ª   ª   ª       ª   ª           attribute_context.proto
ª   ª   ª       ª   ª           attribute_context_pb2.py
ª   ª   ª       ª   ª           audit_context.proto
ª   ª   ª       ª   ª           audit_context_pb2.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---type
ª   ª   ª       ª   ª       calendar_period.proto
ª   ª   ª       ª   ª       calendar_period_pb2.py
ª   ª   ª       ª   ª       color.proto
ª   ª   ª       ª   ª       color_pb2.py
ª   ª   ª       ª   ª       date.proto
ª   ª   ª       ª   ª       datetime.proto
ª   ª   ª       ª   ª       datetime_pb2.py
ª   ª   ª       ª   ª       date_pb2.py
ª   ª   ª       ª   ª       dayofweek.proto
ª   ª   ª       ª   ª       dayofweek_pb2.py
ª   ª   ª       ª   ª       decimal.proto
ª   ª   ª       ª   ª       decimal_pb2.py
ª   ª   ª       ª   ª       expr.proto
ª   ª   ª       ª   ª       expr_pb2.py
ª   ª   ª       ª   ª       fraction.proto
ª   ª   ª       ª   ª       fraction_pb2.py
ª   ª   ª       ª   ª       interval.proto
ª   ª   ª       ª   ª       interval_pb2.py
ª   ª   ª       ª   ª       latlng.proto
ª   ª   ª       ª   ª       latlng_pb2.py
ª   ª   ª       ª   ª       localized_text.proto
ª   ª   ª       ª   ª       localized_text_pb2.py
ª   ª   ª       ª   ª       money.proto
ª   ª   ª       ª   ª       money_pb2.py
ª   ª   ª       ª   ª       month.proto
ª   ª   ª       ª   ª       month_pb2.py
ª   ª   ª       ª   ª       phone_number.proto
ª   ª   ª       ª   ª       phone_number_pb2.py
ª   ª   ª       ª   ª       postal_address.proto
ª   ª   ª       ª   ª       postal_address_pb2.py
ª   ª   ª       ª   ª       quaternion.proto
ª   ª   ª       ª   ª       quaternion_pb2.py
ª   ª   ª       ª   ª       timeofday.proto
ª   ª   ª       ª   ª       timeofday_pb2.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---_async_resumable_media
ª   ª   ª       ª   ª   ª   _download.py
ª   ª   ª       ª   ª   ª   _helpers.py
ª   ª   ª       ª   ª   ª   _upload.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---requests
ª   ª   ª       ª   ª           download.py
ª   ª   ª       ª   ª           upload.py
ª   ª   ª       ª   ª           _request_helpers.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---_upb
ª   ª   ª       ª           _message.cp39-win32.pyd
ª   ª   ª       ª           
ª   ª   ª       +---googleapis_common_protos-1.66.0.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---google_api_core-2.23.0.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---google_auth-2.36.0.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---google_cloud_core-2.4.1.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---google_cloud_storage-2.18.2.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       REQUESTED
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---google_crc32c
ª   ª   ª       ª       cext.py
ª   ª   ª       ª       py.typed
ª   ª   ª       ª       python.py
ª   ª   ª       ª       _checksum.py
ª   ª   ª       ª       __config__.py
ª   ª   ª       ª       __init__.py
ª   ª   ª       ª       
ª   ª   ª       +---google_crc32c-1.6.0.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       zip-safe
ª   ª   ª       ª       
ª   ª   ª       +---google_resumable_media-2.7.2.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---idna
ª   ª   ª       ª       codec.py
ª   ª   ª       ª       compat.py
ª   ª   ª       ª       core.py
ª   ª   ª       ª       idnadata.py
ª   ª   ª       ª       intranges.py
ª   ª   ª       ª       package_data.py
ª   ª   ª       ª       py.typed
ª   ª   ª       ª       uts46data.py
ª   ª   ª       ª       __init__.py
ª   ª   ª       ª       
ª   ª   ª       +---idna-3.10.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.md
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---importlib_metadata
ª   ª   ª       ª   ª   diagnose.py
ª   ª   ª       ª   ª   py.typed
ª   ª   ª       ª   ª   _adapters.py
ª   ª   ª       ª   ª   _collections.py
ª   ª   ª       ª   ª   _compat.py
ª   ª   ª       ª   ª   _functools.py
ª   ª   ª       ª   ª   _itertools.py
ª   ª   ª       ª   ª   _meta.py
ª   ª   ª       ª   ª   _text.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---compat
ª   ª   ª       ª           py311.py
ª   ª   ª       ª           py39.py
ª   ª   ª       ª           __init__.py
ª   ª   ª       ª           
ª   ª   ª       +---importlib_metadata-8.5.0.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---itsdangerous
ª   ª   ª       ª       encoding.py
ª   ª   ª       ª       exc.py
ª   ª   ª       ª       py.typed
ª   ª   ª       ª       serializer.py
ª   ª   ª       ª       signer.py
ª   ª   ª       ª       timed.py
ª   ª   ª       ª       url_safe.py
ª   ª   ª       ª       _json.py
ª   ª   ª       ª       __init__.py
ª   ª   ª       ª       
ª   ª   ª       +---itsdangerous-2.2.0.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---jinja2
ª   ª   ª       ª       async_utils.py
ª   ª   ª       ª       bccache.py
ª   ª   ª       ª       compiler.py
ª   ª   ª       ª       constants.py
ª   ª   ª       ª       debug.py
ª   ª   ª       ª       defaults.py
ª   ª   ª       ª       environment.py
ª   ª   ª       ª       exceptions.py
ª   ª   ª       ª       ext.py
ª   ª   ª       ª       filters.py
ª   ª   ª       ª       idtracking.py
ª   ª   ª       ª       lexer.py
ª   ª   ª       ª       loaders.py
ª   ª   ª       ª       meta.py
ª   ª   ª       ª       nativetypes.py
ª   ª   ª       ª       nodes.py
ª   ª   ª       ª       optimizer.py
ª   ª   ª       ª       parser.py
ª   ª   ª       ª       py.typed
ª   ª   ª       ª       runtime.py
ª   ª   ª       ª       sandbox.py
ª   ª   ª       ª       tests.py
ª   ª   ª       ª       utils.py
ª   ª   ª       ª       visitor.py
ª   ª   ª       ª       _identifier.py
ª   ª   ª       ª       __init__.py
ª   ª   ª       ª       
ª   ª   ª       +---jinja2-3.1.4.dist-info
ª   ª   ª       ª       entry_points.txt
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---joblib
ª   ª   ª       ª   ª   backports.py
ª   ª   ª       ª   ª   compressor.py
ª   ª   ª       ª   ª   disk.py
ª   ª   ª       ª   ª   executor.py
ª   ª   ª       ª   ª   func_inspect.py
ª   ª   ª       ª   ª   hashing.py
ª   ª   ª       ª   ª   logger.py
ª   ª   ª       ª   ª   memory.py
ª   ª   ª       ª   ª   numpy_pickle.py
ª   ª   ª       ª   ª   numpy_pickle_compat.py
ª   ª   ª       ª   ª   numpy_pickle_utils.py
ª   ª   ª       ª   ª   parallel.py
ª   ª   ª       ª   ª   pool.py
ª   ª   ª       ª   ª   testing.py
ª   ª   ª       ª   ª   _cloudpickle_wrapper.py
ª   ª   ª       ª   ª   _dask.py
ª   ª   ª       ª   ª   _memmapping_reducer.py
ª   ª   ª       ª   ª   _multiprocessing_helpers.py
ª   ª   ª       ª   ª   _parallel_backends.py
ª   ª   ª       ª   ª   _store_backends.py
ª   ª   ª       ª   ª   _utils.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---externals
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---cloudpickle
ª   ª   ª       ª   ª   ª       cloudpickle.py
ª   ª   ª       ª   ª   ª       cloudpickle_fast.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---loky
ª   ª   ª       ª   ª       ª   cloudpickle_wrapper.py
ª   ª   ª       ª   ª       ª   initializers.py
ª   ª   ª       ª   ª       ª   process_executor.py
ª   ª   ª       ª   ª       ª   reusable_executor.py
ª   ª   ª       ª   ª       ª   _base.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---backend
ª   ª   ª       ª   ª               context.py
ª   ª   ª       ª   ª               fork_exec.py
ª   ª   ª       ª   ª               popen_loky_posix.py
ª   ª   ª       ª   ª               popen_loky_win32.py
ª   ª   ª       ª   ª               process.py
ª   ª   ª       ª   ª               queues.py
ª   ª   ª       ª   ª               reduction.py
ª   ª   ª       ª   ª               resource_tracker.py
ª   ª   ª       ª   ª               spawn.py
ª   ª   ª       ª   ª               synchronize.py
ª   ª   ª       ª   ª               utils.py
ª   ª   ª       ª   ª               _posix_reduction.py
ª   ª   ª       ª   ª               _win_reduction.py
ª   ª   ª       ª   ª               __init__.py
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---test
ª   ª   ª       ª       ª   common.py
ª   ª   ª       ª       ª   testutils.py
ª   ª   ª       ª       ª   test_backports.py
ª   ª   ª       ª       ª   test_cloudpickle_wrapper.py
ª   ª   ª       ª       ª   test_config.py
ª   ª   ª       ª       ª   test_dask.py
ª   ª   ª       ª       ª   test_disk.py
ª   ª   ª       ª       ª   test_func_inspect.py
ª   ª   ª       ª       ª   test_func_inspect_special_encoding.py
ª   ª   ª       ª       ª   test_hashing.py
ª   ª   ª       ª       ª   test_init.py
ª   ª   ª       ª       ª   test_logger.py
ª   ª   ª       ª       ª   test_memmapping.py
ª   ª   ª       ª       ª   test_memory.py
ª   ª   ª       ª       ª   test_memory_async.py
ª   ª   ª       ª       ª   test_missing_multiprocessing.py
ª   ª   ª       ª       ª   test_module.py
ª   ª   ª       ª       ª   test_numpy_pickle.py
ª   ª   ª       ª       ª   test_numpy_pickle_compat.py
ª   ª   ª       ª       ª   test_numpy_pickle_utils.py
ª   ª   ª       ª       ª   test_parallel.py
ª   ª   ª       ª       ª   test_store_backends.py
ª   ª   ª       ª       ª   test_testing.py
ª   ª   ª       ª       ª   test_utils.py
ª   ª   ª       ª       ª   __init__.py
ª   ª   ª       ª       ª   
ª   ª   ª       ª       +---data
ª   ª   ª       ª               create_numpy_pickle.py
ª   ª   ª       ª               joblib_0.10.0_compressed_pickle_py27_np16.gz
ª   ª   ª       ª               joblib_0.10.0_compressed_pickle_py27_np17.gz
ª   ª   ª       ª               joblib_0.10.0_compressed_pickle_py33_np18.gz
ª   ª   ª       ª               joblib_0.10.0_compressed_pickle_py34_np19.gz
ª   ª   ª       ª               joblib_0.10.0_compressed_pickle_py35_np19.gz
ª   ª   ª       ª               joblib_0.10.0_pickle_py27_np17.pkl
ª   ª   ª       ª               joblib_0.10.0_pickle_py27_np17.pkl.bz2
ª   ª   ª       ª               joblib_0.10.0_pickle_py27_np17.pkl.gzip
ª   ª   ª       ª               joblib_0.10.0_pickle_py27_np17.pkl.lzma
ª   ª   ª       ª               joblib_0.10.0_pickle_py27_np17.pkl.xz
ª   ª   ª       ª               joblib_0.10.0_pickle_py33_np18.pkl
ª   ª   ª       ª               joblib_0.10.0_pickle_py33_np18.pkl.bz2
ª   ª   ª       ª               joblib_0.10.0_pickle_py33_np18.pkl.gzip
ª   ª   ª       ª               joblib_0.10.0_pickle_py33_np18.pkl.lzma
ª   ª   ª       ª               joblib_0.10.0_pickle_py33_np18.pkl.xz
ª   ª   ª       ª               joblib_0.10.0_pickle_py34_np19.pkl
ª   ª   ª       ª               joblib_0.10.0_pickle_py34_np19.pkl.bz2
ª   ª   ª       ª               joblib_0.10.0_pickle_py34_np19.pkl.gzip
ª   ª   ª       ª               joblib_0.10.0_pickle_py34_np19.pkl.lzma
ª   ª   ª       ª               joblib_0.10.0_pickle_py34_np19.pkl.xz
ª   ª   ª       ª               joblib_0.10.0_pickle_py35_np19.pkl
ª   ª   ª       ª               joblib_0.10.0_pickle_py35_np19.pkl.bz2
ª   ª   ª       ª               joblib_0.10.0_pickle_py35_np19.pkl.gzip
ª   ª   ª       ª               joblib_0.10.0_pickle_py35_np19.pkl.lzma
ª   ª   ª       ª               joblib_0.10.0_pickle_py35_np19.pkl.xz
ª   ª   ª       ª               joblib_0.11.0_compressed_pickle_py36_np111.gz
ª   ª   ª       ª               joblib_0.11.0_pickle_py36_np111.pkl
ª   ª   ª       ª               joblib_0.11.0_pickle_py36_np111.pkl.bz2
ª   ª   ª       ª               joblib_0.11.0_pickle_py36_np111.pkl.gzip
ª   ª   ª       ª               joblib_0.11.0_pickle_py36_np111.pkl.lzma
ª   ª   ª       ª               joblib_0.11.0_pickle_py36_np111.pkl.xz
ª   ª   ª       ª               joblib_0.8.4_compressed_pickle_py27_np17.gz
ª   ª   ª       ª               joblib_0.9.2_compressed_pickle_py27_np16.gz
ª   ª   ª       ª               joblib_0.9.2_compressed_pickle_py27_np17.gz
ª   ª   ª       ª               joblib_0.9.2_compressed_pickle_py34_np19.gz
ª   ª   ª       ª               joblib_0.9.2_compressed_pickle_py35_np19.gz
ª   ª   ª       ª               joblib_0.9.2_pickle_py27_np16.pkl
ª   ª   ª       ª               joblib_0.9.2_pickle_py27_np16.pkl_01.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py27_np16.pkl_02.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py27_np16.pkl_03.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py27_np16.pkl_04.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py27_np17.pkl
ª   ª   ª       ª               joblib_0.9.2_pickle_py27_np17.pkl_01.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py27_np17.pkl_02.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py27_np17.pkl_03.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py27_np17.pkl_04.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py33_np18.pkl
ª   ª   ª       ª               joblib_0.9.2_pickle_py33_np18.pkl_01.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py33_np18.pkl_02.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py33_np18.pkl_03.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py33_np18.pkl_04.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py34_np19.pkl
ª   ª   ª       ª               joblib_0.9.2_pickle_py34_np19.pkl_01.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py34_np19.pkl_02.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py34_np19.pkl_03.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py34_np19.pkl_04.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py35_np19.pkl
ª   ª   ª       ª               joblib_0.9.2_pickle_py35_np19.pkl_01.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py35_np19.pkl_02.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py35_np19.pkl_03.npy
ª   ª   ª       ª               joblib_0.9.2_pickle_py35_np19.pkl_04.npy
ª   ª   ª       ª               joblib_0.9.4.dev0_compressed_cache_size_pickle_py35_np19.gz
ª   ª   ª       ª               joblib_0.9.4.dev0_compressed_cache_size_pickle_py35_np19.gz_01.npy.z
ª   ª   ª       ª               joblib_0.9.4.dev0_compressed_cache_size_pickle_py35_np19.gz_02.npy.z
ª   ª   ª       ª               joblib_0.9.4.dev0_compressed_cache_size_pickle_py35_np19.gz_03.npy.z
ª   ª   ª       ª               __init__.py
ª   ª   ª       ª               
ª   ª   ª       +---joblib-1.4.2.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---markupsafe
ª   ª   ª       ª       py.typed
ª   ª   ª       ª       _native.py
ª   ª   ª       ª       _speedups.c
ª   ª   ª       ª       _speedups.cp39-win32.pyd
ª   ª   ª       ª       _speedups.pyi
ª   ª   ª       ª       __init__.py
ª   ª   ª       ª       
ª   ª   ª       +---MarkupSafe-3.0.2.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---numpy
ª   ª   ª       ª   ª   conftest.py
ª   ª   ª       ª   ª   ctypeslib.py
ª   ª   ª       ª   ª   ctypeslib.pyi
ª   ª   ª       ª   ª   dual.py
ª   ª   ª       ª   ª   LICENSE.txt
ª   ª   ª       ª   ª   matlib.py
ª   ª   ª       ª   ª   py.typed
ª   ª   ª       ª   ª   setup.py
ª   ª   ª       ª   ª   version.py
ª   ª   ª       ª   ª   _distributor_init.py
ª   ª   ª       ª   ª   _globals.py
ª   ª   ª       ª   ª   _pytesttester.py
ª   ª   ª       ª   ª   _pytesttester.pyi
ª   ª   ª       ª   ª   _version.py
ª   ª   ª       ª   ª   __config__.py
ª   ª   ª       ª   ª   __init__.cython-30.pxd
ª   ª   ª       ª   ª   __init__.pxd
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   __init__.pyi
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---.libs
ª   ª   ª       ª   ª       libopenblas_v0.3.21-gcc_8_3_0.dll
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---array_api
ª   ª   ª       ª   ª   ª   linalg.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _array_object.py
ª   ª   ª       ª   ª   ª   _constants.py
ª   ª   ª       ª   ª   ª   _creation_functions.py
ª   ª   ª       ª   ª   ª   _data_type_functions.py
ª   ª   ª       ª   ª   ª   _dtypes.py
ª   ª   ª       ª   ª   ª   _elementwise_functions.py
ª   ª   ª       ª   ª   ª   _manipulation_functions.py
ª   ª   ª       ª   ª   ª   _searching_functions.py
ª   ª   ª       ª   ª   ª   _set_functions.py
ª   ª   ª       ª   ª   ª   _sorting_functions.py
ª   ª   ª       ª   ª   ª   _statistical_functions.py
ª   ª   ª       ª   ª   ª   _typing.py
ª   ª   ª       ª   ª   ª   _utility_functions.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           test_array_object.py
ª   ª   ª       ª   ª           test_creation_functions.py
ª   ª   ª       ª   ª           test_data_type_functions.py
ª   ª   ª       ª   ª           test_elementwise_functions.py
ª   ª   ª       ª   ª           test_set_functions.py
ª   ª   ª       ª   ª           test_sorting_functions.py
ª   ª   ª       ª   ª           test_validation.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---compat
ª   ª   ª       ª   ª   ª   py3k.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _inspect.py
ª   ª   ª       ª   ª   ª   _pep440.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           test_compat.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---core
ª   ª   ª       ª   ª   ª   arrayprint.py
ª   ª   ª       ª   ª   ª   arrayprint.pyi
ª   ª   ª       ª   ª   ª   cversions.py
ª   ª   ª       ª   ª   ª   defchararray.py
ª   ª   ª       ª   ª   ª   defchararray.pyi
ª   ª   ª       ª   ª   ª   einsumfunc.py
ª   ª   ª       ª   ª   ª   einsumfunc.pyi
ª   ª   ª       ª   ª   ª   fromnumeric.py
ª   ª   ª       ª   ª   ª   fromnumeric.pyi
ª   ª   ª       ª   ª   ª   function_base.py
ª   ª   ª       ª   ª   ª   function_base.pyi
ª   ª   ª       ª   ª   ª   generate_numpy_api.py
ª   ª   ª       ª   ª   ª   getlimits.py
ª   ª   ª       ª   ª   ª   getlimits.pyi
ª   ª   ª       ª   ª   ª   memmap.py
ª   ª   ª       ª   ª   ª   memmap.pyi
ª   ª   ª       ª   ª   ª   multiarray.py
ª   ª   ª       ª   ª   ª   multiarray.pyi
ª   ª   ª       ª   ª   ª   numeric.py
ª   ª   ª       ª   ª   ª   numeric.pyi
ª   ª   ª       ª   ª   ª   numerictypes.py
ª   ª   ª       ª   ª   ª   numerictypes.pyi
ª   ª   ª       ª   ª   ª   overrides.py
ª   ª   ª       ª   ª   ª   records.py
ª   ª   ª       ª   ª   ª   records.pyi
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   setup_common.py
ª   ª   ª       ª   ª   ª   shape_base.py
ª   ª   ª       ª   ª   ª   shape_base.pyi
ª   ª   ª       ª   ª   ª   umath.py
ª   ª   ª       ª   ª   ª   umath_tests.py
ª   ª   ª       ª   ª   ª   _add_newdocs.py
ª   ª   ª       ª   ª   ª   _add_newdocs_scalars.py
ª   ª   ª       ª   ª   ª   _asarray.py
ª   ª   ª       ª   ª   ª   _asarray.pyi
ª   ª   ª       ª   ª   ª   _dtype.py
ª   ª   ª       ª   ª   ª   _dtype_ctypes.py
ª   ª   ª       ª   ª   ª   _exceptions.py
ª   ª   ª       ª   ª   ª   _internal.py
ª   ª   ª       ª   ª   ª   _internal.pyi
ª   ª   ª       ª   ª   ª   _machar.py
ª   ª   ª       ª   ª   ª   _methods.py
ª   ª   ª       ª   ª   ª   _multiarray_tests.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _multiarray_umath.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _operand_flag_tests.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _rational_tests.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _simd.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _string_helpers.py
ª   ª   ª       ª   ª   ª   _struct_ufunc_tests.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _type_aliases.py
ª   ª   ª       ª   ª   ª   _type_aliases.pyi
ª   ª   ª       ª   ª   ª   _ufunc_config.py
ª   ª   ª       ª   ª   ª   _ufunc_config.pyi
ª   ª   ª       ª   ª   ª   _umath_tests.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __init__.pyi
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---include
ª   ª   ª       ª   ª   ª   +---numpy
ª   ª   ª       ª   ª   ª       ª   .doxyfile
ª   ª   ª       ª   ª   ª       ª   arrayobject.h
ª   ª   ª       ª   ª   ª       ª   arrayscalars.h
ª   ª   ª       ª   ª   ª       ª   experimental_dtype_api.h
ª   ª   ª       ª   ª   ª       ª   halffloat.h
ª   ª   ª       ª   ª   ª       ª   multiarray_api.txt
ª   ª   ª       ª   ª   ª       ª   ndarrayobject.h
ª   ª   ª       ª   ª   ª       ª   ndarraytypes.h
ª   ª   ª       ª   ª   ª       ª   noprefix.h
ª   ª   ª       ª   ª   ª       ª   npy_1_7_deprecated_api.h
ª   ª   ª       ª   ª   ª       ª   npy_3kcompat.h
ª   ª   ª       ª   ª   ª       ª   npy_common.h
ª   ª   ª       ª   ª   ª       ª   npy_cpu.h
ª   ª   ª       ª   ª   ª       ª   npy_endian.h
ª   ª   ª       ª   ª   ª       ª   npy_interrupt.h
ª   ª   ª       ª   ª   ª       ª   npy_math.h
ª   ª   ª       ª   ª   ª       ª   npy_no_deprecated_api.h
ª   ª   ª       ª   ª   ª       ª   npy_os.h
ª   ª   ª       ª   ª   ª       ª   numpyconfig.h
ª   ª   ª       ª   ª   ª       ª   oldnumeric.h
ª   ª   ª       ª   ª   ª       ª   old_defines.h
ª   ª   ª       ª   ª   ª       ª   ufuncobject.h
ª   ª   ª       ª   ª   ª       ª   ufunc_api.txt
ª   ª   ª       ª   ª   ª       ª   utils.h
ª   ª   ª       ª   ª   ª       ª   _neighborhood_iterator_imp.h
ª   ª   ª       ª   ª   ª       ª   _numpyconfig.h
ª   ª   ª       ª   ª   ª       ª   __multiarray_api.h
ª   ª   ª       ª   ª   ª       ª   __ufunc_api.h
ª   ª   ª       ª   ª   ª       ª   
ª   ª   ª       ª   ª   ª       +---libdivide
ª   ª   ª       ª   ª   ª       ª       libdivide.h
ª   ª   ª       ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª   ª   ª       ª       
ª   ª   ª       ª   ª   ª       +---random
ª   ª   ª       ª   ª   ª               bitgen.h
ª   ª   ª       ª   ª   ª               distributions.h
ª   ª   ª       ª   ª   ª               
ª   ª   ª       ª   ª   +---lib
ª   ª   ª       ª   ª   ª   ª   npymath.lib
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---npy-pkg-config
ª   ª   ª       ª   ª   ª           mlib.ini
ª   ª   ª       ª   ª   ª           npymath.ini
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª       ª   test_abc.py
ª   ª   ª       ª   ª       ª   test_api.py
ª   ª   ª       ª   ª       ª   test_argparse.py
ª   ª   ª       ª   ª       ª   test_arraymethod.py
ª   ª   ª       ª   ª       ª   test_arrayprint.py
ª   ª   ª       ª   ª       ª   test_array_coercion.py
ª   ª   ª       ª   ª       ª   test_array_interface.py
ª   ª   ª       ª   ª       ª   test_casting_floatingpoint_errors.py
ª   ª   ª       ª   ª       ª   test_casting_unittests.py
ª   ª   ª       ª   ª       ª   test_conversion_utils.py
ª   ª   ª       ª   ª       ª   test_cpu_dispatcher.py
ª   ª   ª       ª   ª       ª   test_cpu_features.py
ª   ª   ª       ª   ª       ª   test_custom_dtypes.py
ª   ª   ª       ª   ª       ª   test_cython.py
ª   ª   ª       ª   ª       ª   test_datetime.py
ª   ª   ª       ª   ª       ª   test_defchararray.py
ª   ª   ª       ª   ª       ª   test_deprecations.py
ª   ª   ª       ª   ª       ª   test_dlpack.py
ª   ª   ª       ª   ª       ª   test_dtype.py
ª   ª   ª       ª   ª       ª   test_einsum.py
ª   ª   ª       ª   ª       ª   test_errstate.py
ª   ª   ª       ª   ª       ª   test_extint128.py
ª   ª   ª       ª   ª       ª   test_function_base.py
ª   ª   ª       ª   ª       ª   test_getlimits.py
ª   ª   ª       ª   ª       ª   test_half.py
ª   ª   ª       ª   ª       ª   test_hashtable.py
ª   ª   ª       ª   ª       ª   test_indexerrors.py
ª   ª   ª       ª   ª       ª   test_indexing.py
ª   ª   ª       ª   ª       ª   test_item_selection.py
ª   ª   ª       ª   ª       ª   test_limited_api.py
ª   ª   ª       ª   ª       ª   test_longdouble.py
ª   ª   ª       ª   ª       ª   test_machar.py
ª   ª   ª       ª   ª       ª   test_memmap.py
ª   ª   ª       ª   ª       ª   test_mem_overlap.py
ª   ª   ª       ª   ª       ª   test_mem_policy.py
ª   ª   ª       ª   ª       ª   test_multiarray.py
ª   ª   ª       ª   ª       ª   test_nditer.py
ª   ª   ª       ª   ª       ª   test_nep50_promotions.py
ª   ª   ª       ª   ª       ª   test_numeric.py
ª   ª   ª       ª   ª       ª   test_numerictypes.py
ª   ª   ª       ª   ª       ª   test_overrides.py
ª   ª   ª       ª   ª       ª   test_print.py
ª   ª   ª       ª   ª       ª   test_protocols.py
ª   ª   ª       ª   ª       ª   test_records.py
ª   ª   ª       ª   ª       ª   test_regression.py
ª   ª   ª       ª   ª       ª   test_scalarbuffer.py
ª   ª   ª       ª   ª       ª   test_scalarinherit.py
ª   ª   ª       ª   ª       ª   test_scalarmath.py
ª   ª   ª       ª   ª       ª   test_scalarprint.py
ª   ª   ª       ª   ª       ª   test_scalar_ctors.py
ª   ª   ª       ª   ª       ª   test_scalar_methods.py
ª   ª   ª       ª   ª       ª   test_shape_base.py
ª   ª   ª       ª   ª       ª   test_simd.py
ª   ª   ª       ª   ª       ª   test_simd_module.py
ª   ª   ª       ª   ª       ª   test_strings.py
ª   ª   ª       ª   ª       ª   test_ufunc.py
ª   ª   ª       ª   ª       ª   test_umath.py
ª   ª   ª       ª   ª       ª   test_umath_accuracy.py
ª   ª   ª       ª   ª       ª   test_umath_complex.py
ª   ª   ª       ª   ª       ª   test_unicode.py
ª   ª   ª       ª   ª       ª   test__exceptions.py
ª   ª   ª       ª   ª       ª   _locales.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---data
ª   ª   ª       ª   ª       ª       astype_copy.pkl
ª   ª   ª       ª   ª       ª       generate_umath_validation_data.cpp
ª   ª   ª       ª   ª       ª       recarray_from_file.fits
ª   ª   ª       ª   ª       ª       umath-validation-set-arccos.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-arccosh.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-arcsin.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-arcsinh.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-arctan.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-arctanh.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-cbrt.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-cos.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-cosh.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-exp.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-exp2.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-expm1.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-log.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-log10.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-log1p.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-log2.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-README.txt
ª   ª   ª       ª   ª       ª       umath-validation-set-sin.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-sinh.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-tan.csv
ª   ª   ª       ª   ª       ª       umath-validation-set-tanh.csv
ª   ª   ª       ª   ª       ª       
ª   ª   ª       ª   ª       +---examples
ª   ª   ª       ª   ª           +---cython
ª   ª   ª       ª   ª           ª       checks.pyx
ª   ª   ª       ª   ª           ª       setup.py
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---limited_api
ª   ª   ª       ª   ª                   limited_api.c
ª   ª   ª       ª   ª                   setup.py
ª   ª   ª       ª   ª                   
ª   ª   ª       ª   +---distutils
ª   ª   ª       ª   ª   ª   armccompiler.py
ª   ª   ª       ª   ª   ª   ccompiler.py
ª   ª   ª       ª   ª   ª   ccompiler_opt.py
ª   ª   ª       ª   ª   ª   conv_template.py
ª   ª   ª       ª   ª   ª   core.py
ª   ª   ª       ª   ª   ª   cpuinfo.py
ª   ª   ª       ª   ª   ª   exec_command.py
ª   ª   ª       ª   ª   ª   extension.py
ª   ª   ª       ª   ª   ª   from_template.py
ª   ª   ª       ª   ª   ª   intelccompiler.py
ª   ª   ª       ª   ª   ª   lib2def.py
ª   ª   ª       ª   ª   ª   line_endings.py
ª   ª   ª       ª   ª   ª   log.py
ª   ª   ª       ª   ª   ª   mingw32ccompiler.py
ª   ª   ª       ª   ª   ª   misc_util.py
ª   ª   ª       ª   ª   ª   msvc9compiler.py
ª   ª   ª       ª   ª   ª   msvccompiler.py
ª   ª   ª       ª   ª   ª   npy_pkg_config.py
ª   ª   ª       ª   ª   ª   numpy_distribution.py
ª   ª   ª       ª   ª   ª   pathccompiler.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   system_info.py
ª   ª   ª       ª   ª   ª   unixccompiler.py
ª   ª   ª       ª   ª   ª   _shell_utils.py
ª   ª   ª       ª   ª   ª   __config__.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __init__.pyi
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---checks
ª   ª   ª       ª   ª   ª       cpu_asimd.c
ª   ª   ª       ª   ª   ª       cpu_asimddp.c
ª   ª   ª       ª   ª   ª       cpu_asimdfhm.c
ª   ª   ª       ª   ª   ª       cpu_asimdhp.c
ª   ª   ª       ª   ª   ª       cpu_avx.c
ª   ª   ª       ª   ª   ª       cpu_avx2.c
ª   ª   ª       ª   ª   ª       cpu_avx512cd.c
ª   ª   ª       ª   ª   ª       cpu_avx512f.c
ª   ª   ª       ª   ª   ª       cpu_avx512_clx.c
ª   ª   ª       ª   ª   ª       cpu_avx512_cnl.c
ª   ª   ª       ª   ª   ª       cpu_avx512_icl.c
ª   ª   ª       ª   ª   ª       cpu_avx512_knl.c
ª   ª   ª       ª   ª   ª       cpu_avx512_knm.c
ª   ª   ª       ª   ª   ª       cpu_avx512_skx.c
ª   ª   ª       ª   ª   ª       cpu_f16c.c
ª   ª   ª       ª   ª   ª       cpu_fma3.c
ª   ª   ª       ª   ª   ª       cpu_fma4.c
ª   ª   ª       ª   ª   ª       cpu_neon.c
ª   ª   ª       ª   ª   ª       cpu_neon_fp16.c
ª   ª   ª       ª   ª   ª       cpu_neon_vfpv4.c
ª   ª   ª       ª   ª   ª       cpu_popcnt.c
ª   ª   ª       ª   ª   ª       cpu_sse.c
ª   ª   ª       ª   ª   ª       cpu_sse2.c
ª   ª   ª       ª   ª   ª       cpu_sse3.c
ª   ª   ª       ª   ª   ª       cpu_sse41.c
ª   ª   ª       ª   ª   ª       cpu_sse42.c
ª   ª   ª       ª   ª   ª       cpu_ssse3.c
ª   ª   ª       ª   ª   ª       cpu_vsx.c
ª   ª   ª       ª   ª   ª       cpu_vsx2.c
ª   ª   ª       ª   ª   ª       cpu_vsx3.c
ª   ª   ª       ª   ª   ª       cpu_vsx4.c
ª   ª   ª       ª   ª   ª       cpu_vx.c
ª   ª   ª       ª   ª   ª       cpu_vxe.c
ª   ª   ª       ª   ª   ª       cpu_vxe2.c
ª   ª   ª       ª   ª   ª       cpu_xop.c
ª   ª   ª       ª   ª   ª       extra_avx512bw_mask.c
ª   ª   ª       ª   ª   ª       extra_avx512dq_mask.c
ª   ª   ª       ª   ª   ª       extra_avx512f_reduce.c
ª   ª   ª       ª   ª   ª       extra_vsx4_mma.c
ª   ª   ª       ª   ª   ª       extra_vsx_asm.c
ª   ª   ª       ª   ª   ª       test_flags.c
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---command
ª   ª   ª       ª   ª   ª       autodist.py
ª   ª   ª       ª   ª   ª       bdist_rpm.py
ª   ª   ª       ª   ª   ª       build.py
ª   ª   ª       ª   ª   ª       build_clib.py
ª   ª   ª       ª   ª   ª       build_ext.py
ª   ª   ª       ª   ª   ª       build_py.py
ª   ª   ª       ª   ª   ª       build_scripts.py
ª   ª   ª       ª   ª   ª       build_src.py
ª   ª   ª       ª   ª   ª       config.py
ª   ª   ª       ª   ª   ª       config_compiler.py
ª   ª   ª       ª   ª   ª       develop.py
ª   ª   ª       ª   ª   ª       egg_info.py
ª   ª   ª       ª   ª   ª       install.py
ª   ª   ª       ª   ª   ª       install_clib.py
ª   ª   ª       ª   ª   ª       install_data.py
ª   ª   ª       ª   ª   ª       install_headers.py
ª   ª   ª       ª   ª   ª       sdist.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---fcompiler
ª   ª   ª       ª   ª   ª       absoft.py
ª   ª   ª       ª   ª   ª       arm.py
ª   ª   ª       ª   ª   ª       compaq.py
ª   ª   ª       ª   ª   ª       environment.py
ª   ª   ª       ª   ª   ª       fujitsu.py
ª   ª   ª       ª   ª   ª       g95.py
ª   ª   ª       ª   ª   ª       gnu.py
ª   ª   ª       ª   ª   ª       hpux.py
ª   ª   ª       ª   ª   ª       ibm.py
ª   ª   ª       ª   ª   ª       intel.py
ª   ª   ª       ª   ª   ª       lahey.py
ª   ª   ª       ª   ª   ª       mips.py
ª   ª   ª       ª   ª   ª       nag.py
ª   ª   ª       ª   ª   ª       none.py
ª   ª   ª       ª   ª   ª       nv.py
ª   ª   ª       ª   ª   ª       pathf95.py
ª   ª   ª       ª   ª   ª       pg.py
ª   ª   ª       ª   ª   ª       sun.py
ª   ª   ª       ª   ª   ª       vast.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---mingw
ª   ª   ª       ª   ª   ª       gfortran_vs2003_hack.c
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           test_build_ext.py
ª   ª   ª       ª   ª           test_ccompiler_opt.py
ª   ª   ª       ª   ª           test_ccompiler_opt_conf.py
ª   ª   ª       ª   ª           test_exec_command.py
ª   ª   ª       ª   ª           test_fcompiler.py
ª   ª   ª       ª   ª           test_fcompiler_gnu.py
ª   ª   ª       ª   ª           test_fcompiler_intel.py
ª   ª   ª       ª   ª           test_fcompiler_nagfor.py
ª   ª   ª       ª   ª           test_from_template.py
ª   ª   ª       ª   ª           test_log.py
ª   ª   ª       ª   ª           test_mingw32ccompiler.py
ª   ª   ª       ª   ª           test_misc_util.py
ª   ª   ª       ª   ª           test_npy_pkg_config.py
ª   ª   ª       ª   ª           test_shell_utils.py
ª   ª   ª       ª   ª           test_system_info.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---doc
ª   ª   ª       ª   ª       constants.py
ª   ª   ª       ª   ª       ufuncs.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---f2py
ª   ª   ª       ª   ª   ª   auxfuncs.py
ª   ª   ª       ª   ª   ª   capi_maps.py
ª   ª   ª       ª   ª   ª   cb_rules.py
ª   ª   ª       ª   ª   ª   cfuncs.py
ª   ª   ª       ª   ª   ª   common_rules.py
ª   ª   ª       ª   ª   ª   crackfortran.py
ª   ª   ª       ª   ª   ª   diagnose.py
ª   ª   ª       ª   ª   ª   f2py2e.py
ª   ª   ª       ª   ª   ª   f90mod_rules.py
ª   ª   ª       ª   ª   ª   func2subr.py
ª   ª   ª       ª   ª   ª   rules.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   symbolic.py
ª   ª   ª       ª   ª   ª   use_rules.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __init__.pyi
ª   ª   ª       ª   ª   ª   __main__.py
ª   ª   ª       ª   ª   ª   __version__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---src
ª   ª   ª       ª   ª   ª       fortranobject.c
ª   ª   ª       ª   ª   ª       fortranobject.h
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª       ª   test_abstract_interface.py
ª   ª   ª       ª   ª       ª   test_array_from_pyobj.py
ª   ª   ª       ª   ª       ª   test_assumed_shape.py
ª   ª   ª       ª   ª       ª   test_block_docstring.py
ª   ª   ª       ª   ª       ª   test_callback.py
ª   ª   ª       ª   ª       ª   test_character.py
ª   ª   ª       ª   ª       ª   test_common.py
ª   ª   ª       ª   ª       ª   test_compile_function.py
ª   ª   ª       ª   ª       ª   test_crackfortran.py
ª   ª   ª       ª   ª       ª   test_docs.py
ª   ª   ª       ª   ª       ª   test_f2cmap.py
ª   ª   ª       ª   ª       ª   test_f2py2e.py
ª   ª   ª       ª   ª       ª   test_kind.py
ª   ª   ª       ª   ª       ª   test_mixed.py
ª   ª   ª       ª   ª       ª   test_module_doc.py
ª   ª   ª       ª   ª       ª   test_parameter.py
ª   ª   ª       ª   ª       ª   test_quoted_character.py
ª   ª   ª       ª   ª       ª   test_regression.py
ª   ª   ª       ª   ª       ª   test_return_character.py
ª   ª   ª       ª   ª       ª   test_return_complex.py
ª   ª   ª       ª   ª       ª   test_return_integer.py
ª   ª   ª       ª   ª       ª   test_return_logical.py
ª   ª   ª       ª   ª       ª   test_return_real.py
ª   ª   ª       ª   ª       ª   test_semicolon_split.py
ª   ª   ª       ª   ª       ª   test_size.py
ª   ª   ª       ª   ª       ª   test_string.py
ª   ª   ª       ª   ª       ª   test_symbolic.py
ª   ª   ª       ª   ª       ª   test_value_attrspec.py
ª   ª   ª       ª   ª       ª   util.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---src
ª   ª   ª       ª   ª           +---abstract_interface
ª   ª   ª       ª   ª           ª       foo.f90
ª   ª   ª       ª   ª           ª       gh18403_mod.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---array_from_pyobj
ª   ª   ª       ª   ª           ª       wrapmodule.c
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---assumed_shape
ª   ª   ª       ª   ª           ª       .f2py_f2cmap
ª   ª   ª       ª   ª           ª       foo_free.f90
ª   ª   ª       ª   ª           ª       foo_mod.f90
ª   ª   ª       ª   ª           ª       foo_use.f90
ª   ª   ª       ª   ª           ª       precision.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---block_docstring
ª   ª   ª       ª   ª           ª       foo.f
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---callback
ª   ª   ª       ª   ª           ª       foo.f
ª   ª   ª       ª   ª           ª       gh17797.f90
ª   ª   ª       ª   ª           ª       gh18335.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---cli
ª   ª   ª       ª   ª           ª       hi77.f
ª   ª   ª       ª   ª           ª       hiworld.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---common
ª   ª   ª       ª   ª           ª       block.f
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---crackfortran
ª   ª   ª       ª   ª           ª       accesstype.f90
ª   ª   ª       ª   ª           ª       foo_deps.f90
ª   ª   ª       ª   ª           ª       gh15035.f
ª   ª   ª       ª   ª           ª       gh17859.f
ª   ª   ª       ª   ª           ª       gh2848.f90
ª   ª   ª       ª   ª           ª       operators.f90
ª   ª   ª       ª   ª           ª       privatemod.f90
ª   ª   ª       ª   ª           ª       publicmod.f90
ª   ª   ª       ª   ª           ª       pubprivmod.f90
ª   ª   ª       ª   ª           ª       unicode_comment.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---f2cmap
ª   ª   ª       ª   ª           ª       .f2py_f2cmap
ª   ª   ª       ª   ª           ª       isoFortranEnvMap.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---kind
ª   ª   ª       ª   ª           ª       foo.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---mixed
ª   ª   ª       ª   ª           ª       foo.f
ª   ª   ª       ª   ª           ª       foo_fixed.f90
ª   ª   ª       ª   ª           ª       foo_free.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---module_data
ª   ª   ª       ª   ª           ª       mod.mod
ª   ª   ª       ª   ª           ª       module_data_docstring.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---negative_bounds
ª   ª   ª       ª   ª           ª       issue_20853.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---parameter
ª   ª   ª       ª   ª           ª       constant_both.f90
ª   ª   ª       ª   ª           ª       constant_compound.f90
ª   ª   ª       ª   ª           ª       constant_integer.f90
ª   ª   ª       ª   ª           ª       constant_non_compound.f90
ª   ª   ª       ª   ª           ª       constant_real.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---quoted_character
ª   ª   ª       ª   ª           ª       foo.f
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---regression
ª   ª   ª       ª   ª           ª       inout.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---return_character
ª   ª   ª       ª   ª           ª       foo77.f
ª   ª   ª       ª   ª           ª       foo90.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---return_complex
ª   ª   ª       ª   ª           ª       foo77.f
ª   ª   ª       ª   ª           ª       foo90.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---return_integer
ª   ª   ª       ª   ª           ª       foo77.f
ª   ª   ª       ª   ª           ª       foo90.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---return_logical
ª   ª   ª       ª   ª           ª       foo77.f
ª   ª   ª       ª   ª           ª       foo90.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---return_real
ª   ª   ª       ª   ª           ª       foo77.f
ª   ª   ª       ª   ª           ª       foo90.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---size
ª   ª   ª       ª   ª           ª       foo.f90
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---string
ª   ª   ª       ª   ª           ª       char.f90
ª   ª   ª       ª   ª           ª       fixed_string.f90
ª   ª   ª       ª   ª           ª       scalar_string.f90
ª   ª   ª       ª   ª           ª       string.f
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---value_attrspec
ª   ª   ª       ª   ª                   gh21665.f90
ª   ª   ª       ª   ª                   
ª   ª   ª       ª   +---fft
ª   ª   ª       ª   ª   ª   helper.py
ª   ª   ª       ª   ª   ª   helper.pyi
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _pocketfft.py
ª   ª   ª       ª   ª   ª   _pocketfft.pyi
ª   ª   ª       ª   ª   ª   _pocketfft_internal.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __init__.pyi
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           test_helper.py
ª   ª   ª       ª   ª           test_pocketfft.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---lib
ª   ª   ª       ª   ª   ª   arraypad.py
ª   ª   ª       ª   ª   ª   arraypad.pyi
ª   ª   ª       ª   ª   ª   arraysetops.py
ª   ª   ª       ª   ª   ª   arraysetops.pyi
ª   ª   ª       ª   ª   ª   arrayterator.py
ª   ª   ª       ª   ª   ª   arrayterator.pyi
ª   ª   ª       ª   ª   ª   format.py
ª   ª   ª       ª   ª   ª   format.pyi
ª   ª   ª       ª   ª   ª   function_base.py
ª   ª   ª       ª   ª   ª   function_base.pyi
ª   ª   ª       ª   ª   ª   histograms.py
ª   ª   ª       ª   ª   ª   histograms.pyi
ª   ª   ª       ª   ª   ª   index_tricks.py
ª   ª   ª       ª   ª   ª   index_tricks.pyi
ª   ª   ª       ª   ª   ª   mixins.py
ª   ª   ª       ª   ª   ª   mixins.pyi
ª   ª   ª       ª   ª   ª   nanfunctions.py
ª   ª   ª       ª   ª   ª   nanfunctions.pyi
ª   ª   ª       ª   ª   ª   npyio.py
ª   ª   ª       ª   ª   ª   npyio.pyi
ª   ª   ª       ª   ª   ª   polynomial.py
ª   ª   ª       ª   ª   ª   polynomial.pyi
ª   ª   ª       ª   ª   ª   recfunctions.py
ª   ª   ª       ª   ª   ª   scimath.py
ª   ª   ª       ª   ª   ª   scimath.pyi
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   shape_base.py
ª   ª   ª       ª   ª   ª   shape_base.pyi
ª   ª   ª       ª   ª   ª   stride_tricks.py
ª   ª   ª       ª   ª   ª   stride_tricks.pyi
ª   ª   ª       ª   ª   ª   twodim_base.py
ª   ª   ª       ª   ª   ª   twodim_base.pyi
ª   ª   ª       ª   ª   ª   type_check.py
ª   ª   ª       ª   ª   ª   type_check.pyi
ª   ª   ª       ª   ª   ª   ufunclike.py
ª   ª   ª       ª   ª   ª   ufunclike.pyi
ª   ª   ª       ª   ª   ª   user_array.py
ª   ª   ª       ª   ª   ª   utils.py
ª   ª   ª       ª   ª   ª   utils.pyi
ª   ª   ª       ª   ª   ª   _datasource.py
ª   ª   ª       ª   ª   ª   _iotools.py
ª   ª   ª       ª   ª   ª   _version.py
ª   ª   ª       ª   ª   ª   _version.pyi
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __init__.pyi
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª       ª   test_arraypad.py
ª   ª   ª       ª   ª       ª   test_arraysetops.py
ª   ª   ª       ª   ª       ª   test_arrayterator.py
ª   ª   ª       ª   ª       ª   test_financial_expired.py
ª   ª   ª       ª   ª       ª   test_format.py
ª   ª   ª       ª   ª       ª   test_function_base.py
ª   ª   ª       ª   ª       ª   test_histograms.py
ª   ª   ª       ª   ª       ª   test_index_tricks.py
ª   ª   ª       ª   ª       ª   test_io.py
ª   ª   ª       ª   ª       ª   test_loadtxt.py
ª   ª   ª       ª   ª       ª   test_mixins.py
ª   ª   ª       ª   ª       ª   test_nanfunctions.py
ª   ª   ª       ª   ª       ª   test_packbits.py
ª   ª   ª       ª   ª       ª   test_polynomial.py
ª   ª   ª       ª   ª       ª   test_recfunctions.py
ª   ª   ª       ª   ª       ª   test_regression.py
ª   ª   ª       ª   ª       ª   test_shape_base.py
ª   ª   ª       ª   ª       ª   test_stride_tricks.py
ª   ª   ª       ª   ª       ª   test_twodim_base.py
ª   ª   ª       ª   ª       ª   test_type_check.py
ª   ª   ª       ª   ª       ª   test_ufunclike.py
ª   ª   ª       ª   ª       ª   test_utils.py
ª   ª   ª       ª   ª       ª   test__datasource.py
ª   ª   ª       ª   ª       ª   test__iotools.py
ª   ª   ª       ª   ª       ª   test__version.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---data
ª   ª   ª       ª   ª               py2-objarr.npy
ª   ª   ª       ª   ª               py2-objarr.npz
ª   ª   ª       ª   ª               py3-objarr.npy
ª   ª   ª       ª   ª               py3-objarr.npz
ª   ª   ª       ª   ª               python3.npy
ª   ª   ª       ª   ª               win64python2.npy
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---linalg
ª   ª   ª       ª   ª   ª   lapack_lite.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   linalg.py
ª   ª   ª       ª   ª   ª   linalg.pyi
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _umath_linalg.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __init__.pyi
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           test_deprecations.py
ª   ª   ª       ª   ª           test_linalg.py
ª   ª   ª       ª   ª           test_regression.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---ma
ª   ª   ª       ª   ª   ª   bench.py
ª   ª   ª       ª   ª   ª   core.py
ª   ª   ª       ª   ª   ª   core.pyi
ª   ª   ª       ª   ª   ª   extras.py
ª   ª   ª       ª   ª   ª   extras.pyi
ª   ª   ª       ª   ª   ª   mrecords.py
ª   ª   ª       ª   ª   ª   mrecords.pyi
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   testutils.py
ª   ª   ª       ª   ª   ª   timer_comparison.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __init__.pyi
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           test_core.py
ª   ª   ª       ª   ª           test_deprecations.py
ª   ª   ª       ª   ª           test_extras.py
ª   ª   ª       ª   ª           test_mrecords.py
ª   ª   ª       ª   ª           test_old_ma.py
ª   ª   ª       ª   ª           test_regression.py
ª   ª   ª       ª   ª           test_subclassing.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---matrixlib
ª   ª   ª       ª   ª   ª   defmatrix.py
ª   ª   ª       ª   ª   ª   defmatrix.pyi
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __init__.pyi
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           test_defmatrix.py
ª   ª   ª       ª   ª           test_interaction.py
ª   ª   ª       ª   ª           test_masked_matrix.py
ª   ª   ª       ª   ª           test_matrix_linalg.py
ª   ª   ª       ª   ª           test_multiarray.py
ª   ª   ª       ª   ª           test_numeric.py
ª   ª   ª       ª   ª           test_regression.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---polynomial
ª   ª   ª       ª   ª   ª   chebyshev.py
ª   ª   ª       ª   ª   ª   chebyshev.pyi
ª   ª   ª       ª   ª   ª   hermite.py
ª   ª   ª       ª   ª   ª   hermite.pyi
ª   ª   ª       ª   ª   ª   hermite_e.py
ª   ª   ª       ª   ª   ª   hermite_e.pyi
ª   ª   ª       ª   ª   ª   laguerre.py
ª   ª   ª       ª   ª   ª   laguerre.pyi
ª   ª   ª       ª   ª   ª   legendre.py
ª   ª   ª       ª   ª   ª   legendre.pyi
ª   ª   ª       ª   ª   ª   polynomial.py
ª   ª   ª       ª   ª   ª   polynomial.pyi
ª   ª   ª       ª   ª   ª   polyutils.py
ª   ª   ª       ª   ª   ª   polyutils.pyi
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _polybase.py
ª   ª   ª       ª   ª   ª   _polybase.pyi
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __init__.pyi
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           test_chebyshev.py
ª   ª   ª       ª   ª           test_classes.py
ª   ª   ª       ª   ª           test_hermite.py
ª   ª   ª       ª   ª           test_hermite_e.py
ª   ª   ª       ª   ª           test_laguerre.py
ª   ª   ª       ª   ª           test_legendre.py
ª   ª   ª       ª   ª           test_polynomial.py
ª   ª   ª       ª   ª           test_polyutils.py
ª   ª   ª       ª   ª           test_printing.py
ª   ª   ª       ª   ª           test_symbol.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---random
ª   ª   ª       ª   ª   ª   bit_generator.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   bit_generator.pxd
ª   ª   ª       ª   ª   ª   bit_generator.pyi
ª   ª   ª       ª   ª   ª   c_distributions.pxd
ª   ª   ª       ª   ª   ª   mtrand.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   mtrand.pyi
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _bounded_integers.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _bounded_integers.pxd
ª   ª   ª       ª   ª   ª   _common.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _common.pxd
ª   ª   ª       ª   ª   ª   _generator.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _generator.pyi
ª   ª   ª       ª   ª   ª   _mt19937.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _mt19937.pyi
ª   ª   ª       ª   ª   ª   _pcg64.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _pcg64.pyi
ª   ª   ª       ª   ª   ª   _philox.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _philox.pyi
ª   ª   ª       ª   ª   ª   _pickle.py
ª   ª   ª       ª   ª   ª   _sfc64.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _sfc64.pyi
ª   ª   ª       ª   ª   ª   __init__.pxd
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __init__.pyi
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---lib
ª   ª   ª       ª   ª   ª       npyrandom.lib
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª   ª   ª   test_direct.py
ª   ª   ª       ª   ª   ª   ª   test_extending.py
ª   ª   ª       ª   ª   ª   ª   test_generator_mt19937.py
ª   ª   ª       ª   ª   ª   ª   test_generator_mt19937_regressions.py
ª   ª   ª       ª   ª   ª   ª   test_random.py
ª   ª   ª       ª   ª   ª   ª   test_randomstate.py
ª   ª   ª       ª   ª   ª   ª   test_randomstate_regression.py
ª   ª   ª       ª   ª   ª   ª   test_regression.py
ª   ª   ª       ª   ª   ª   ª   test_seed_sequence.py
ª   ª   ª       ª   ª   ª   ª   test_smoke.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---data
ª   ª   ª       ª   ª   ª           mt19937-testset-1.csv
ª   ª   ª       ª   ª   ª           mt19937-testset-2.csv
ª   ª   ª       ª   ª   ª           pcg64-testset-1.csv
ª   ª   ª       ª   ª   ª           pcg64-testset-2.csv
ª   ª   ª       ª   ª   ª           pcg64dxsm-testset-1.csv
ª   ª   ª       ª   ª   ª           pcg64dxsm-testset-2.csv
ª   ª   ª       ª   ª   ª           philox-testset-1.csv
ª   ª   ª       ª   ª   ª           philox-testset-2.csv
ª   ª   ª       ª   ª   ª           sfc64-testset-1.csv
ª   ª   ª       ª   ª   ª           sfc64-testset-2.csv
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---_examples
ª   ª   ª       ª   ª       +---cffi
ª   ª   ª       ª   ª       ª       extending.py
ª   ª   ª       ª   ª       ª       parse.py
ª   ª   ª       ª   ª       ª       
ª   ª   ª       ª   ª       +---cython
ª   ª   ª       ª   ª       ª       extending.pyx
ª   ª   ª       ª   ª       ª       extending_distributions.pyx
ª   ª   ª       ª   ª       ª       setup.py
ª   ª   ª       ª   ª       ª       
ª   ª   ª       ª   ª       +---numba
ª   ª   ª       ª   ª               extending.py
ª   ª   ª       ª   ª               extending_distributions.py
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---testing
ª   ª   ª       ª   ª   ª   print_coercion_tables.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   utils.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __init__.pyi
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª   ª       test_doctesting.py
ª   ª   ª       ª   ª   ª       test_utils.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_private
ª   ª   ª       ª   ª           decorators.py
ª   ª   ª       ª   ª           extbuild.py
ª   ª   ª       ª   ª           noseclasses.py
ª   ª   ª       ª   ª           nosetester.py
ª   ª   ª       ª   ª           parameterized.py
ª   ª   ª       ª   ª           utils.py
ª   ª   ª       ª   ª           utils.pyi
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---tests
ª   ª   ª       ª   ª       test_ctypeslib.py
ª   ª   ª       ª   ª       test_lazyloading.py
ª   ª   ª       ª   ª       test_matlib.py
ª   ª   ª       ª   ª       test_numpy_version.py
ª   ª   ª       ª   ª       test_public_api.py
ª   ª   ª       ª   ª       test_reloading.py
ª   ª   ª       ª   ª       test_scripts.py
ª   ª   ª       ª   ª       test_warnings.py
ª   ª   ª       ª   ª       test__all__.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---typing
ª   ª   ª       ª   ª   ª   mypy_plugin.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª       ª   test_generic_alias.py
ª   ª   ª       ª   ª       ª   test_isfile.py
ª   ª   ª       ª   ª       ª   test_runtime.py
ª   ª   ª       ª   ª       ª   test_typing.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---data
ª   ª   ª       ª   ª           ª   mypy.ini
ª   ª   ª       ª   ª           ª   
ª   ª   ª       ª   ª           +---fail
ª   ª   ª       ª   ª           ª       arithmetic.pyi
ª   ª   ª       ª   ª           ª       arrayprint.pyi
ª   ª   ª       ª   ª           ª       arrayterator.pyi
ª   ª   ª       ª   ª           ª       array_constructors.pyi
ª   ª   ª       ª   ª           ª       array_like.pyi
ª   ª   ª       ª   ª           ª       array_pad.pyi
ª   ª   ª       ª   ª           ª       bitwise_ops.pyi
ª   ª   ª       ª   ª           ª       char.pyi
ª   ª   ª       ª   ª           ª       chararray.pyi
ª   ª   ª       ª   ª           ª       comparisons.pyi
ª   ª   ª       ª   ª           ª       constants.pyi
ª   ª   ª       ª   ª           ª       datasource.pyi
ª   ª   ª       ª   ª           ª       dtype.pyi
ª   ª   ª       ª   ª           ª       einsumfunc.pyi
ª   ª   ª       ª   ª           ª       false_positives.pyi
ª   ª   ª       ª   ª           ª       flatiter.pyi
ª   ª   ª       ª   ª           ª       fromnumeric.pyi
ª   ª   ª       ª   ª           ª       histograms.pyi
ª   ª   ª       ª   ª           ª       index_tricks.pyi
ª   ª   ª       ª   ª           ª       lib_function_base.pyi
ª   ª   ª       ª   ª           ª       lib_polynomial.pyi
ª   ª   ª       ª   ª           ª       lib_utils.pyi
ª   ª   ª       ª   ª           ª       lib_version.pyi
ª   ª   ª       ª   ª           ª       linalg.pyi
ª   ª   ª       ª   ª           ª       memmap.pyi
ª   ª   ª       ª   ª           ª       modules.pyi
ª   ª   ª       ª   ª           ª       multiarray.pyi
ª   ª   ª       ª   ª           ª       ndarray.pyi
ª   ª   ª       ª   ª           ª       ndarray_misc.pyi
ª   ª   ª       ª   ª           ª       nditer.pyi
ª   ª   ª       ª   ª           ª       nested_sequence.pyi
ª   ª   ª       ª   ª           ª       npyio.pyi
ª   ª   ª       ª   ª           ª       numerictypes.pyi
ª   ª   ª       ª   ª           ª       random.pyi
ª   ª   ª       ª   ª           ª       rec.pyi
ª   ª   ª       ª   ª           ª       scalars.pyi
ª   ª   ª       ª   ª           ª       shape_base.pyi
ª   ª   ª       ª   ª           ª       stride_tricks.pyi
ª   ª   ª       ª   ª           ª       testing.pyi
ª   ª   ª       ª   ª           ª       twodim_base.pyi
ª   ª   ª       ª   ª           ª       type_check.pyi
ª   ª   ª       ª   ª           ª       ufunclike.pyi
ª   ª   ª       ª   ª           ª       ufuncs.pyi
ª   ª   ª       ª   ª           ª       ufunc_config.pyi
ª   ª   ª       ª   ª           ª       warnings_and_errors.pyi
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---misc
ª   ª   ª       ª   ª           ª       extended_precision.pyi
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---pass
ª   ª   ª       ª   ª           ª       arithmetic.py
ª   ª   ª       ª   ª           ª       arrayprint.py
ª   ª   ª       ª   ª           ª       arrayterator.py
ª   ª   ª       ª   ª           ª       array_constructors.py
ª   ª   ª       ª   ª           ª       array_like.py
ª   ª   ª       ª   ª           ª       bitwise_ops.py
ª   ª   ª       ª   ª           ª       comparisons.py
ª   ª   ª       ª   ª           ª       dtype.py
ª   ª   ª       ª   ª           ª       einsumfunc.py
ª   ª   ª       ª   ª           ª       flatiter.py
ª   ª   ª       ª   ª           ª       fromnumeric.py
ª   ª   ª       ª   ª           ª       index_tricks.py
ª   ª   ª       ª   ª           ª       lib_utils.py
ª   ª   ª       ª   ª           ª       lib_version.py
ª   ª   ª       ª   ª           ª       literal.py
ª   ª   ª       ª   ª           ª       mod.py
ª   ª   ª       ª   ª           ª       modules.py
ª   ª   ª       ª   ª           ª       multiarray.py
ª   ª   ª       ª   ª           ª       ndarray_conversion.py
ª   ª   ª       ª   ª           ª       ndarray_misc.py
ª   ª   ª       ª   ª           ª       ndarray_shape_manipulation.py
ª   ª   ª       ª   ª           ª       numeric.py
ª   ª   ª       ª   ª           ª       numerictypes.py
ª   ª   ª       ª   ª           ª       random.py
ª   ª   ª       ª   ª           ª       scalars.py
ª   ª   ª       ª   ª           ª       simple.py
ª   ª   ª       ª   ª           ª       simple_py3.py
ª   ª   ª       ª   ª           ª       ufunclike.py
ª   ª   ª       ª   ª           ª       ufuncs.py
ª   ª   ª       ª   ª           ª       ufunc_config.py
ª   ª   ª       ª   ª           ª       warnings_and_errors.py
ª   ª   ª       ª   ª           ª       
ª   ª   ª       ª   ª           +---reveal
ª   ª   ª       ª   ª                   arithmetic.pyi
ª   ª   ª       ª   ª                   arraypad.pyi
ª   ª   ª       ª   ª                   arrayprint.pyi
ª   ª   ª       ª   ª                   arraysetops.pyi
ª   ª   ª       ª   ª                   arrayterator.pyi
ª   ª   ª       ª   ª                   array_constructors.pyi
ª   ª   ª       ª   ª                   bitwise_ops.pyi
ª   ª   ª       ª   ª                   char.pyi
ª   ª   ª       ª   ª                   chararray.pyi
ª   ª   ª       ª   ª                   comparisons.pyi
ª   ª   ª       ª   ª                   constants.pyi
ª   ª   ª       ª   ª                   ctypeslib.pyi
ª   ª   ª       ª   ª                   datasource.pyi
ª   ª   ª       ª   ª                   dtype.pyi
ª   ª   ª       ª   ª                   einsumfunc.pyi
ª   ª   ª       ª   ª                   emath.pyi
ª   ª   ª       ª   ª                   false_positives.pyi
ª   ª   ª       ª   ª                   fft.pyi
ª   ª   ª       ª   ª                   flatiter.pyi
ª   ª   ª       ª   ª                   fromnumeric.pyi
ª   ª   ª       ª   ª                   getlimits.pyi
ª   ª   ª       ª   ª                   histograms.pyi
ª   ª   ª       ª   ª                   index_tricks.pyi
ª   ª   ª       ª   ª                   lib_function_base.pyi
ª   ª   ª       ª   ª                   lib_polynomial.pyi
ª   ª   ª       ª   ª                   lib_utils.pyi
ª   ª   ª       ª   ª                   lib_version.pyi
ª   ª   ª       ª   ª                   linalg.pyi
ª   ª   ª       ª   ª                   matrix.pyi
ª   ª   ª       ª   ª                   memmap.pyi
ª   ª   ª       ª   ª                   mod.pyi
ª   ª   ª       ª   ª                   modules.pyi
ª   ª   ª       ª   ª                   multiarray.pyi
ª   ª   ª       ª   ª                   nbit_base_example.pyi
ª   ª   ª       ª   ª                   ndarray_conversion.pyi
ª   ª   ª       ª   ª                   ndarray_misc.pyi
ª   ª   ª       ª   ª                   ndarray_shape_manipulation.pyi
ª   ª   ª       ª   ª                   nditer.pyi
ª   ª   ª       ª   ª                   nested_sequence.pyi
ª   ª   ª       ª   ª                   npyio.pyi
ª   ª   ª       ª   ª                   numeric.pyi
ª   ª   ª       ª   ª                   numerictypes.pyi
ª   ª   ª       ª   ª                   random.pyi
ª   ª   ª       ª   ª                   rec.pyi
ª   ª   ª       ª   ª                   scalars.pyi
ª   ª   ª       ª   ª                   shape_base.pyi
ª   ª   ª       ª   ª                   stride_tricks.pyi
ª   ª   ª       ª   ª                   testing.pyi
ª   ª   ª       ª   ª                   twodim_base.pyi
ª   ª   ª       ª   ª                   type_check.pyi
ª   ª   ª       ª   ª                   ufunclike.pyi
ª   ª   ª       ª   ª                   ufuncs.pyi
ª   ª   ª       ª   ª                   ufunc_config.pyi
ª   ª   ª       ª   ª                   version.pyi
ª   ª   ª       ª   ª                   warnings_and_errors.pyi
ª   ª   ª       ª   ª                   
ª   ª   ª       ª   +---_pyinstaller
ª   ª   ª       ª   ª       hook-numpy.py
ª   ª   ª       ª   ª       pyinstaller-smoke.py
ª   ª   ª       ª   ª       test_pyinstaller.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---_typing
ª   ª   ª       ª           setup.py
ª   ª   ª       ª           _add_docstring.py
ª   ª   ª       ª           _array_like.py
ª   ª   ª       ª           _callable.pyi
ª   ª   ª       ª           _char_codes.py
ª   ª   ª       ª           _dtype_like.py
ª   ª   ª       ª           _extended_precision.py
ª   ª   ª       ª           _generic_alias.py
ª   ª   ª       ª           _nbit.py
ª   ª   ª       ª           _nested_sequence.py
ª   ª   ª       ª           _scalars.py
ª   ª   ª       ª           _shape.py
ª   ª   ª       ª           _ufunc.pyi
ª   ª   ª       ª           __init__.py
ª   ª   ª       ª           
ª   ª   ª       +---numpy-1.24.4.dist-info
ª   ª   ª       ª       entry_points.txt
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª       LICENSES_bundled.txt
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---pandas
ª   ª   ª       ª   ª   conftest.py
ª   ª   ª       ª   ª   testing.py
ª   ª   ª       ª   ª   _typing.py
ª   ª   ª       ª   ª   _version.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---api
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---extensions
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---indexers
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---interchange
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---types
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---arrays
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---compat
ª   ª   ª       ª   ª   ª   compressors.py
ª   ª   ª       ª   ª   ª   pickle_compat.py
ª   ª   ª       ª   ª   ª   pyarrow.py
ª   ª   ª       ª   ª   ª   _constants.py
ª   ª   ª       ª   ª   ª   _optional.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---numpy
ª   ª   ª       ª   ª           function.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---core
ª   ª   ª       ª   ª   ª   accessor.py
ª   ª   ª       ª   ª   ª   algorithms.py
ª   ª   ª       ª   ª   ª   api.py
ª   ª   ª       ª   ª   ª   apply.py
ª   ª   ª       ª   ª   ª   arraylike.py
ª   ª   ª       ª   ª   ª   base.py
ª   ª   ª       ª   ª   ª   common.py
ª   ª   ª       ª   ª   ª   config_init.py
ª   ª   ª       ª   ª   ª   construction.py
ª   ª   ª       ª   ª   ª   flags.py
ª   ª   ª       ª   ª   ª   frame.py
ª   ª   ª       ª   ª   ª   generic.py
ª   ª   ª       ª   ª   ª   indexing.py
ª   ª   ª       ª   ª   ª   missing.py
ª   ª   ª       ª   ª   ª   nanops.py
ª   ª   ª       ª   ª   ª   resample.py
ª   ª   ª       ª   ª   ª   roperator.py
ª   ª   ª       ª   ª   ª   sample.py
ª   ª   ª       ª   ª   ª   series.py
ª   ª   ª       ª   ª   ª   shared_docs.py
ª   ª   ª       ª   ª   ª   sorting.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---arrays
ª   ª   ª       ª   ª   ª   ª   base.py
ª   ª   ª       ª   ª   ª   ª   boolean.py
ª   ª   ª       ª   ª   ª   ª   categorical.py
ª   ª   ª       ª   ª   ª   ª   datetimelike.py
ª   ª   ª       ª   ª   ª   ª   datetimes.py
ª   ª   ª       ª   ª   ª   ª   floating.py
ª   ª   ª       ª   ª   ª   ª   integer.py
ª   ª   ª       ª   ª   ª   ª   interval.py
ª   ª   ª       ª   ª   ª   ª   masked.py
ª   ª   ª       ª   ª   ª   ª   numeric.py
ª   ª   ª       ª   ª   ª   ª   numpy_.py
ª   ª   ª       ª   ª   ª   ª   period.py
ª   ª   ª       ª   ª   ª   ª   string_.py
ª   ª   ª       ª   ª   ª   ª   string_arrow.py
ª   ª   ª       ª   ª   ª   ª   timedeltas.py
ª   ª   ª       ª   ª   ª   ª   _mixins.py
ª   ª   ª       ª   ª   ª   ª   _ranges.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---arrow
ª   ª   ª       ª   ª   ª   ª       array.py
ª   ª   ª       ª   ª   ª   ª       dtype.py
ª   ª   ª       ª   ª   ª   ª       extension_types.py
ª   ª   ª       ª   ª   ª   ª       _arrow_utils.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---sparse
ª   ª   ª       ª   ª   ª           accessor.py
ª   ª   ª       ª   ª   ª           array.py
ª   ª   ª       ª   ª   ª           dtype.py
ª   ª   ª       ª   ª   ª           scipy_sparse.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---array_algos
ª   ª   ª       ª   ª   ª       datetimelike_accumulations.py
ª   ª   ª       ª   ª   ª       masked_accumulations.py
ª   ª   ª       ª   ª   ª       masked_reductions.py
ª   ª   ª       ª   ª   ª       putmask.py
ª   ª   ª       ª   ª   ª       quantile.py
ª   ª   ª       ª   ª   ª       replace.py
ª   ª   ª       ª   ª   ª       take.py
ª   ª   ª       ª   ª   ª       transforms.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---computation
ª   ª   ª       ª   ª   ª       align.py
ª   ª   ª       ª   ª   ª       api.py
ª   ª   ª       ª   ª   ª       check.py
ª   ª   ª       ª   ª   ª       common.py
ª   ª   ª       ª   ª   ª       engines.py
ª   ª   ª       ª   ª   ª       eval.py
ª   ª   ª       ª   ª   ª       expr.py
ª   ª   ª       ª   ª   ª       expressions.py
ª   ª   ª       ª   ª   ª       ops.py
ª   ª   ª       ª   ª   ª       parsing.py
ª   ª   ª       ª   ª   ª       pytables.py
ª   ª   ª       ª   ª   ª       scope.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---dtypes
ª   ª   ª       ª   ª   ª       api.py
ª   ª   ª       ª   ª   ª       astype.py
ª   ª   ª       ª   ª   ª       base.py
ª   ª   ª       ª   ª   ª       cast.py
ª   ª   ª       ª   ª   ª       common.py
ª   ª   ª       ª   ª   ª       concat.py
ª   ª   ª       ª   ª   ª       dtypes.py
ª   ª   ª       ª   ª   ª       generic.py
ª   ª   ª       ª   ª   ª       inference.py
ª   ª   ª       ª   ª   ª       missing.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---groupby
ª   ª   ª       ª   ª   ª       base.py
ª   ª   ª       ª   ª   ª       categorical.py
ª   ª   ª       ª   ª   ª       generic.py
ª   ª   ª       ª   ª   ª       groupby.py
ª   ª   ª       ª   ª   ª       grouper.py
ª   ª   ª       ª   ª   ª       indexing.py
ª   ª   ª       ª   ª   ª       numba_.py
ª   ª   ª       ª   ª   ª       ops.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---indexers
ª   ª   ª       ª   ª   ª       objects.py
ª   ª   ª       ª   ª   ª       utils.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---indexes
ª   ª   ª       ª   ª   ª       accessors.py
ª   ª   ª       ª   ª   ª       api.py
ª   ª   ª       ª   ª   ª       base.py
ª   ª   ª       ª   ª   ª       category.py
ª   ª   ª       ª   ª   ª       datetimelike.py
ª   ª   ª       ª   ª   ª       datetimes.py
ª   ª   ª       ª   ª   ª       extension.py
ª   ª   ª       ª   ª   ª       frozen.py
ª   ª   ª       ª   ª   ª       interval.py
ª   ª   ª       ª   ª   ª       multi.py
ª   ª   ª       ª   ª   ª       period.py
ª   ª   ª       ª   ª   ª       range.py
ª   ª   ª       ª   ª   ª       timedeltas.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---interchange
ª   ª   ª       ª   ª   ª       buffer.py
ª   ª   ª       ª   ª   ª       column.py
ª   ª   ª       ª   ª   ª       dataframe.py
ª   ª   ª       ª   ª   ª       dataframe_protocol.py
ª   ª   ª       ª   ª   ª       from_dataframe.py
ª   ª   ª       ª   ª   ª       utils.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---internals
ª   ª   ª       ª   ª   ª       api.py
ª   ª   ª       ª   ª   ª       array_manager.py
ª   ª   ª       ª   ª   ª       base.py
ª   ª   ª       ª   ª   ª       blocks.py
ª   ª   ª       ª   ª   ª       concat.py
ª   ª   ª       ª   ª   ª       construction.py
ª   ª   ª       ª   ª   ª       managers.py
ª   ª   ª       ª   ª   ª       ops.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---methods
ª   ª   ª       ª   ª   ª       describe.py
ª   ª   ª       ª   ª   ª       selectn.py
ª   ª   ª       ª   ª   ª       to_dict.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---ops
ª   ª   ª       ª   ª   ª       array_ops.py
ª   ª   ª       ª   ª   ª       common.py
ª   ª   ª       ª   ª   ª       dispatch.py
ª   ª   ª       ª   ª   ª       docstrings.py
ª   ª   ª       ª   ª   ª       invalid.py
ª   ª   ª       ª   ª   ª       mask_ops.py
ª   ª   ª       ª   ª   ª       methods.py
ª   ª   ª       ª   ª   ª       missing.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---reshape
ª   ª   ª       ª   ª   ª       api.py
ª   ª   ª       ª   ª   ª       concat.py
ª   ª   ª       ª   ª   ª       encoding.py
ª   ª   ª       ª   ª   ª       melt.py
ª   ª   ª       ª   ª   ª       merge.py
ª   ª   ª       ª   ª   ª       pivot.py
ª   ª   ª       ª   ª   ª       reshape.py
ª   ª   ª       ª   ª   ª       tile.py
ª   ª   ª       ª   ª   ª       util.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---sparse
ª   ª   ª       ª   ª   ª       api.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---strings
ª   ª   ª       ª   ª   ª       accessor.py
ª   ª   ª       ª   ª   ª       base.py
ª   ª   ª       ª   ª   ª       object_array.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---tools
ª   ª   ª       ª   ª   ª       datetimes.py
ª   ª   ª       ª   ª   ª       numeric.py
ª   ª   ª       ª   ª   ª       timedeltas.py
ª   ª   ª       ª   ª   ª       times.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---util
ª   ª   ª       ª   ª   ª       hashing.py
ª   ª   ª       ª   ª   ª       numba_.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---window
ª   ª   ª       ª   ª   ª       common.py
ª   ª   ª       ª   ª   ª       doc.py
ª   ª   ª       ª   ª   ª       ewm.py
ª   ª   ª       ª   ª   ª       expanding.py
ª   ª   ª       ª   ª   ª       numba_.py
ª   ª   ª       ª   ª   ª       online.py
ª   ª   ª       ª   ª   ª       rolling.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_numba
ª   ª   ª       ª   ª       ª   executor.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---kernels
ª   ª   ª       ª   ª               mean_.py
ª   ª   ª       ª   ª               min_max_.py
ª   ª   ª       ª   ª               shared.py
ª   ª   ª       ª   ª               sum_.py
ª   ª   ª       ª   ª               var_.py
ª   ª   ª       ª   ª               __init__.py
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---errors
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---io
ª   ª   ª       ª   ª   ª   api.py
ª   ª   ª       ª   ª   ª   clipboards.py
ª   ª   ª       ª   ª   ª   common.py
ª   ª   ª       ª   ª   ª   feather_format.py
ª   ª   ª       ª   ª   ª   gbq.py
ª   ª   ª       ª   ª   ª   html.py
ª   ª   ª       ª   ª   ª   orc.py
ª   ª   ª       ª   ª   ª   parquet.py
ª   ª   ª       ª   ª   ª   pickle.py
ª   ª   ª       ª   ª   ª   pytables.py
ª   ª   ª       ª   ª   ª   spss.py
ª   ª   ª       ª   ª   ª   sql.py
ª   ª   ª       ª   ª   ª   stata.py
ª   ª   ª       ª   ª   ª   xml.py
ª   ª   ª       ª   ª   ª   _util.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---clipboard
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---excel
ª   ª   ª       ª   ª   ª       _base.py
ª   ª   ª       ª   ª   ª       _odfreader.py
ª   ª   ª       ª   ª   ª       _odswriter.py
ª   ª   ª       ª   ª   ª       _openpyxl.py
ª   ª   ª       ª   ª   ª       _pyxlsb.py
ª   ª   ª       ª   ª   ª       _util.py
ª   ª   ª       ª   ª   ª       _xlrd.py
ª   ª   ª       ª   ª   ª       _xlsxwriter.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---formats
ª   ª   ª       ª   ª   ª   ª   console.py
ª   ª   ª       ª   ª   ª   ª   css.py
ª   ª   ª       ª   ª   ª   ª   csvs.py
ª   ª   ª       ª   ª   ª   ª   excel.py
ª   ª   ª       ª   ª   ª   ª   format.py
ª   ª   ª       ª   ª   ª   ª   html.py
ª   ª   ª       ª   ª   ª   ª   info.py
ª   ª   ª       ª   ª   ª   ª   latex.py
ª   ª   ª       ª   ª   ª   ª   printing.py
ª   ª   ª       ª   ª   ª   ª   string.py
ª   ª   ª       ª   ª   ª   ª   style.py
ª   ª   ª       ª   ª   ª   ª   style_render.py
ª   ª   ª       ª   ª   ª   ª   xml.py
ª   ª   ª       ª   ª   ª   ª   _color_data.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---templates
ª   ª   ª       ª   ª   ª           html.tpl
ª   ª   ª       ª   ª   ª           html_style.tpl
ª   ª   ª       ª   ª   ª           html_table.tpl
ª   ª   ª       ª   ª   ª           latex.tpl
ª   ª   ª       ª   ª   ª           latex_longtable.tpl
ª   ª   ª       ª   ª   ª           latex_table.tpl
ª   ª   ª       ª   ª   ª           string.tpl
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---json
ª   ª   ª       ª   ª   ª       _json.py
ª   ª   ª       ª   ª   ª       _normalize.py
ª   ª   ª       ª   ª   ª       _table_schema.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---parsers
ª   ª   ª       ª   ª   ª       arrow_parser_wrapper.py
ª   ª   ª       ª   ª   ª       base_parser.py
ª   ª   ª       ª   ª   ª       c_parser_wrapper.py
ª   ª   ª       ª   ª   ª       python_parser.py
ª   ª   ª       ª   ª   ª       readers.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---sas
ª   ª   ª       ª   ª           byteswap.pyx
ª   ª   ª       ª   ª           sas.pyx
ª   ª   ª       ª   ª           sas7bdat.py
ª   ª   ª       ª   ª           sasreader.py
ª   ª   ª       ª   ª           sas_constants.py
ª   ª   ª       ª   ª           sas_xport.py
ª   ª   ª       ª   ª           _byteswap.cp39-win32.pyd
ª   ª   ª       ª   ª           _byteswap.pyi
ª   ª   ª       ª   ª           _sas.cp39-win32.pyd
ª   ª   ª       ª   ª           _sas.pyi
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---plotting
ª   ª   ª       ª   ª   ª   _core.py
ª   ª   ª       ª   ª   ª   _misc.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---_matplotlib
ª   ª   ª       ª   ª           boxplot.py
ª   ª   ª       ª   ª           converter.py
ª   ª   ª       ª   ª           core.py
ª   ª   ª       ª   ª           groupby.py
ª   ª   ª       ª   ª           hist.py
ª   ª   ª       ª   ª           misc.py
ª   ª   ª       ª   ª           style.py
ª   ª   ª       ª   ª           timeseries.py
ª   ª   ª       ª   ª           tools.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---tests
ª   ª   ª       ª   ª   ª   test_aggregation.py
ª   ª   ª       ª   ª   ª   test_algos.py
ª   ª   ª       ª   ª   ª   test_common.py
ª   ª   ª       ª   ª   ª   test_downstream.py
ª   ª   ª       ª   ª   ª   test_errors.py
ª   ª   ª       ª   ª   ª   test_expressions.py
ª   ª   ª       ª   ª   ª   test_flags.py
ª   ª   ª       ª   ª   ª   test_multilevel.py
ª   ª   ª       ª   ª   ª   test_nanops.py
ª   ª   ª       ª   ª   ª   test_optional_dependency.py
ª   ª   ª       ª   ª   ª   test_register_accessor.py
ª   ª   ª       ª   ª   ª   test_sorting.py
ª   ª   ª       ª   ª   ª   test_take.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---api
ª   ª   ª       ª   ª   ª       test_api.py
ª   ª   ª       ª   ª   ª       test_types.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---apply
ª   ª   ª       ª   ª   ª       common.py
ª   ª   ª       ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª       test_frame_apply.py
ª   ª   ª       ª   ª   ª       test_frame_apply_relabeling.py
ª   ª   ª       ª   ª   ª       test_frame_transform.py
ª   ª   ª       ª   ª   ª       test_invalid_arg.py
ª   ª   ª       ª   ª   ª       test_series_apply.py
ª   ª   ª       ª   ª   ª       test_series_apply_relabeling.py
ª   ª   ª       ª   ª   ª       test_series_transform.py
ª   ª   ª       ª   ª   ª       test_str.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---arithmetic
ª   ª   ª       ª   ª   ª       common.py
ª   ª   ª       ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª       test_array_ops.py
ª   ª   ª       ª   ª   ª       test_categorical.py
ª   ª   ª       ª   ª   ª       test_datetime64.py
ª   ª   ª       ª   ª   ª       test_interval.py
ª   ª   ª       ª   ª   ª       test_numeric.py
ª   ª   ª       ª   ª   ª       test_object.py
ª   ª   ª       ª   ª   ª       test_period.py
ª   ª   ª       ª   ª   ª       test_timedelta64.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---arrays
ª   ª   ª       ª   ª   ª   ª   masked_shared.py
ª   ª   ª       ª   ª   ª   ª   test_array.py
ª   ª   ª       ª   ª   ª   ª   test_datetimelike.py
ª   ª   ª       ª   ª   ª   ª   test_datetimes.py
ª   ª   ª       ª   ª   ª   ª   test_ndarray_backed.py
ª   ª   ª       ª   ª   ª   ª   test_period.py
ª   ª   ª       ª   ª   ª   ª   test_timedeltas.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---boolean
ª   ª   ª       ª   ª   ª   ª       test_arithmetic.py
ª   ª   ª       ª   ª   ª   ª       test_astype.py
ª   ª   ª       ª   ª   ª   ª       test_comparison.py
ª   ª   ª       ª   ª   ª   ª       test_construction.py
ª   ª   ª       ª   ª   ª   ª       test_function.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_logical.py
ª   ª   ª       ª   ª   ª   ª       test_ops.py
ª   ª   ª       ª   ª   ª   ª       test_reduction.py
ª   ª   ª       ª   ª   ª   ª       test_repr.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---categorical
ª   ª   ª       ª   ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª   ª       test_algos.py
ª   ª   ª       ª   ª   ª   ª       test_analytics.py
ª   ª   ª       ª   ª   ª   ª       test_api.py
ª   ª   ª       ª   ª   ª   ª       test_astype.py
ª   ª   ª       ª   ª   ª   ª       test_constructors.py
ª   ª   ª       ª   ª   ª   ª       test_dtypes.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_missing.py
ª   ª   ª       ª   ª   ª   ª       test_operators.py
ª   ª   ª       ª   ª   ª   ª       test_replace.py
ª   ª   ª       ª   ª   ª   ª       test_repr.py
ª   ª   ª       ª   ª   ª   ª       test_sorting.py
ª   ª   ª       ª   ª   ª   ª       test_subclass.py
ª   ª   ª       ª   ª   ª   ª       test_take.py
ª   ª   ª       ª   ª   ª   ª       test_warnings.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---datetimes
ª   ª   ª       ª   ª   ª   ª       test_constructors.py
ª   ª   ª       ª   ª   ª   ª       test_cumulative.py
ª   ª   ª       ª   ª   ª   ª       test_reductions.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---floating
ª   ª   ª       ª   ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª   ª       test_arithmetic.py
ª   ª   ª       ª   ª   ª   ª       test_astype.py
ª   ª   ª       ª   ª   ª   ª       test_comparison.py
ª   ª   ª       ª   ª   ª   ª       test_concat.py
ª   ª   ª       ª   ª   ª   ª       test_construction.py
ª   ª   ª       ª   ª   ª   ª       test_function.py
ª   ª   ª       ª   ª   ª   ª       test_repr.py
ª   ª   ª       ª   ª   ª   ª       test_to_numpy.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---integer
ª   ª   ª       ª   ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª   ª       test_arithmetic.py
ª   ª   ª       ª   ª   ª   ª       test_comparison.py
ª   ª   ª       ª   ª   ª   ª       test_concat.py
ª   ª   ª       ª   ª   ª   ª       test_construction.py
ª   ª   ª       ª   ª   ª   ª       test_dtypes.py
ª   ª   ª       ª   ª   ª   ª       test_function.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_repr.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---interval
ª   ª   ª       ª   ª   ª   ª       test_astype.py
ª   ª   ª       ª   ª   ª   ª       test_interval.py
ª   ª   ª       ª   ª   ª   ª       test_ops.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---masked
ª   ª   ª       ª   ª   ª   ª       test_arithmetic.py
ª   ª   ª       ª   ª   ª   ª       test_arrow_compat.py
ª   ª   ª       ª   ª   ª   ª       test_function.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---numpy_
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_numpy.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---period
ª   ª   ª       ª   ª   ª   ª       test_arrow_compat.py
ª   ª   ª       ª   ª   ª   ª       test_astype.py
ª   ª   ª       ª   ª   ª   ª       test_constructors.py
ª   ª   ª       ª   ª   ª   ª       test_reductions.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---sparse
ª   ª   ª       ª   ª   ª   ª       test_accessor.py
ª   ª   ª       ª   ª   ª   ª       test_arithmetics.py
ª   ª   ª       ª   ª   ª   ª       test_array.py
ª   ª   ª       ª   ª   ª   ª       test_astype.py
ª   ª   ª       ª   ª   ª   ª       test_combine_concat.py
ª   ª   ª       ª   ª   ª   ª       test_constructors.py
ª   ª   ª       ª   ª   ª   ª       test_dtype.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_libsparse.py
ª   ª   ª       ª   ª   ª   ª       test_reductions.py
ª   ª   ª       ª   ª   ª   ª       test_unary.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---string_
ª   ª   ª       ª   ª   ª   ª       test_string.py
ª   ª   ª       ª   ª   ª   ª       test_string_arrow.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---timedeltas
ª   ª   ª       ª   ª   ª           test_constructors.py
ª   ª   ª       ª   ª   ª           test_cumulative.py
ª   ª   ª       ª   ª   ª           test_reductions.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---base
ª   ª   ª       ª   ª   ª       common.py
ª   ª   ª       ª   ª   ª       test_constructors.py
ª   ª   ª       ª   ª   ª       test_conversion.py
ª   ª   ª       ª   ª   ª       test_fillna.py
ª   ª   ª       ª   ª   ª       test_misc.py
ª   ª   ª       ª   ª   ª       test_transpose.py
ª   ª   ª       ª   ª   ª       test_unique.py
ª   ª   ª       ª   ª   ª       test_value_counts.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---computation
ª   ª   ª       ª   ª   ª       test_compat.py
ª   ª   ª       ª   ª   ª       test_eval.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---config
ª   ª   ª       ª   ª   ª       test_config.py
ª   ª   ª       ª   ª   ª       test_localization.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---construction
ª   ª   ª       ª   ª   ª       test_extract_array.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---copy_view
ª   ª   ª       ª   ª   ª   ª   test_array.py
ª   ª   ª       ª   ª   ª   ª   test_astype.py
ª   ª   ª       ª   ª   ª   ª   test_clip.py
ª   ª   ª       ª   ª   ª   ª   test_constructors.py
ª   ª   ª       ª   ª   ª   ª   test_core_functionalities.py
ª   ª   ª       ª   ª   ª   ª   test_functions.py
ª   ª   ª       ª   ª   ª   ª   test_indexing.py
ª   ª   ª       ª   ª   ª   ª   test_internals.py
ª   ª   ª       ª   ª   ª   ª   test_interp_fillna.py
ª   ª   ª       ª   ª   ª   ª   test_methods.py
ª   ª   ª       ª   ª   ª   ª   test_replace.py
ª   ª   ª       ª   ª   ª   ª   test_setitem.py
ª   ª   ª       ª   ª   ª   ª   test_util.py
ª   ª   ª       ª   ª   ª   ª   util.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---index
ª   ª   ª       ª   ª   ª           test_datetimeindex.py
ª   ª   ª       ª   ª   ª           test_index.py
ª   ª   ª       ª   ª   ª           test_periodindex.py
ª   ª   ª       ª   ª   ª           test_timedeltaindex.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---dtypes
ª   ª   ª       ª   ª   ª   ª   test_common.py
ª   ª   ª       ª   ª   ª   ª   test_concat.py
ª   ª   ª       ª   ª   ª   ª   test_dtypes.py
ª   ª   ª       ª   ª   ª   ª   test_generic.py
ª   ª   ª       ª   ª   ª   ª   test_inference.py
ª   ª   ª       ª   ª   ª   ª   test_missing.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---cast
ª   ª   ª       ª   ª   ª           test_can_hold_element.py
ª   ª   ª       ª   ª   ª           test_construct_from_scalar.py
ª   ª   ª       ª   ª   ª           test_construct_ndarray.py
ª   ª   ª       ª   ª   ª           test_construct_object_arr.py
ª   ª   ª       ª   ª   ª           test_dict_compat.py
ª   ª   ª       ª   ª   ª           test_downcast.py
ª   ª   ª       ª   ª   ª           test_find_common_type.py
ª   ª   ª       ª   ª   ª           test_infer_datetimelike.py
ª   ª   ª       ª   ª   ª           test_infer_dtype.py
ª   ª   ª       ª   ª   ª           test_maybe_box_native.py
ª   ª   ª       ª   ª   ª           test_promote.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---extension
ª   ª   ª       ª   ª   ª   ª   conftest.py
ª   ª   ª       ª   ª   ª   ª   test_arrow.py
ª   ª   ª       ª   ª   ª   ª   test_boolean.py
ª   ª   ª       ª   ª   ª   ª   test_categorical.py
ª   ª   ª       ª   ª   ª   ª   test_common.py
ª   ª   ª       ª   ª   ª   ª   test_datetime.py
ª   ª   ª       ª   ª   ª   ª   test_extension.py
ª   ª   ª       ª   ª   ª   ª   test_external_block.py
ª   ª   ª       ª   ª   ª   ª   test_floating.py
ª   ª   ª       ª   ª   ª   ª   test_integer.py
ª   ª   ª       ª   ª   ª   ª   test_interval.py
ª   ª   ª       ª   ª   ª   ª   test_numpy.py
ª   ª   ª       ª   ª   ª   ª   test_period.py
ª   ª   ª       ª   ª   ª   ª   test_sparse.py
ª   ª   ª       ª   ª   ª   ª   test_string.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---array_with_attr
ª   ª   ª       ª   ª   ª   ª       array.py
ª   ª   ª       ª   ª   ª   ª       test_array_with_attr.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---base
ª   ª   ª       ª   ª   ª   ª       accumulate.py
ª   ª   ª       ª   ª   ª   ª       base.py
ª   ª   ª       ª   ª   ª   ª       casting.py
ª   ª   ª       ª   ª   ª   ª       constructors.py
ª   ª   ª       ª   ª   ª   ª       dim2.py
ª   ª   ª       ª   ª   ª   ª       dtype.py
ª   ª   ª       ª   ª   ª   ª       getitem.py
ª   ª   ª       ª   ª   ª   ª       groupby.py
ª   ª   ª       ª   ª   ª   ª       index.py
ª   ª   ª       ª   ª   ª   ª       interface.py
ª   ª   ª       ª   ª   ª   ª       io.py
ª   ª   ª       ª   ª   ª   ª       methods.py
ª   ª   ª       ª   ª   ª   ª       missing.py
ª   ª   ª       ª   ª   ª   ª       ops.py
ª   ª   ª       ª   ª   ª   ª       printing.py
ª   ª   ª       ª   ª   ª   ª       reduce.py
ª   ª   ª       ª   ª   ª   ª       reshaping.py
ª   ª   ª       ª   ª   ª   ª       setitem.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---date
ª   ª   ª       ª   ª   ª   ª       array.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---decimal
ª   ª   ª       ª   ª   ª   ª       array.py
ª   ª   ª       ª   ª   ª   ª       test_decimal.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---json
ª   ª   ª       ª   ª   ª   ª       array.py
ª   ª   ª       ª   ª   ª   ª       test_json.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---list
ª   ª   ª       ª   ª   ª           array.py
ª   ª   ª       ª   ª   ª           test_list.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---frame
ª   ª   ª       ª   ª   ª   ª   common.py
ª   ª   ª       ª   ª   ª   ª   conftest.py
ª   ª   ª       ª   ª   ª   ª   test_alter_axes.py
ª   ª   ª       ª   ª   ª   ª   test_api.py
ª   ª   ª       ª   ª   ª   ª   test_arithmetic.py
ª   ª   ª       ª   ª   ª   ª   test_block_internals.py
ª   ª   ª       ª   ª   ª   ª   test_constructors.py
ª   ª   ª       ª   ª   ª   ª   test_cumulative.py
ª   ª   ª       ª   ª   ª   ª   test_iteration.py
ª   ª   ª       ª   ª   ª   ª   test_logical_ops.py
ª   ª   ª       ª   ª   ª   ª   test_nonunique_indexes.py
ª   ª   ª       ª   ª   ª   ª   test_npfuncs.py
ª   ª   ª       ª   ª   ª   ª   test_query_eval.py
ª   ª   ª       ª   ª   ª   ª   test_reductions.py
ª   ª   ª       ª   ª   ª   ª   test_repr_info.py
ª   ª   ª       ª   ª   ª   ª   test_stack_unstack.py
ª   ª   ª       ª   ª   ª   ª   test_subclass.py
ª   ª   ª       ª   ª   ª   ª   test_ufunc.py
ª   ª   ª       ª   ª   ª   ª   test_unary.py
ª   ª   ª       ª   ª   ª   ª   test_validate.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---constructors
ª   ª   ª       ª   ª   ª   ª       test_from_dict.py
ª   ª   ª       ª   ª   ª   ª       test_from_records.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---indexing
ª   ª   ª       ª   ª   ª   ª       test_coercion.py
ª   ª   ª       ª   ª   ª   ª       test_delitem.py
ª   ª   ª       ª   ª   ª   ª       test_get.py
ª   ª   ª       ª   ª   ª   ª       test_getitem.py
ª   ª   ª       ª   ª   ª   ª       test_get_value.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_insert.py
ª   ª   ª       ª   ª   ª   ª       test_mask.py
ª   ª   ª       ª   ª   ª   ª       test_setitem.py
ª   ª   ª       ª   ª   ª   ª       test_set_value.py
ª   ª   ª       ª   ª   ª   ª       test_take.py
ª   ª   ª       ª   ª   ª   ª       test_where.py
ª   ª   ª       ª   ª   ª   ª       test_xs.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---methods
ª   ª   ª       ª   ª   ª           test_add_prefix_suffix.py
ª   ª   ª       ª   ª   ª           test_align.py
ª   ª   ª       ª   ª   ª           test_asfreq.py
ª   ª   ª       ª   ª   ª           test_asof.py
ª   ª   ª       ª   ª   ª           test_assign.py
ª   ª   ª       ª   ª   ª           test_astype.py
ª   ª   ª       ª   ª   ª           test_at_time.py
ª   ª   ª       ª   ª   ª           test_between_time.py
ª   ª   ª       ª   ª   ª           test_clip.py
ª   ª   ª       ª   ª   ª           test_combine.py
ª   ª   ª       ª   ª   ª           test_combine_first.py
ª   ª   ª       ª   ª   ª           test_compare.py
ª   ª   ª       ª   ª   ª           test_convert_dtypes.py
ª   ª   ª       ª   ª   ª           test_copy.py
ª   ª   ª       ª   ª   ª           test_count.py
ª   ª   ª       ª   ª   ª           test_cov_corr.py
ª   ª   ª       ª   ª   ª           test_describe.py
ª   ª   ª       ª   ª   ª           test_diff.py
ª   ª   ª       ª   ª   ª           test_dot.py
ª   ª   ª       ª   ª   ª           test_drop.py
ª   ª   ª       ª   ª   ª           test_droplevel.py
ª   ª   ª       ª   ª   ª           test_dropna.py
ª   ª   ª       ª   ª   ª           test_drop_duplicates.py
ª   ª   ª       ª   ª   ª           test_dtypes.py
ª   ª   ª       ª   ª   ª           test_duplicated.py
ª   ª   ª       ª   ª   ª           test_equals.py
ª   ª   ª       ª   ª   ª           test_explode.py
ª   ª   ª       ª   ª   ª           test_fillna.py
ª   ª   ª       ª   ª   ª           test_filter.py
ª   ª   ª       ª   ª   ª           test_first_and_last.py
ª   ª   ª       ª   ª   ª           test_first_valid_index.py
ª   ª   ª       ª   ª   ª           test_get_numeric_data.py
ª   ª   ª       ª   ª   ª           test_head_tail.py
ª   ª   ª       ª   ª   ª           test_infer_objects.py
ª   ª   ª       ª   ª   ª           test_interpolate.py
ª   ª   ª       ª   ª   ª           test_isetitem.py
ª   ª   ª       ª   ª   ª           test_isin.py
ª   ª   ª       ª   ª   ª           test_is_homogeneous_dtype.py
ª   ª   ª       ª   ª   ª           test_join.py
ª   ª   ª       ª   ª   ª           test_matmul.py
ª   ª   ª       ª   ª   ª           test_nlargest.py
ª   ª   ª       ª   ª   ª           test_pct_change.py
ª   ª   ª       ª   ª   ª           test_pipe.py
ª   ª   ª       ª   ª   ª           test_pop.py
ª   ª   ª       ª   ª   ª           test_quantile.py
ª   ª   ª       ª   ª   ª           test_rank.py
ª   ª   ª       ª   ª   ª           test_reindex.py
ª   ª   ª       ª   ª   ª           test_reindex_like.py
ª   ª   ª       ª   ª   ª           test_rename.py
ª   ª   ª       ª   ª   ª           test_rename_axis.py
ª   ª   ª       ª   ª   ª           test_reorder_levels.py
ª   ª   ª       ª   ª   ª           test_replace.py
ª   ª   ª       ª   ª   ª           test_reset_index.py
ª   ª   ª       ª   ª   ª           test_round.py
ª   ª   ª       ª   ª   ª           test_sample.py
ª   ª   ª       ª   ª   ª           test_select_dtypes.py
ª   ª   ª       ª   ª   ª           test_set_axis.py
ª   ª   ª       ª   ª   ª           test_set_index.py
ª   ª   ª       ª   ª   ª           test_shift.py
ª   ª   ª       ª   ª   ª           test_sort_index.py
ª   ª   ª       ª   ª   ª           test_sort_values.py
ª   ª   ª       ª   ª   ª           test_swapaxes.py
ª   ª   ª       ª   ª   ª           test_swaplevel.py
ª   ª   ª       ª   ª   ª           test_to_csv.py
ª   ª   ª       ª   ª   ª           test_to_dict.py
ª   ª   ª       ª   ª   ª           test_to_dict_of_blocks.py
ª   ª   ª       ª   ª   ª           test_to_numpy.py
ª   ª   ª       ª   ª   ª           test_to_period.py
ª   ª   ª       ª   ª   ª           test_to_records.py
ª   ª   ª       ª   ª   ª           test_to_timestamp.py
ª   ª   ª       ª   ª   ª           test_transpose.py
ª   ª   ª       ª   ª   ª           test_truncate.py
ª   ª   ª       ª   ª   ª           test_tz_convert.py
ª   ª   ª       ª   ª   ª           test_tz_localize.py
ª   ª   ª       ª   ª   ª           test_update.py
ª   ª   ª       ª   ª   ª           test_values.py
ª   ª   ª       ª   ª   ª           test_value_counts.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---generic
ª   ª   ª       ª   ª   ª       test_duplicate_labels.py
ª   ª   ª       ª   ª   ª       test_finalize.py
ª   ª   ª       ª   ª   ª       test_frame.py
ª   ª   ª       ª   ª   ª       test_generic.py
ª   ª   ª       ª   ª   ª       test_label_or_level_utils.py
ª   ª   ª       ª   ª   ª       test_series.py
ª   ª   ª       ª   ª   ª       test_to_xarray.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---groupby
ª   ª   ª       ª   ª   ª   ª   conftest.py
ª   ª   ª       ª   ª   ª   ª   test_allowlist.py
ª   ª   ª       ª   ª   ª   ª   test_any_all.py
ª   ª   ª       ª   ª   ª   ª   test_api_consistency.py
ª   ª   ª       ª   ª   ª   ª   test_apply.py
ª   ª   ª       ª   ª   ª   ª   test_apply_mutate.py
ª   ª   ª       ª   ª   ª   ª   test_bin_groupby.py
ª   ª   ª       ª   ª   ª   ª   test_categorical.py
ª   ª   ª       ª   ª   ª   ª   test_counting.py
ª   ª   ª       ª   ª   ª   ª   test_filters.py
ª   ª   ª       ª   ª   ª   ª   test_function.py
ª   ª   ª       ª   ª   ª   ª   test_groupby.py
ª   ª   ª       ª   ª   ª   ª   test_groupby_dropna.py
ª   ª   ª       ª   ª   ª   ª   test_groupby_shift_diff.py
ª   ª   ª       ª   ª   ª   ª   test_groupby_subclass.py
ª   ª   ª       ª   ª   ª   ª   test_grouping.py
ª   ª   ª       ª   ª   ª   ª   test_indexing.py
ª   ª   ª       ª   ª   ª   ª   test_index_as_string.py
ª   ª   ª       ª   ª   ª   ª   test_libgroupby.py
ª   ª   ª       ª   ª   ª   ª   test_min_max.py
ª   ª   ª       ª   ª   ª   ª   test_missing.py
ª   ª   ª       ª   ª   ª   ª   test_nth.py
ª   ª   ª       ª   ª   ª   ª   test_numba.py
ª   ª   ª       ª   ª   ª   ª   test_nunique.py
ª   ª   ª       ª   ª   ª   ª   test_pipe.py
ª   ª   ª       ª   ª   ª   ª   test_quantile.py
ª   ª   ª       ª   ª   ª   ª   test_raises.py
ª   ª   ª       ª   ª   ª   ª   test_rank.py
ª   ª   ª       ª   ª   ª   ª   test_sample.py
ª   ª   ª       ª   ª   ª   ª   test_size.py
ª   ª   ª       ª   ª   ª   ª   test_timegrouper.py
ª   ª   ª       ª   ª   ª   ª   test_value_counts.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---aggregate
ª   ª   ª       ª   ª   ª   ª       test_aggregate.py
ª   ª   ª       ª   ª   ª   ª       test_cython.py
ª   ª   ª       ª   ª   ª   ª       test_numba.py
ª   ª   ª       ª   ª   ª   ª       test_other.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---transform
ª   ª   ª       ª   ª   ª           test_numba.py
ª   ª   ª       ª   ª   ª           test_transform.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---indexes
ª   ª   ª       ª   ª   ª   ª   common.py
ª   ª   ª       ª   ª   ª   ª   conftest.py
ª   ª   ª       ª   ª   ª   ª   datetimelike.py
ª   ª   ª       ª   ª   ª   ª   test_any_index.py
ª   ª   ª       ª   ª   ª   ª   test_base.py
ª   ª   ª       ª   ª   ª   ª   test_common.py
ª   ª   ª       ª   ª   ª   ª   test_engines.py
ª   ª   ª       ª   ª   ª   ª   test_frozen.py
ª   ª   ª       ª   ª   ª   ª   test_indexing.py
ª   ª   ª       ª   ª   ª   ª   test_index_new.py
ª   ª   ª       ª   ª   ª   ª   test_numpy_compat.py
ª   ª   ª       ª   ª   ª   ª   test_setops.py
ª   ª   ª       ª   ª   ª   ª   test_subclass.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---base_class
ª   ª   ª       ª   ª   ª   ª       test_constructors.py
ª   ª   ª       ª   ª   ª   ª       test_formats.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_pickle.py
ª   ª   ª       ª   ª   ª   ª       test_reshape.py
ª   ª   ª       ª   ª   ª   ª       test_setops.py
ª   ª   ª       ª   ª   ª   ª       test_where.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---categorical
ª   ª   ª       ª   ª   ª   ª       test_append.py
ª   ª   ª       ª   ª   ª   ª       test_astype.py
ª   ª   ª       ª   ª   ª   ª       test_category.py
ª   ª   ª       ª   ª   ª   ª       test_constructors.py
ª   ª   ª       ª   ª   ª   ª       test_equals.py
ª   ª   ª       ª   ª   ª   ª       test_fillna.py
ª   ª   ª       ª   ª   ª   ª       test_formats.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_map.py
ª   ª   ª       ª   ª   ª   ª       test_reindex.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---datetimelike_
ª   ª   ª       ª   ª   ª   ª       test_drop_duplicates.py
ª   ª   ª       ª   ª   ª   ª       test_equals.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_is_monotonic.py
ª   ª   ª       ª   ª   ª   ª       test_nat.py
ª   ª   ª       ª   ª   ª   ª       test_sort_values.py
ª   ª   ª       ª   ª   ª   ª       test_value_counts.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---datetimes
ª   ª   ª       ª   ª   ª   ª   ª   test_asof.py
ª   ª   ª       ª   ª   ª   ª   ª   test_constructors.py
ª   ª   ª       ª   ª   ª   ª   ª   test_datetime.py
ª   ª   ª       ª   ª   ª   ª   ª   test_datetimelike.py
ª   ª   ª       ª   ª   ª   ª   ª   test_date_range.py
ª   ª   ª       ª   ª   ª   ª   ª   test_delete.py
ª   ª   ª       ª   ª   ª   ª   ª   test_formats.py
ª   ª   ª       ª   ª   ª   ª   ª   test_freq_attr.py
ª   ª   ª       ª   ª   ª   ª   ª   test_indexing.py
ª   ª   ª       ª   ª   ª   ª   ª   test_join.py
ª   ª   ª       ª   ª   ª   ª   ª   test_map.py
ª   ª   ª       ª   ª   ª   ª   ª   test_misc.py
ª   ª   ª       ª   ª   ª   ª   ª   test_npfuncs.py
ª   ª   ª       ª   ª   ª   ª   ª   test_ops.py
ª   ª   ª       ª   ª   ª   ª   ª   test_partial_slicing.py
ª   ª   ª       ª   ª   ª   ª   ª   test_pickle.py
ª   ª   ª       ª   ª   ª   ª   ª   test_reindex.py
ª   ª   ª       ª   ª   ª   ª   ª   test_scalar_compat.py
ª   ª   ª       ª   ª   ª   ª   ª   test_setops.py
ª   ª   ª       ª   ª   ª   ª   ª   test_timezones.py
ª   ª   ª       ª   ª   ª   ª   ª   test_unique.py
ª   ª   ª       ª   ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   ª   +---methods
ª   ª   ª       ª   ª   ª   ª           test_astype.py
ª   ª   ª       ª   ª   ª   ª           test_factorize.py
ª   ª   ª       ª   ª   ª   ª           test_fillna.py
ª   ª   ª       ª   ª   ª   ª           test_insert.py
ª   ª   ª       ª   ª   ª   ª           test_isocalendar.py
ª   ª   ª       ª   ª   ª   ª           test_repeat.py
ª   ª   ª       ª   ª   ª   ª           test_shift.py
ª   ª   ª       ª   ª   ª   ª           test_snap.py
ª   ª   ª       ª   ª   ª   ª           test_to_frame.py
ª   ª   ª       ª   ª   ª   ª           test_to_period.py
ª   ª   ª       ª   ª   ª   ª           test_to_series.py
ª   ª   ª       ª   ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª   ª           
ª   ª   ª       ª   ª   ª   +---interval
ª   ª   ª       ª   ª   ª   ª       test_astype.py
ª   ª   ª       ª   ª   ª   ª       test_base.py
ª   ª   ª       ª   ª   ª   ª       test_constructors.py
ª   ª   ª       ª   ª   ª   ª       test_equals.py
ª   ª   ª       ª   ª   ª   ª       test_formats.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_interval.py
ª   ª   ª       ª   ª   ª   ª       test_interval_range.py
ª   ª   ª       ª   ª   ª   ª       test_interval_tree.py
ª   ª   ª       ª   ª   ª   ª       test_join.py
ª   ª   ª       ª   ª   ª   ª       test_pickle.py
ª   ª   ª       ª   ª   ª   ª       test_setops.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---multi
ª   ª   ª       ª   ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª   ª       test_analytics.py
ª   ª   ª       ª   ª   ª   ª       test_astype.py
ª   ª   ª       ª   ª   ª   ª       test_compat.py
ª   ª   ª       ª   ª   ª   ª       test_constructors.py
ª   ª   ª       ª   ª   ª   ª       test_conversion.py
ª   ª   ª       ª   ª   ª   ª       test_copy.py
ª   ª   ª       ª   ª   ª   ª       test_drop.py
ª   ª   ª       ª   ª   ª   ª       test_duplicates.py
ª   ª   ª       ª   ª   ª   ª       test_equivalence.py
ª   ª   ª       ª   ª   ª   ª       test_formats.py
ª   ª   ª       ª   ª   ª   ª       test_get_level_values.py
ª   ª   ª       ª   ª   ª   ª       test_get_set.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_integrity.py
ª   ª   ª       ª   ª   ª   ª       test_isin.py
ª   ª   ª       ª   ª   ª   ª       test_join.py
ª   ª   ª       ª   ª   ª   ª       test_lexsort.py
ª   ª   ª       ª   ª   ª   ª       test_missing.py
ª   ª   ª       ª   ª   ª   ª       test_monotonic.py
ª   ª   ª       ª   ª   ª   ª       test_names.py
ª   ª   ª       ª   ª   ª   ª       test_partial_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_pickle.py
ª   ª   ª       ª   ª   ª   ª       test_reindex.py
ª   ª   ª       ª   ª   ª   ª       test_reshape.py
ª   ª   ª       ª   ª   ª   ª       test_setops.py
ª   ª   ª       ª   ª   ª   ª       test_sorting.py
ª   ª   ª       ª   ª   ª   ª       test_take.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---numeric
ª   ª   ª       ª   ª   ª   ª       test_astype.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_join.py
ª   ª   ª       ª   ª   ª   ª       test_numeric.py
ª   ª   ª       ª   ª   ª   ª       test_setops.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---object
ª   ª   ª       ª   ª   ª   ª       test_astype.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---period
ª   ª   ª       ª   ª   ª   ª   ª   test_constructors.py
ª   ª   ª       ª   ª   ª   ª   ª   test_formats.py
ª   ª   ª       ª   ª   ª   ª   ª   test_freq_attr.py
ª   ª   ª       ª   ª   ª   ª   ª   test_indexing.py
ª   ª   ª       ª   ª   ª   ª   ª   test_join.py
ª   ª   ª       ª   ª   ª   ª   ª   test_monotonic.py
ª   ª   ª       ª   ª   ª   ª   ª   test_partial_slicing.py
ª   ª   ª       ª   ª   ª   ª   ª   test_period.py
ª   ª   ª       ª   ª   ª   ª   ª   test_period_range.py
ª   ª   ª       ª   ª   ª   ª   ª   test_pickle.py
ª   ª   ª       ª   ª   ª   ª   ª   test_resolution.py
ª   ª   ª       ª   ª   ª   ª   ª   test_scalar_compat.py
ª   ª   ª       ª   ª   ª   ª   ª   test_searchsorted.py
ª   ª   ª       ª   ª   ª   ª   ª   test_setops.py
ª   ª   ª       ª   ª   ª   ª   ª   test_tools.py
ª   ª   ª       ª   ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   ª   +---methods
ª   ª   ª       ª   ª   ª   ª           test_asfreq.py
ª   ª   ª       ª   ª   ª   ª           test_astype.py
ª   ª   ª       ª   ª   ª   ª           test_factorize.py
ª   ª   ª       ª   ª   ª   ª           test_fillna.py
ª   ª   ª       ª   ª   ª   ª           test_insert.py
ª   ª   ª       ª   ª   ª   ª           test_is_full.py
ª   ª   ª       ª   ª   ª   ª           test_repeat.py
ª   ª   ª       ª   ª   ª   ª           test_shift.py
ª   ª   ª       ª   ª   ª   ª           test_to_timestamp.py
ª   ª   ª       ª   ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª   ª           
ª   ª   ª       ª   ª   ª   +---ranges
ª   ª   ª       ª   ª   ª   ª       test_constructors.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_join.py
ª   ª   ª       ª   ª   ª   ª       test_range.py
ª   ª   ª       ª   ª   ª   ª       test_setops.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---timedeltas
ª   ª   ª       ª   ª   ª       ª   test_constructors.py
ª   ª   ª       ª   ª   ª       ª   test_delete.py
ª   ª   ª       ª   ª   ª       ª   test_formats.py
ª   ª   ª       ª   ª   ª       ª   test_freq_attr.py
ª   ª   ª       ª   ª   ª       ª   test_indexing.py
ª   ª   ª       ª   ª   ª       ª   test_join.py
ª   ª   ª       ª   ª   ª       ª   test_ops.py
ª   ª   ª       ª   ª   ª       ª   test_pickle.py
ª   ª   ª       ª   ª   ª       ª   test_scalar_compat.py
ª   ª   ª       ª   ª   ª       ª   test_searchsorted.py
ª   ª   ª       ª   ª   ª       ª   test_setops.py
ª   ª   ª       ª   ª   ª       ª   test_timedelta.py
ª   ª   ª       ª   ª   ª       ª   test_timedelta_range.py
ª   ª   ª       ª   ª   ª       ª   __init__.py
ª   ª   ª       ª   ª   ª       ª   
ª   ª   ª       ª   ª   ª       +---methods
ª   ª   ª       ª   ª   ª               test_astype.py
ª   ª   ª       ª   ª   ª               test_factorize.py
ª   ª   ª       ª   ª   ª               test_fillna.py
ª   ª   ª       ª   ª   ª               test_insert.py
ª   ª   ª       ª   ª   ª               test_repeat.py
ª   ª   ª       ª   ª   ª               test_shift.py
ª   ª   ª       ª   ª   ª               __init__.py
ª   ª   ª       ª   ª   ª               
ª   ª   ª       ª   ª   +---indexing
ª   ª   ª       ª   ª   ª   ª   common.py
ª   ª   ª       ª   ª   ª   ª   conftest.py
ª   ª   ª       ª   ª   ª   ª   test_at.py
ª   ª   ª       ª   ª   ª   ª   test_categorical.py
ª   ª   ª       ª   ª   ª   ª   test_chaining_and_caching.py
ª   ª   ª       ª   ª   ª   ª   test_check_indexer.py
ª   ª   ª       ª   ª   ª   ª   test_coercion.py
ª   ª   ª       ª   ª   ª   ª   test_datetime.py
ª   ª   ª       ª   ª   ª   ª   test_floats.py
ª   ª   ª       ª   ª   ª   ª   test_iat.py
ª   ª   ª       ª   ª   ª   ª   test_iloc.py
ª   ª   ª       ª   ª   ª   ª   test_indexers.py
ª   ª   ª       ª   ª   ª   ª   test_indexing.py
ª   ª   ª       ª   ª   ª   ª   test_loc.py
ª   ª   ª       ª   ª   ª   ª   test_na_indexing.py
ª   ª   ª       ª   ª   ª   ª   test_partial.py
ª   ª   ª       ª   ª   ª   ª   test_scalar.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---interval
ª   ª   ª       ª   ª   ª   ª       test_interval.py
ª   ª   ª       ª   ª   ª   ª       test_interval_new.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---multiindex
ª   ª   ª       ª   ª   ª           test_chaining_and_caching.py
ª   ª   ª       ª   ª   ª           test_datetime.py
ª   ª   ª       ª   ª   ª           test_getitem.py
ª   ª   ª       ª   ª   ª           test_iloc.py
ª   ª   ª       ª   ª   ª           test_indexing_slow.py
ª   ª   ª       ª   ª   ª           test_loc.py
ª   ª   ª       ª   ª   ª           test_multiindex.py
ª   ª   ª       ª   ª   ª           test_partial.py
ª   ª   ª       ª   ª   ª           test_setitem.py
ª   ª   ª       ª   ª   ª           test_slice.py
ª   ª   ª       ª   ª   ª           test_sorted.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---interchange
ª   ª   ª       ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª       test_impl.py
ª   ª   ª       ª   ª   ª       test_spec_conformance.py
ª   ª   ª       ª   ª   ª       test_utils.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---internals
ª   ª   ª       ª   ª   ª       test_api.py
ª   ª   ª       ª   ª   ª       test_internals.py
ª   ª   ª       ª   ª   ª       test_managers.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---io
ª   ª   ª       ª   ª   ª   ª   conftest.py
ª   ª   ª       ª   ª   ª   ª   generate_legacy_storage_files.py
ª   ª   ª       ª   ª   ª   ª   test_clipboard.py
ª   ª   ª       ª   ª   ª   ª   test_common.py
ª   ª   ª       ª   ª   ª   ª   test_compression.py
ª   ª   ª       ª   ª   ª   ª   test_feather.py
ª   ª   ª       ª   ª   ª   ª   test_fsspec.py
ª   ª   ª       ª   ª   ª   ª   test_gcs.py
ª   ª   ª       ª   ª   ª   ª   test_html.py
ª   ª   ª       ª   ª   ª   ª   test_orc.py
ª   ª   ª       ª   ª   ª   ª   test_parquet.py
ª   ª   ª       ª   ª   ª   ª   test_pickle.py
ª   ª   ª       ª   ª   ª   ª   test_s3.py
ª   ª   ª       ª   ª   ª   ª   test_spss.py
ª   ª   ª       ª   ª   ª   ª   test_sql.py
ª   ª   ª       ª   ª   ª   ª   test_stata.py
ª   ª   ª       ª   ª   ª   ª   test_user_agent.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---data
ª   ª   ª       ª   ª   ª   ª   ª   gbq_fake_job.txt
ª   ª   ª       ª   ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   ª   +---fixed_width
ª   ª   ª       ª   ª   ª   ª   ª       fixed_width_format.txt
ª   ª   ª       ª   ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   ª   +---legacy_pickle
ª   ª   ª       ª   ª   ª   ª   ª   +---1.2.4
ª   ª   ª       ª   ª   ª   ª   ª           empty_frame_v1_2_4-GH#42345.pkl
ª   ª   ª       ª   ª   ª   ª   ª           
ª   ª   ª       ª   ª   ª   ª   +---parquet
ª   ª   ª       ª   ª   ª   ª   ª       simple.parquet
ª   ª   ª       ª   ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   ª   +---pickle
ª   ª   ª       ª   ª   ª   ª   ª       test_mi_py27.pkl
ª   ª   ª       ª   ª   ª   ª   ª       test_py27.pkl
ª   ª   ª       ª   ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   ª   +---xml
ª   ª   ª       ª   ª   ª   ª           baby_names.xml
ª   ª   ª       ª   ª   ª   ª           books.xml
ª   ª   ª       ª   ª   ª   ª           cta_rail_lines.kml
ª   ª   ª       ª   ª   ª   ª           doc_ch_utf.xml
ª   ª   ª       ª   ª   ª   ª           flatten_doc.xsl
ª   ª   ª       ª   ª   ª   ª           row_field_output.xsl
ª   ª   ª       ª   ª   ª   ª           
ª   ª   ª       ª   ª   ª   +---excel
ª   ª   ª       ª   ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª   ª       test_odf.py
ª   ª   ª       ª   ª   ª   ª       test_odswriter.py
ª   ª   ª       ª   ª   ª   ª       test_openpyxl.py
ª   ª   ª       ª   ª   ª   ª       test_readers.py
ª   ª   ª       ª   ª   ª   ª       test_style.py
ª   ª   ª       ª   ª   ª   ª       test_writers.py
ª   ª   ª       ª   ª   ª   ª       test_xlrd.py
ª   ª   ª       ª   ª   ª   ª       test_xlsxwriter.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---formats
ª   ª   ª       ª   ª   ª   ª   ª   test_console.py
ª   ª   ª       ª   ª   ª   ª   ª   test_css.py
ª   ª   ª       ª   ª   ª   ª   ª   test_eng_formatting.py
ª   ª   ª       ª   ª   ª   ª   ª   test_format.py
ª   ª   ª       ª   ª   ª   ª   ª   test_info.py
ª   ª   ª       ª   ª   ª   ª   ª   test_printing.py
ª   ª   ª       ª   ª   ª   ª   ª   test_series_info.py
ª   ª   ª       ª   ª   ª   ª   ª   test_to_csv.py
ª   ª   ª       ª   ª   ª   ª   ª   test_to_excel.py
ª   ª   ª       ª   ª   ª   ª   ª   test_to_html.py
ª   ª   ª       ª   ª   ª   ª   ª   test_to_latex.py
ª   ª   ª       ª   ª   ª   ª   ª   test_to_markdown.py
ª   ª   ª       ª   ª   ª   ª   ª   test_to_string.py
ª   ª   ª       ª   ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   ª   +---style
ª   ª   ª       ª   ª   ª   ª           test_bar.py
ª   ª   ª       ª   ª   ª   ª           test_exceptions.py
ª   ª   ª       ª   ª   ª   ª           test_format.py
ª   ª   ª       ª   ª   ª   ª           test_highlight.py
ª   ª   ª       ª   ª   ª   ª           test_html.py
ª   ª   ª       ª   ª   ª   ª           test_matplotlib.py
ª   ª   ª       ª   ª   ª   ª           test_non_unique.py
ª   ª   ª       ª   ª   ª   ª           test_style.py
ª   ª   ª       ª   ª   ª   ª           test_tooltip.py
ª   ª   ª       ª   ª   ª   ª           test_to_latex.py
ª   ª   ª       ª   ª   ª   ª           test_to_string.py
ª   ª   ª       ª   ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª   ª           
ª   ª   ª       ª   ª   ª   +---json
ª   ª   ª       ª   ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª   ª       test_compression.py
ª   ª   ª       ª   ª   ª   ª       test_deprecated_kwargs.py
ª   ª   ª       ª   ª   ª   ª       test_json_table_schema.py
ª   ª   ª       ª   ª   ª   ª       test_json_table_schema_ext_dtype.py
ª   ª   ª       ª   ª   ª   ª       test_normalize.py
ª   ª   ª       ª   ª   ª   ª       test_pandas.py
ª   ª   ª       ª   ª   ª   ª       test_readlines.py
ª   ª   ª       ª   ª   ª   ª       test_ujson.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---parser
ª   ª   ª       ª   ª   ª   ª   ª   conftest.py
ª   ª   ª       ª   ª   ª   ª   ª   test_comment.py
ª   ª   ª       ª   ª   ª   ª   ª   test_compression.py
ª   ª   ª       ª   ª   ª   ª   ª   test_concatenate_chunks.py
ª   ª   ª       ª   ª   ª   ª   ª   test_converters.py
ª   ª   ª       ª   ª   ª   ª   ª   test_c_parser_only.py
ª   ª   ª       ª   ª   ª   ª   ª   test_dialect.py
ª   ª   ª       ª   ª   ª   ª   ª   test_encoding.py
ª   ª   ª       ª   ª   ª   ª   ª   test_header.py
ª   ª   ª       ª   ª   ª   ª   ª   test_index_col.py
ª   ª   ª       ª   ª   ª   ª   ª   test_mangle_dupes.py
ª   ª   ª       ª   ª   ª   ª   ª   test_multi_thread.py
ª   ª   ª       ª   ª   ª   ª   ª   test_na_values.py
ª   ª   ª       ª   ª   ª   ª   ª   test_network.py
ª   ª   ª       ª   ª   ª   ª   ª   test_parse_dates.py
ª   ª   ª       ª   ª   ª   ª   ª   test_python_parser_only.py
ª   ª   ª       ª   ª   ª   ª   ª   test_quoting.py
ª   ª   ª       ª   ª   ª   ª   ª   test_read_fwf.py
ª   ª   ª       ª   ª   ª   ª   ª   test_skiprows.py
ª   ª   ª       ª   ª   ª   ª   ª   test_textreader.py
ª   ª   ª       ª   ª   ª   ª   ª   test_unsupported.py
ª   ª   ª       ª   ª   ª   ª   ª   test_upcast.py
ª   ª   ª       ª   ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   ª   +---common
ª   ª   ª       ª   ª   ª   ª   ª       test_chunksize.py
ª   ª   ª       ª   ª   ª   ª   ª       test_common_basic.py
ª   ª   ª       ª   ª   ª   ª   ª       test_data_list.py
ª   ª   ª       ª   ª   ª   ª   ª       test_decimal.py
ª   ª   ª       ª   ª   ª   ª   ª       test_file_buffer_url.py
ª   ª   ª       ª   ª   ª   ª   ª       test_float.py
ª   ª   ª       ª   ª   ª   ª   ª       test_index.py
ª   ª   ª       ª   ª   ª   ª   ª       test_inf.py
ª   ª   ª       ª   ª   ª   ª   ª       test_ints.py
ª   ª   ª       ª   ª   ª   ª   ª       test_iterator.py
ª   ª   ª       ª   ª   ª   ª   ª       test_read_errors.py
ª   ª   ª       ª   ª   ª   ª   ª       test_verbose.py
ª   ª   ª       ª   ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   ª   +---dtypes
ª   ª   ª       ª   ª   ª   ª   ª       test_categorical.py
ª   ª   ª       ª   ª   ª   ª   ª       test_dtypes_basic.py
ª   ª   ª       ª   ª   ª   ª   ª       test_empty.py
ª   ª   ª       ª   ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   ª   +---usecols
ª   ª   ª       ª   ª   ª   ª           test_parse_dates.py
ª   ª   ª       ª   ª   ª   ª           test_strings.py
ª   ª   ª       ª   ª   ª   ª           test_usecols_basic.py
ª   ª   ª       ª   ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª   ª           
ª   ª   ª       ª   ª   ª   +---pytables
ª   ª   ª       ª   ª   ª   ª       common.py
ª   ª   ª       ª   ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª   ª       test_append.py
ª   ª   ª       ª   ª   ª   ª       test_categorical.py
ª   ª   ª       ª   ª   ª   ª       test_compat.py
ª   ª   ª       ª   ª   ª   ª       test_complex.py
ª   ª   ª       ª   ª   ª   ª       test_errors.py
ª   ª   ª       ª   ª   ª   ª       test_file_handling.py
ª   ª   ª       ª   ª   ª   ª       test_keys.py
ª   ª   ª       ª   ª   ª   ª       test_put.py
ª   ª   ª       ª   ª   ª   ª       test_pytables_missing.py
ª   ª   ª       ª   ª   ª   ª       test_read.py
ª   ª   ª       ª   ª   ª   ª       test_retain_attributes.py
ª   ª   ª       ª   ª   ª   ª       test_round_trip.py
ª   ª   ª       ª   ª   ª   ª       test_select.py
ª   ª   ª       ª   ª   ª   ª       test_store.py
ª   ª   ª       ª   ª   ª   ª       test_subclass.py
ª   ª   ª       ª   ª   ª   ª       test_timezones.py
ª   ª   ª       ª   ª   ª   ª       test_time_series.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---sas
ª   ª   ª       ª   ª   ª   ª       test_byteswap.py
ª   ª   ª       ª   ª   ª   ª       test_sas.py
ª   ª   ª       ª   ª   ª   ª       test_sas7bdat.py
ª   ª   ª       ª   ª   ª   ª       test_xport.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---xml
ª   ª   ª       ª   ª   ª           test_to_xml.py
ª   ª   ª       ª   ª   ª           test_xml.py
ª   ª   ª       ª   ª   ª           test_xml_dtypes.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---libs
ª   ª   ª       ª   ª   ª       test_hashtable.py
ª   ª   ª       ª   ª   ª       test_join.py
ª   ª   ª       ª   ª   ª       test_lib.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---plotting
ª   ª   ª       ª   ª   ª   ª   common.py
ª   ª   ª       ª   ª   ª   ª   conftest.py
ª   ª   ª       ª   ª   ª   ª   test_backend.py
ª   ª   ª       ª   ª   ª   ª   test_boxplot_method.py
ª   ª   ª       ª   ª   ª   ª   test_common.py
ª   ª   ª       ª   ª   ª   ª   test_converter.py
ª   ª   ª       ª   ª   ª   ª   test_datetimelike.py
ª   ª   ª       ª   ª   ª   ª   test_groupby.py
ª   ª   ª       ª   ª   ª   ª   test_hist_method.py
ª   ª   ª       ª   ª   ª   ª   test_misc.py
ª   ª   ª       ª   ª   ª   ª   test_series.py
ª   ª   ª       ª   ª   ª   ª   test_style.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---frame
ª   ª   ª       ª   ª   ª           test_frame.py
ª   ª   ª       ª   ª   ª           test_frame_color.py
ª   ª   ª       ª   ª   ª           test_frame_groupby.py
ª   ª   ª       ª   ª   ª           test_frame_legend.py
ª   ª   ª       ª   ª   ª           test_frame_subplots.py
ª   ª   ª       ª   ª   ª           test_hist_box_by.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---reductions
ª   ª   ª       ª   ª   ª       test_reductions.py
ª   ª   ª       ª   ª   ª       test_stat_reductions.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---resample
ª   ª   ª       ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª       test_base.py
ª   ª   ª       ª   ª   ª       test_datetime_index.py
ª   ª   ª       ª   ª   ª       test_period_index.py
ª   ª   ª       ª   ª   ª       test_resampler_grouper.py
ª   ª   ª       ª   ª   ª       test_resample_api.py
ª   ª   ª       ª   ª   ª       test_timedelta.py
ª   ª   ª       ª   ª   ª       test_time_grouper.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---reshape
ª   ª   ª       ª   ª   ª   ª   test_crosstab.py
ª   ª   ª       ª   ª   ª   ª   test_cut.py
ª   ª   ª       ª   ª   ª   ª   test_from_dummies.py
ª   ª   ª       ª   ª   ª   ª   test_get_dummies.py
ª   ª   ª       ª   ª   ª   ª   test_melt.py
ª   ª   ª       ª   ª   ª   ª   test_pivot.py
ª   ª   ª       ª   ª   ª   ª   test_pivot_multilevel.py
ª   ª   ª       ª   ª   ª   ª   test_qcut.py
ª   ª   ª       ª   ª   ª   ª   test_union_categoricals.py
ª   ª   ª       ª   ª   ª   ª   test_util.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---concat
ª   ª   ª       ª   ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª   ª       test_append.py
ª   ª   ª       ª   ª   ª   ª       test_append_common.py
ª   ª   ª       ª   ª   ª   ª       test_categorical.py
ª   ª   ª       ª   ª   ª   ª       test_concat.py
ª   ª   ª       ª   ª   ª   ª       test_dataframe.py
ª   ª   ª       ª   ª   ª   ª       test_datetimes.py
ª   ª   ª       ª   ª   ª   ª       test_empty.py
ª   ª   ª       ª   ª   ª   ª       test_index.py
ª   ª   ª       ª   ª   ª   ª       test_invalid.py
ª   ª   ª       ª   ª   ª   ª       test_series.py
ª   ª   ª       ª   ª   ª   ª       test_sort.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---merge
ª   ª   ª       ª   ª   ª           test_join.py
ª   ª   ª       ª   ª   ª           test_merge.py
ª   ª   ª       ª   ª   ª           test_merge_asof.py
ª   ª   ª       ª   ª   ª           test_merge_cross.py
ª   ª   ª       ª   ª   ª           test_merge_index_as_string.py
ª   ª   ª       ª   ª   ª           test_merge_ordered.py
ª   ª   ª       ª   ª   ª           test_multi.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---scalar
ª   ª   ª       ª   ª   ª   ª   test_nat.py
ª   ª   ª       ª   ª   ª   ª   test_na_scalar.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---interval
ª   ª   ª       ª   ª   ª   ª       test_arithmetic.py
ª   ª   ª       ª   ª   ª   ª       test_interval.py
ª   ª   ª       ª   ª   ª   ª       test_ops.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---period
ª   ª   ª       ª   ª   ª   ª       test_asfreq.py
ª   ª   ª       ª   ª   ª   ª       test_period.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---timedelta
ª   ª   ª       ª   ª   ª   ª       test_arithmetic.py
ª   ª   ª       ª   ª   ª   ª       test_constructors.py
ª   ª   ª       ª   ª   ª   ª       test_formats.py
ª   ª   ª       ª   ª   ª   ª       test_timedelta.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---timestamp
ª   ª   ª       ª   ª   ª           test_arithmetic.py
ª   ª   ª       ª   ª   ª           test_comparisons.py
ª   ª   ª       ª   ª   ª           test_constructors.py
ª   ª   ª       ª   ª   ª           test_formats.py
ª   ª   ª       ª   ª   ª           test_rendering.py
ª   ª   ª       ª   ª   ª           test_timestamp.py
ª   ª   ª       ª   ª   ª           test_timezones.py
ª   ª   ª       ª   ª   ª           test_unary_ops.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---series
ª   ª   ª       ª   ª   ª   ª   test_api.py
ª   ª   ª       ª   ª   ª   ª   test_arithmetic.py
ª   ª   ª       ª   ª   ª   ª   test_constructors.py
ª   ª   ª       ª   ª   ª   ª   test_cumulative.py
ª   ª   ª       ª   ª   ª   ª   test_iteration.py
ª   ª   ª       ª   ª   ª   ª   test_logical_ops.py
ª   ª   ª       ª   ª   ª   ª   test_missing.py
ª   ª   ª       ª   ª   ª   ª   test_npfuncs.py
ª   ª   ª       ª   ª   ª   ª   test_reductions.py
ª   ª   ª       ª   ª   ª   ª   test_repr.py
ª   ª   ª       ª   ª   ª   ª   test_subclass.py
ª   ª   ª       ª   ª   ª   ª   test_ufunc.py
ª   ª   ª       ª   ª   ª   ª   test_unary.py
ª   ª   ª       ª   ª   ª   ª   test_validate.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---accessors
ª   ª   ª       ª   ª   ª   ª       test_cat_accessor.py
ª   ª   ª       ª   ª   ª   ª       test_dt_accessor.py
ª   ª   ª       ª   ª   ª   ª       test_sparse_accessor.py
ª   ª   ª       ª   ª   ª   ª       test_str_accessor.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---indexing
ª   ª   ª       ª   ª   ª   ª       test_datetime.py
ª   ª   ª       ª   ª   ª   ª       test_delitem.py
ª   ª   ª       ª   ª   ª   ª       test_get.py
ª   ª   ª       ª   ª   ª   ª       test_getitem.py
ª   ª   ª       ª   ª   ª   ª       test_indexing.py
ª   ª   ª       ª   ª   ª   ª       test_mask.py
ª   ª   ª       ª   ª   ª   ª       test_setitem.py
ª   ª   ª       ª   ª   ª   ª       test_set_value.py
ª   ª   ª       ª   ª   ª   ª       test_take.py
ª   ª   ª       ª   ª   ª   ª       test_where.py
ª   ª   ª       ª   ª   ª   ª       test_xs.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---methods
ª   ª   ª       ª   ª   ª           test_add_prefix_suffix.py
ª   ª   ª       ª   ª   ª           test_align.py
ª   ª   ª       ª   ª   ª           test_argsort.py
ª   ª   ª       ª   ª   ª           test_asof.py
ª   ª   ª       ª   ª   ª           test_astype.py
ª   ª   ª       ª   ª   ª           test_autocorr.py
ª   ª   ª       ª   ª   ª           test_between.py
ª   ª   ª       ª   ª   ª           test_clip.py
ª   ª   ª       ª   ª   ª           test_combine.py
ª   ª   ª       ª   ª   ª           test_combine_first.py
ª   ª   ª       ª   ª   ª           test_compare.py
ª   ª   ª       ª   ª   ª           test_convert_dtypes.py
ª   ª   ª       ª   ª   ª           test_copy.py
ª   ª   ª       ª   ª   ª           test_count.py
ª   ª   ª       ª   ª   ª           test_cov_corr.py
ª   ª   ª       ª   ª   ª           test_describe.py
ª   ª   ª       ª   ª   ª           test_diff.py
ª   ª   ª       ª   ª   ª           test_drop.py
ª   ª   ª       ª   ª   ª           test_dropna.py
ª   ª   ª       ª   ª   ª           test_drop_duplicates.py
ª   ª   ª       ª   ª   ª           test_dtypes.py
ª   ª   ª       ª   ª   ª           test_duplicated.py
ª   ª   ª       ª   ª   ª           test_equals.py
ª   ª   ª       ª   ª   ª           test_explode.py
ª   ª   ª       ª   ª   ª           test_fillna.py
ª   ª   ª       ª   ª   ª           test_get_numeric_data.py
ª   ª   ª       ª   ª   ª           test_head_tail.py
ª   ª   ª       ª   ª   ª           test_infer_objects.py
ª   ª   ª       ª   ª   ª           test_interpolate.py
ª   ª   ª       ª   ª   ª           test_isin.py
ª   ª   ª       ª   ª   ª           test_isna.py
ª   ª   ª       ª   ª   ª           test_is_monotonic.py
ª   ª   ª       ª   ª   ª           test_is_unique.py
ª   ª   ª       ª   ª   ª           test_item.py
ª   ª   ª       ª   ª   ª           test_matmul.py
ª   ª   ª       ª   ª   ª           test_nlargest.py
ª   ª   ª       ª   ª   ª           test_nunique.py
ª   ª   ª       ª   ª   ª           test_pct_change.py
ª   ª   ª       ª   ª   ª           test_pop.py
ª   ª   ª       ª   ª   ª           test_quantile.py
ª   ª   ª       ª   ª   ª           test_rank.py
ª   ª   ª       ª   ª   ª           test_reindex.py
ª   ª   ª       ª   ª   ª           test_reindex_like.py
ª   ª   ª       ª   ª   ª           test_rename.py
ª   ª   ª       ª   ª   ª           test_rename_axis.py
ª   ª   ª       ª   ª   ª           test_repeat.py
ª   ª   ª       ª   ª   ª           test_replace.py
ª   ª   ª       ª   ª   ª           test_reset_index.py
ª   ª   ª       ª   ª   ª           test_round.py
ª   ª   ª       ª   ª   ª           test_searchsorted.py
ª   ª   ª       ª   ª   ª           test_set_name.py
ª   ª   ª       ª   ª   ª           test_sort_index.py
ª   ª   ª       ª   ª   ª           test_sort_values.py
ª   ª   ª       ª   ª   ª           test_tolist.py
ª   ª   ª       ª   ª   ª           test_to_csv.py
ª   ª   ª       ª   ª   ª           test_to_dict.py
ª   ª   ª       ª   ª   ª           test_to_frame.py
ª   ª   ª       ª   ª   ª           test_to_numpy.py
ª   ª   ª       ª   ª   ª           test_truncate.py
ª   ª   ª       ª   ª   ª           test_tz_localize.py
ª   ª   ª       ª   ª   ª           test_unique.py
ª   ª   ª       ª   ª   ª           test_unstack.py
ª   ª   ª       ª   ª   ª           test_update.py
ª   ª   ª       ª   ª   ª           test_values.py
ª   ª   ª       ª   ª   ª           test_value_counts.py
ª   ª   ª       ª   ª   ª           test_view.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---strings
ª   ª   ª       ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª       test_api.py
ª   ª   ª       ª   ª   ª       test_case_justify.py
ª   ª   ª       ª   ª   ª       test_cat.py
ª   ª   ª       ª   ª   ª       test_extract.py
ª   ª   ª       ª   ª   ª       test_find_replace.py
ª   ª   ª       ª   ª   ª       test_get_dummies.py
ª   ª   ª       ª   ª   ª       test_split_partition.py
ª   ª   ª       ª   ª   ª       test_strings.py
ª   ª   ª       ª   ª   ª       test_string_array.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---tools
ª   ª   ª       ª   ª   ª       test_to_datetime.py
ª   ª   ª       ª   ª   ª       test_to_numeric.py
ª   ª   ª       ª   ª   ª       test_to_time.py
ª   ª   ª       ª   ª   ª       test_to_timedelta.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---tseries
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---frequencies
ª   ª   ª       ª   ª   ª   ª       test_frequencies.py
ª   ª   ª       ª   ª   ª   ª       test_freq_code.py
ª   ª   ª       ª   ª   ª   ª       test_inference.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---holiday
ª   ª   ª       ª   ª   ª   ª       test_calendar.py
ª   ª   ª       ª   ª   ª   ª       test_federal.py
ª   ª   ª       ª   ª   ª   ª       test_holiday.py
ª   ª   ª       ª   ª   ª   ª       test_observance.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---offsets
ª   ª   ª       ª   ª   ª           common.py
ª   ª   ª       ª   ª   ª           conftest.py
ª   ª   ª       ª   ª   ª           test_business_day.py
ª   ª   ª       ª   ª   ª           test_business_hour.py
ª   ª   ª       ª   ª   ª           test_business_month.py
ª   ª   ª       ª   ª   ª           test_business_quarter.py
ª   ª   ª       ª   ª   ª           test_business_year.py
ª   ª   ª       ª   ª   ª           test_common.py
ª   ª   ª       ª   ª   ª           test_custom_business_day.py
ª   ª   ª       ª   ª   ª           test_custom_business_hour.py
ª   ª   ª       ª   ª   ª           test_custom_business_month.py
ª   ª   ª       ª   ª   ª           test_dst.py
ª   ª   ª       ª   ª   ª           test_easter.py
ª   ª   ª       ª   ª   ª           test_fiscal.py
ª   ª   ª       ª   ª   ª           test_index.py
ª   ª   ª       ª   ª   ª           test_month.py
ª   ª   ª       ª   ª   ª           test_offsets.py
ª   ª   ª       ª   ª   ª           test_offsets_properties.py
ª   ª   ª       ª   ª   ª           test_quarter.py
ª   ª   ª       ª   ª   ª           test_ticks.py
ª   ª   ª       ª   ª   ª           test_week.py
ª   ª   ª       ª   ª   ª           test_year.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---tslibs
ª   ª   ª       ª   ª   ª       test_api.py
ª   ª   ª       ª   ª   ª       test_array_to_datetime.py
ª   ª   ª       ª   ª   ª       test_ccalendar.py
ª   ª   ª       ª   ª   ª       test_conversion.py
ª   ª   ª       ª   ª   ª       test_fields.py
ª   ª   ª       ª   ª   ª       test_libfrequencies.py
ª   ª   ª       ª   ª   ª       test_liboffsets.py
ª   ª   ª       ª   ª   ª       test_np_datetime.py
ª   ª   ª       ª   ª   ª       test_parse_iso8601.py
ª   ª   ª       ª   ª   ª       test_parsing.py
ª   ª   ª       ª   ª   ª       test_period_asfreq.py
ª   ª   ª       ª   ª   ª       test_resolution.py
ª   ª   ª       ª   ª   ª       test_timedeltas.py
ª   ª   ª       ª   ª   ª       test_timezones.py
ª   ª   ª       ª   ª   ª       test_to_offset.py
ª   ª   ª       ª   ª   ª       test_tzconversion.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---util
ª   ª   ª       ª   ª   ª       conftest.py
ª   ª   ª       ª   ª   ª       test_assert_almost_equal.py
ª   ª   ª       ª   ª   ª       test_assert_attr_equal.py
ª   ª   ª       ª   ª   ª       test_assert_categorical_equal.py
ª   ª   ª       ª   ª   ª       test_assert_extension_array_equal.py
ª   ª   ª       ª   ª   ª       test_assert_frame_equal.py
ª   ª   ª       ª   ª   ª       test_assert_index_equal.py
ª   ª   ª       ª   ª   ª       test_assert_interval_array_equal.py
ª   ª   ª       ª   ª   ª       test_assert_numpy_array_equal.py
ª   ª   ª       ª   ª   ª       test_assert_produces_warning.py
ª   ª   ª       ª   ª   ª       test_assert_series_equal.py
ª   ª   ª       ª   ª   ª       test_deprecate.py
ª   ª   ª       ª   ª   ª       test_deprecate_kwarg.py
ª   ª   ª       ª   ª   ª       test_deprecate_nonkeyword_arguments.py
ª   ª   ª       ª   ª   ª       test_doc.py
ª   ª   ª       ª   ª   ª       test_hashing.py
ª   ª   ª       ª   ª   ª       test_make_objects.py
ª   ª   ª       ª   ª   ª       test_numba.py
ª   ª   ª       ª   ª   ª       test_rewrite_warning.py
ª   ª   ª       ª   ª   ª       test_safe_import.py
ª   ª   ª       ª   ª   ª       test_shares_memory.py
ª   ª   ª       ª   ª   ª       test_show_versions.py
ª   ª   ª       ª   ª   ª       test_str_methods.py
ª   ª   ª       ª   ª   ª       test_util.py
ª   ª   ª       ª   ª   ª       test_validate_args.py
ª   ª   ª       ª   ª   ª       test_validate_args_and_kwargs.py
ª   ª   ª       ª   ª   ª       test_validate_inclusive.py
ª   ª   ª       ª   ª   ª       test_validate_kwargs.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---window
ª   ª   ª       ª   ª       ª   conftest.py
ª   ª   ª       ª   ª       ª   test_api.py
ª   ª   ª       ª   ª       ª   test_apply.py
ª   ª   ª       ª   ª       ª   test_base_indexer.py
ª   ª   ª       ª   ª       ª   test_cython_aggregations.py
ª   ª   ª       ª   ª       ª   test_dtypes.py
ª   ª   ª       ª   ª       ª   test_ewm.py
ª   ª   ª       ª   ª       ª   test_expanding.py
ª   ª   ª       ª   ª       ª   test_groupby.py
ª   ª   ª       ª   ª       ª   test_numba.py
ª   ª   ª       ª   ª       ª   test_online.py
ª   ª   ª       ª   ª       ª   test_pairwise.py
ª   ª   ª       ª   ª       ª   test_rolling.py
ª   ª   ª       ª   ª       ª   test_rolling_functions.py
ª   ª   ª       ª   ª       ª   test_rolling_quantile.py
ª   ª   ª       ª   ª       ª   test_rolling_skew_kurt.py
ª   ª   ª       ª   ª       ª   test_timeseries_window.py
ª   ª   ª       ª   ª       ª   test_win_type.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---moments
ª   ª   ª       ª   ª               conftest.py
ª   ª   ª       ª   ª               test_moments_consistency_ewm.py
ª   ª   ª       ª   ª               test_moments_consistency_expanding.py
ª   ª   ª       ª   ª               test_moments_consistency_rolling.py
ª   ª   ª       ª   ª               __init__.py
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---tseries
ª   ª   ª       ª   ª       api.py
ª   ª   ª       ª   ª       frequencies.py
ª   ª   ª       ª   ª       holiday.py
ª   ª   ª       ª   ª       offsets.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---util
ª   ª   ª       ª   ª   ª   _decorators.py
ª   ª   ª       ª   ª   ª   _doctools.py
ª   ª   ª       ª   ª   ª   _exceptions.py
ª   ª   ª       ª   ª   ª   _print_versions.py
ª   ª   ª       ª   ª   ª   _str_methods.py
ª   ª   ª       ª   ª   ª   _tester.py
ª   ª   ª       ª   ª   ª   _test_decorators.py
ª   ª   ª       ª   ª   ª   _validators.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---version
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---_config
ª   ª   ª       ª   ª       config.py
ª   ª   ª       ª   ª       dates.py
ª   ª   ª       ª   ª       display.py
ª   ª   ª       ª   ª       localization.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---_libs
ª   ª   ª       ª   ª   ª   algos.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   algos.pxd
ª   ª   ª       ª   ª   ª   algos.pyi
ª   ª   ª       ª   ª   ª   algos.pyx
ª   ª   ª       ª   ª   ª   algos_common_helper.pxi.in
ª   ª   ª       ª   ª   ª   algos_take_helper.pxi.in
ª   ª   ª       ª   ª   ª   arrays.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   arrays.pxd
ª   ª   ª       ª   ª   ª   arrays.pyi
ª   ª   ª       ª   ª   ª   arrays.pyx
ª   ª   ª       ª   ª   ª   dtypes.pxd
ª   ª   ª       ª   ª   ª   groupby.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   groupby.pyi
ª   ª   ª       ª   ª   ª   groupby.pyx
ª   ª   ª       ª   ª   ª   hashing.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   hashing.pyi
ª   ª   ª       ª   ª   ª   hashing.pyx
ª   ª   ª       ª   ª   ª   hashtable.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   hashtable.pxd
ª   ª   ª       ª   ª   ª   hashtable.pyi
ª   ª   ª       ª   ª   ª   hashtable.pyx
ª   ª   ª       ª   ª   ª   hashtable_class_helper.pxi.in
ª   ª   ª       ª   ª   ª   hashtable_func_helper.pxi.in
ª   ª   ª       ª   ª   ª   index.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   index.pyi
ª   ª   ª       ª   ª   ª   index.pyx
ª   ª   ª       ª   ª   ª   indexing.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   indexing.pyi
ª   ª   ª       ª   ª   ª   indexing.pyx
ª   ª   ª       ª   ª   ª   index_class_helper.pxi.in
ª   ª   ª       ª   ª   ª   internals.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   internals.pyi
ª   ª   ª       ª   ª   ª   internals.pyx
ª   ª   ª       ª   ª   ª   interval.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   interval.pyi
ª   ª   ª       ª   ª   ª   interval.pyx
ª   ª   ª       ª   ª   ª   intervaltree.pxi.in
ª   ª   ª       ª   ª   ª   join.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   join.pyi
ª   ª   ª       ª   ª   ª   join.pyx
ª   ª   ª       ª   ª   ª   json.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   json.pyi
ª   ª   ª       ª   ª   ª   khash.pxd
ª   ª   ª       ª   ª   ª   khash_for_primitive_helper.pxi.in
ª   ª   ª       ª   ª   ª   lib.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   lib.pxd
ª   ª   ª       ª   ª   ª   lib.pyi
ª   ª   ª       ª   ª   ª   lib.pyx
ª   ª   ª       ª   ª   ª   missing.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   missing.pxd
ª   ª   ª       ª   ª   ª   missing.pyi
ª   ª   ª       ª   ª   ª   missing.pyx
ª   ª   ª       ª   ª   ª   ops.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ops.pyi
ª   ª   ª       ª   ª   ª   ops.pyx
ª   ª   ª       ª   ª   ª   ops_dispatch.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ops_dispatch.pyi
ª   ª   ª       ª   ª   ª   ops_dispatch.pyx
ª   ª   ª       ª   ª   ª   parsers.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   parsers.pyi
ª   ª   ª       ª   ª   ª   parsers.pyx
ª   ª   ª       ª   ª   ª   properties.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   properties.pyi
ª   ª   ª       ª   ª   ª   properties.pyx
ª   ª   ª       ª   ª   ª   reduction.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   reduction.pyi
ª   ª   ª       ª   ª   ª   reduction.pyx
ª   ª   ª       ª   ª   ª   reshape.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   reshape.pyi
ª   ª   ª       ª   ª   ª   reshape.pyx
ª   ª   ª       ª   ª   ª   sparse.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   sparse.pyi
ª   ª   ª       ª   ª   ª   sparse.pyx
ª   ª   ª       ª   ª   ª   sparse_op_helper.pxi.in
ª   ª   ª       ª   ª   ª   testing.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   testing.pyi
ª   ª   ª       ª   ª   ª   testing.pyx
ª   ª   ª       ª   ª   ª   tslib.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   tslib.pyi
ª   ª   ª       ª   ª   ª   tslib.pyx
ª   ª   ª       ª   ª   ª   util.pxd
ª   ª   ª       ª   ª   ª   writers.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   writers.pyi
ª   ª   ª       ª   ª   ª   writers.pyx
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tslibs
ª   ª   ª       ª   ª   ª       base.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       base.pxd
ª   ª   ª       ª   ª   ª       base.pyx
ª   ª   ª       ª   ª   ª       ccalendar.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       ccalendar.pxd
ª   ª   ª       ª   ª   ª       ccalendar.pyi
ª   ª   ª       ª   ª   ª       ccalendar.pyx
ª   ª   ª       ª   ª   ª       conversion.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       conversion.pxd
ª   ª   ª       ª   ª   ª       conversion.pyi
ª   ª   ª       ª   ª   ª       conversion.pyx
ª   ª   ª       ª   ª   ª       dtypes.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       dtypes.pxd
ª   ª   ª       ª   ª   ª       dtypes.pyi
ª   ª   ª       ª   ª   ª       dtypes.pyx
ª   ª   ª       ª   ª   ª       fields.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       fields.pyi
ª   ª   ª       ª   ª   ª       fields.pyx
ª   ª   ª       ª   ª   ª       nattype.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       nattype.pxd
ª   ª   ª       ª   ª   ª       nattype.pyi
ª   ª   ª       ª   ª   ª       nattype.pyx
ª   ª   ª       ª   ª   ª       np_datetime.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       np_datetime.pxd
ª   ª   ª       ª   ª   ª       np_datetime.pyi
ª   ª   ª       ª   ª   ª       np_datetime.pyx
ª   ª   ª       ª   ª   ª       offsets.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       offsets.pxd
ª   ª   ª       ª   ª   ª       offsets.pyi
ª   ª   ª       ª   ª   ª       offsets.pyx
ª   ª   ª       ª   ª   ª       parsing.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       parsing.pxd
ª   ª   ª       ª   ª   ª       parsing.pyi
ª   ª   ª       ª   ª   ª       parsing.pyx
ª   ª   ª       ª   ª   ª       period.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       period.pxd
ª   ª   ª       ª   ª   ª       period.pyi
ª   ª   ª       ª   ª   ª       period.pyx
ª   ª   ª       ª   ª   ª       strptime.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       strptime.pxd
ª   ª   ª       ª   ª   ª       strptime.pyi
ª   ª   ª       ª   ª   ª       strptime.pyx
ª   ª   ª       ª   ª   ª       timedeltas.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       timedeltas.pxd
ª   ª   ª       ª   ª   ª       timedeltas.pyi
ª   ª   ª       ª   ª   ª       timedeltas.pyx
ª   ª   ª       ª   ª   ª       timestamps.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       timestamps.pxd
ª   ª   ª       ª   ª   ª       timestamps.pyi
ª   ª   ª       ª   ª   ª       timestamps.pyx
ª   ª   ª       ª   ª   ª       timezones.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       timezones.pxd
ª   ª   ª       ª   ª   ª       timezones.pyi
ª   ª   ª       ª   ª   ª       timezones.pyx
ª   ª   ª       ª   ª   ª       tzconversion.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       tzconversion.pxd
ª   ª   ª       ª   ª   ª       tzconversion.pyi
ª   ª   ª       ª   ª   ª       tzconversion.pyx
ª   ª   ª       ª   ª   ª       util.pxd
ª   ª   ª       ª   ª   ª       vectorized.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       vectorized.pyi
ª   ª   ª       ª   ª   ª       vectorized.pyx
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---window
ª   ª   ª       ª   ª           aggregations.cp39-win32.pyd
ª   ª   ª       ª   ª           aggregations.pyi
ª   ª   ª       ª   ª           aggregations.pyx
ª   ª   ª       ª   ª           concrt140.dll
ª   ª   ª       ª   ª           indexers.cp39-win32.pyd
ª   ª   ª       ª   ª           indexers.pyi
ª   ª   ª       ª   ª           indexers.pyx
ª   ª   ª       ª   ª           msvcp140.dll
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---_testing
ª   ª   ª       ª           asserters.py
ª   ª   ª       ª           compat.py
ª   ª   ª       ª           contexts.py
ª   ª   ª       ª           _hypothesis.py
ª   ª   ª       ª           _io.py
ª   ª   ª       ª           _random.py
ª   ª   ª       ª           _warnings.py
ª   ª   ª       ª           __init__.py
ª   ª   ª       ª           
ª   ª   ª       +---pandas-2.0.3.dist-info
ª   ª   ª       ª       AUTHORS.md
ª   ª   ª       ª       entry_points.txt
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       REQUESTED
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---pip
ª   ª   ª       ª   ª   py.typed
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   __main__.py
ª   ª   ª       ª   ª   __pip-runner__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---_internal
ª   ª   ª       ª   ª   ª   build_env.py
ª   ª   ª       ª   ª   ª   cache.py
ª   ª   ª       ª   ª   ª   configuration.py
ª   ª   ª       ª   ª   ª   exceptions.py
ª   ª   ª       ª   ª   ª   main.py
ª   ª   ª       ª   ª   ª   pyproject.py
ª   ª   ª       ª   ª   ª   self_outdated_check.py
ª   ª   ª       ª   ª   ª   wheel_builder.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---cli
ª   ª   ª       ª   ª   ª       autocompletion.py
ª   ª   ª       ª   ª   ª       base_command.py
ª   ª   ª       ª   ª   ª       cmdoptions.py
ª   ª   ª       ª   ª   ª       command_context.py
ª   ª   ª       ª   ª   ª       index_command.py
ª   ª   ª       ª   ª   ª       main.py
ª   ª   ª       ª   ª   ª       main_parser.py
ª   ª   ª       ª   ª   ª       parser.py
ª   ª   ª       ª   ª   ª       progress_bars.py
ª   ª   ª       ª   ª   ª       req_command.py
ª   ª   ª       ª   ª   ª       spinners.py
ª   ª   ª       ª   ª   ª       status_codes.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---commands
ª   ª   ª       ª   ª   ª       cache.py
ª   ª   ª       ª   ª   ª       check.py
ª   ª   ª       ª   ª   ª       completion.py
ª   ª   ª       ª   ª   ª       configuration.py
ª   ª   ª       ª   ª   ª       debug.py
ª   ª   ª       ª   ª   ª       download.py
ª   ª   ª       ª   ª   ª       freeze.py
ª   ª   ª       ª   ª   ª       hash.py
ª   ª   ª       ª   ª   ª       help.py
ª   ª   ª       ª   ª   ª       index.py
ª   ª   ª       ª   ª   ª       inspect.py
ª   ª   ª       ª   ª   ª       install.py
ª   ª   ª       ª   ª   ª       list.py
ª   ª   ª       ª   ª   ª       search.py
ª   ª   ª       ª   ª   ª       show.py
ª   ª   ª       ª   ª   ª       uninstall.py
ª   ª   ª       ª   ª   ª       wheel.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---distributions
ª   ª   ª       ª   ª   ª       base.py
ª   ª   ª       ª   ª   ª       installed.py
ª   ª   ª       ª   ª   ª       sdist.py
ª   ª   ª       ª   ª   ª       wheel.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---index
ª   ª   ª       ª   ª   ª       collector.py
ª   ª   ª       ª   ª   ª       package_finder.py
ª   ª   ª       ª   ª   ª       sources.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---locations
ª   ª   ª       ª   ª   ª       base.py
ª   ª   ª       ª   ª   ª       _distutils.py
ª   ª   ª       ª   ª   ª       _sysconfig.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---metadata
ª   ª   ª       ª   ª   ª   ª   base.py
ª   ª   ª       ª   ª   ª   ª   pkg_resources.py
ª   ª   ª       ª   ª   ª   ª   _json.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---importlib
ª   ª   ª       ª   ª   ª           _compat.py
ª   ª   ª       ª   ª   ª           _dists.py
ª   ª   ª       ª   ª   ª           _envs.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---models
ª   ª   ª       ª   ª   ª       candidate.py
ª   ª   ª       ª   ª   ª       direct_url.py
ª   ª   ª       ª   ª   ª       format_control.py
ª   ª   ª       ª   ª   ª       index.py
ª   ª   ª       ª   ª   ª       installation_report.py
ª   ª   ª       ª   ª   ª       link.py
ª   ª   ª       ª   ª   ª       scheme.py
ª   ª   ª       ª   ª   ª       search_scope.py
ª   ª   ª       ª   ª   ª       selection_prefs.py
ª   ª   ª       ª   ª   ª       target_python.py
ª   ª   ª       ª   ª   ª       wheel.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---network
ª   ª   ª       ª   ª   ª       auth.py
ª   ª   ª       ª   ª   ª       cache.py
ª   ª   ª       ª   ª   ª       download.py
ª   ª   ª       ª   ª   ª       lazy_wheel.py
ª   ª   ª       ª   ª   ª       session.py
ª   ª   ª       ª   ª   ª       utils.py
ª   ª   ª       ª   ª   ª       xmlrpc.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---operations
ª   ª   ª       ª   ª   ª   ª   check.py
ª   ª   ª       ª   ª   ª   ª   freeze.py
ª   ª   ª       ª   ª   ª   ª   prepare.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---build
ª   ª   ª       ª   ª   ª   ª       build_tracker.py
ª   ª   ª       ª   ª   ª   ª       metadata.py
ª   ª   ª       ª   ª   ª   ª       metadata_editable.py
ª   ª   ª       ª   ª   ª   ª       metadata_legacy.py
ª   ª   ª       ª   ª   ª   ª       wheel.py
ª   ª   ª       ª   ª   ª   ª       wheel_editable.py
ª   ª   ª       ª   ª   ª   ª       wheel_legacy.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---install
ª   ª   ª       ª   ª   ª           editable_legacy.py
ª   ª   ª       ª   ª   ª           wheel.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---req
ª   ª   ª       ª   ª   ª       constructors.py
ª   ª   ª       ª   ª   ª       req_file.py
ª   ª   ª       ª   ª   ª       req_install.py
ª   ª   ª       ª   ª   ª       req_set.py
ª   ª   ª       ª   ª   ª       req_uninstall.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---resolution
ª   ª   ª       ª   ª   ª   ª   base.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---legacy
ª   ª   ª       ª   ª   ª   ª       resolver.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---resolvelib
ª   ª   ª       ª   ª   ª           base.py
ª   ª   ª       ª   ª   ª           candidates.py
ª   ª   ª       ª   ª   ª           factory.py
ª   ª   ª       ª   ª   ª           found_candidates.py
ª   ª   ª       ª   ª   ª           provider.py
ª   ª   ª       ª   ª   ª           reporter.py
ª   ª   ª       ª   ª   ª           requirements.py
ª   ª   ª       ª   ª   ª           resolver.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---utils
ª   ª   ª       ª   ª   ª       appdirs.py
ª   ª   ª       ª   ª   ª       compat.py
ª   ª   ª       ª   ª   ª       compatibility_tags.py
ª   ª   ª       ª   ª   ª       datetime.py
ª   ª   ª       ª   ª   ª       deprecation.py
ª   ª   ª       ª   ª   ª       direct_url_helpers.py
ª   ª   ª       ª   ª   ª       egg_link.py
ª   ª   ª       ª   ª   ª       encoding.py
ª   ª   ª       ª   ª   ª       entrypoints.py
ª   ª   ª       ª   ª   ª       filesystem.py
ª   ª   ª       ª   ª   ª       filetypes.py
ª   ª   ª       ª   ª   ª       glibc.py
ª   ª   ª       ª   ª   ª       hashes.py
ª   ª   ª       ª   ª   ª       logging.py
ª   ª   ª       ª   ª   ª       misc.py
ª   ª   ª       ª   ª   ª       packaging.py
ª   ª   ª       ª   ª   ª       retry.py
ª   ª   ª       ª   ª   ª       setuptools_build.py
ª   ª   ª       ª   ª   ª       subprocess.py
ª   ª   ª       ª   ª   ª       temp_dir.py
ª   ª   ª       ª   ª   ª       unpacking.py
ª   ª   ª       ª   ª   ª       urls.py
ª   ª   ª       ª   ª   ª       virtualenv.py
ª   ª   ª       ª   ª   ª       wheel.py
ª   ª   ª       ª   ª   ª       _jaraco_text.py
ª   ª   ª       ª   ª   ª       _log.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---vcs
ª   ª   ª       ª   ª           bazaar.py
ª   ª   ª       ª   ª           git.py
ª   ª   ª       ª   ª           mercurial.py
ª   ª   ª       ª   ª           subversion.py
ª   ª   ª       ª   ª           versioncontrol.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---_vendor
ª   ª   ª       ª       ª   typing_extensions.py
ª   ª   ª       ª       ª   vendor.txt
ª   ª   ª       ª       ª   __init__.py
ª   ª   ª       ª       ª   
ª   ª   ª       ª       +---cachecontrol
ª   ª   ª       ª       ª   ª   adapter.py
ª   ª   ª       ª       ª   ª   cache.py
ª   ª   ª       ª       ª   ª   controller.py
ª   ª   ª       ª       ª   ª   filewrapper.py
ª   ª   ª       ª       ª   ª   heuristics.py
ª   ª   ª       ª       ª   ª   py.typed
ª   ª   ª       ª       ª   ª   serialize.py
ª   ª   ª       ª       ª   ª   wrapper.py
ª   ª   ª       ª       ª   ª   _cmd.py
ª   ª   ª       ª       ª   ª   __init__.py
ª   ª   ª       ª       ª   ª   
ª   ª   ª       ª       ª   +---caches
ª   ª   ª       ª       ª           file_cache.py
ª   ª   ª       ª       ª           redis_cache.py
ª   ª   ª       ª       ª           __init__.py
ª   ª   ª       ª       ª           
ª   ª   ª       ª       +---certifi
ª   ª   ª       ª       ª       cacert.pem
ª   ª   ª       ª       ª       core.py
ª   ª   ª       ª       ª       py.typed
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       __main__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---distlib
ª   ª   ª       ª       ª       compat.py
ª   ª   ª       ª       ª       database.py
ª   ª   ª       ª       ª       index.py
ª   ª   ª       ª       ª       locators.py
ª   ª   ª       ª       ª       manifest.py
ª   ª   ª       ª       ª       markers.py
ª   ª   ª       ª       ª       metadata.py
ª   ª   ª       ª       ª       resources.py
ª   ª   ª       ª       ª       scripts.py
ª   ª   ª       ª       ª       t32.exe
ª   ª   ª       ª       ª       t64-arm.exe
ª   ª   ª       ª       ª       t64.exe
ª   ª   ª       ª       ª       util.py
ª   ª   ª       ª       ª       version.py
ª   ª   ª       ª       ª       w32.exe
ª   ª   ª       ª       ª       w64-arm.exe
ª   ª   ª       ª       ª       w64.exe
ª   ª   ª       ª       ª       wheel.py
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---distro
ª   ª   ª       ª       ª       distro.py
ª   ª   ª       ª       ª       py.typed
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       __main__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---idna
ª   ª   ª       ª       ª       codec.py
ª   ª   ª       ª       ª       compat.py
ª   ª   ª       ª       ª       core.py
ª   ª   ª       ª       ª       idnadata.py
ª   ª   ª       ª       ª       intranges.py
ª   ª   ª       ª       ª       package_data.py
ª   ª   ª       ª       ª       py.typed
ª   ª   ª       ª       ª       uts46data.py
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---msgpack
ª   ª   ª       ª       ª       exceptions.py
ª   ª   ª       ª       ª       ext.py
ª   ª   ª       ª       ª       fallback.py
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---packaging
ª   ª   ª       ª       ª       markers.py
ª   ª   ª       ª       ª       metadata.py
ª   ª   ª       ª       ª       py.typed
ª   ª   ª       ª       ª       requirements.py
ª   ª   ª       ª       ª       specifiers.py
ª   ª   ª       ª       ª       tags.py
ª   ª   ª       ª       ª       utils.py
ª   ª   ª       ª       ª       version.py
ª   ª   ª       ª       ª       _elffile.py
ª   ª   ª       ª       ª       _manylinux.py
ª   ª   ª       ª       ª       _musllinux.py
ª   ª   ª       ª       ª       _parser.py
ª   ª   ª       ª       ª       _structures.py
ª   ª   ª       ª       ª       _tokenizer.py
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---pkg_resources
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---platformdirs
ª   ª   ª       ª       ª       android.py
ª   ª   ª       ª       ª       api.py
ª   ª   ª       ª       ª       macos.py
ª   ª   ª       ª       ª       py.typed
ª   ª   ª       ª       ª       unix.py
ª   ª   ª       ª       ª       version.py
ª   ª   ª       ª       ª       windows.py
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       __main__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---pygments
ª   ª   ª       ª       ª   ª   cmdline.py
ª   ª   ª       ª       ª   ª   console.py
ª   ª   ª       ª       ª   ª   filter.py
ª   ª   ª       ª       ª   ª   formatter.py
ª   ª   ª       ª       ª   ª   lexer.py
ª   ª   ª       ª       ª   ª   modeline.py
ª   ª   ª       ª       ª   ª   plugin.py
ª   ª   ª       ª       ª   ª   regexopt.py
ª   ª   ª       ª       ª   ª   scanner.py
ª   ª   ª       ª       ª   ª   sphinxext.py
ª   ª   ª       ª       ª   ª   style.py
ª   ª   ª       ª       ª   ª   token.py
ª   ª   ª       ª       ª   ª   unistring.py
ª   ª   ª       ª       ª   ª   util.py
ª   ª   ª       ª       ª   ª   __init__.py
ª   ª   ª       ª       ª   ª   __main__.py
ª   ª   ª       ª       ª   ª   
ª   ª   ª       ª       ª   +---filters
ª   ª   ª       ª       ª   ª       __init__.py
ª   ª   ª       ª       ª   ª       
ª   ª   ª       ª       ª   +---formatters
ª   ª   ª       ª       ª   ª       bbcode.py
ª   ª   ª       ª       ª   ª       groff.py
ª   ª   ª       ª       ª   ª       html.py
ª   ª   ª       ª       ª   ª       img.py
ª   ª   ª       ª       ª   ª       irc.py
ª   ª   ª       ª       ª   ª       latex.py
ª   ª   ª       ª       ª   ª       other.py
ª   ª   ª       ª       ª   ª       pangomarkup.py
ª   ª   ª       ª       ª   ª       rtf.py
ª   ª   ª       ª       ª   ª       svg.py
ª   ª   ª       ª       ª   ª       terminal.py
ª   ª   ª       ª       ª   ª       terminal256.py
ª   ª   ª       ª       ª   ª       _mapping.py
ª   ª   ª       ª       ª   ª       __init__.py
ª   ª   ª       ª       ª   ª       
ª   ª   ª       ª       ª   +---lexers
ª   ª   ª       ª       ª   ª       python.py
ª   ª   ª       ª       ª   ª       _mapping.py
ª   ª   ª       ª       ª   ª       __init__.py
ª   ª   ª       ª       ª   ª       
ª   ª   ª       ª       ª   +---styles
ª   ª   ª       ª       ª           _mapping.py
ª   ª   ª       ª       ª           __init__.py
ª   ª   ª       ª       ª           
ª   ª   ª       ª       +---pyproject_hooks
ª   ª   ª       ª       ª   ª   _compat.py
ª   ª   ª       ª       ª   ª   _impl.py
ª   ª   ª       ª       ª   ª   __init__.py
ª   ª   ª       ª       ª   ª   
ª   ª   ª       ª       ª   +---_in_process
ª   ª   ª       ª       ª           _in_process.py
ª   ª   ª       ª       ª           __init__.py
ª   ª   ª       ª       ª           
ª   ª   ª       ª       +---requests
ª   ª   ª       ª       ª       adapters.py
ª   ª   ª       ª       ª       api.py
ª   ª   ª       ª       ª       auth.py
ª   ª   ª       ª       ª       certs.py
ª   ª   ª       ª       ª       compat.py
ª   ª   ª       ª       ª       cookies.py
ª   ª   ª       ª       ª       exceptions.py
ª   ª   ª       ª       ª       help.py
ª   ª   ª       ª       ª       hooks.py
ª   ª   ª       ª       ª       models.py
ª   ª   ª       ª       ª       packages.py
ª   ª   ª       ª       ª       sessions.py
ª   ª   ª       ª       ª       status_codes.py
ª   ª   ª       ª       ª       structures.py
ª   ª   ª       ª       ª       utils.py
ª   ª   ª       ª       ª       _internal_utils.py
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       __version__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---resolvelib
ª   ª   ª       ª       ª   ª   providers.py
ª   ª   ª       ª       ª   ª   py.typed
ª   ª   ª       ª       ª   ª   reporters.py
ª   ª   ª       ª       ª   ª   resolvers.py
ª   ª   ª       ª       ª   ª   structs.py
ª   ª   ª       ª       ª   ª   __init__.py
ª   ª   ª       ª       ª   ª   
ª   ª   ª       ª       ª   +---compat
ª   ª   ª       ª       ª           collections_abc.py
ª   ª   ª       ª       ª           __init__.py
ª   ª   ª       ª       ª           
ª   ª   ª       ª       +---rich
ª   ª   ª       ª       ª       abc.py
ª   ª   ª       ª       ª       align.py
ª   ª   ª       ª       ª       ansi.py
ª   ª   ª       ª       ª       bar.py
ª   ª   ª       ª       ª       box.py
ª   ª   ª       ª       ª       cells.py
ª   ª   ª       ª       ª       color.py
ª   ª   ª       ª       ª       color_triplet.py
ª   ª   ª       ª       ª       columns.py
ª   ª   ª       ª       ª       console.py
ª   ª   ª       ª       ª       constrain.py
ª   ª   ª       ª       ª       containers.py
ª   ª   ª       ª       ª       control.py
ª   ª   ª       ª       ª       default_styles.py
ª   ª   ª       ª       ª       diagnose.py
ª   ª   ª       ª       ª       emoji.py
ª   ª   ª       ª       ª       errors.py
ª   ª   ª       ª       ª       filesize.py
ª   ª   ª       ª       ª       file_proxy.py
ª   ª   ª       ª       ª       highlighter.py
ª   ª   ª       ª       ª       json.py
ª   ª   ª       ª       ª       jupyter.py
ª   ª   ª       ª       ª       layout.py
ª   ª   ª       ª       ª       live.py
ª   ª   ª       ª       ª       live_render.py
ª   ª   ª       ª       ª       logging.py
ª   ª   ª       ª       ª       markup.py
ª   ª   ª       ª       ª       measure.py
ª   ª   ª       ª       ª       padding.py
ª   ª   ª       ª       ª       pager.py
ª   ª   ª       ª       ª       palette.py
ª   ª   ª       ª       ª       panel.py
ª   ª   ª       ª       ª       pretty.py
ª   ª   ª       ª       ª       progress.py
ª   ª   ª       ª       ª       progress_bar.py
ª   ª   ª       ª       ª       prompt.py
ª   ª   ª       ª       ª       protocol.py
ª   ª   ª       ª       ª       py.typed
ª   ª   ª       ª       ª       region.py
ª   ª   ª       ª       ª       repr.py
ª   ª   ª       ª       ª       rule.py
ª   ª   ª       ª       ª       scope.py
ª   ª   ª       ª       ª       screen.py
ª   ª   ª       ª       ª       segment.py
ª   ª   ª       ª       ª       spinner.py
ª   ª   ª       ª       ª       status.py
ª   ª   ª       ª       ª       style.py
ª   ª   ª       ª       ª       styled.py
ª   ª   ª       ª       ª       syntax.py
ª   ª   ª       ª       ª       table.py
ª   ª   ª       ª       ª       terminal_theme.py
ª   ª   ª       ª       ª       text.py
ª   ª   ª       ª       ª       theme.py
ª   ª   ª       ª       ª       themes.py
ª   ª   ª       ª       ª       traceback.py
ª   ª   ª       ª       ª       tree.py
ª   ª   ª       ª       ª       _cell_widths.py
ª   ª   ª       ª       ª       _emoji_codes.py
ª   ª   ª       ª       ª       _emoji_replace.py
ª   ª   ª       ª       ª       _export_format.py
ª   ª   ª       ª       ª       _extension.py
ª   ª   ª       ª       ª       _fileno.py
ª   ª   ª       ª       ª       _inspect.py
ª   ª   ª       ª       ª       _log_render.py
ª   ª   ª       ª       ª       _loop.py
ª   ª   ª       ª       ª       _null_file.py
ª   ª   ª       ª       ª       _palettes.py
ª   ª   ª       ª       ª       _pick.py
ª   ª   ª       ª       ª       _ratio.py
ª   ª   ª       ª       ª       _spinners.py
ª   ª   ª       ª       ª       _stack.py
ª   ª   ª       ª       ª       _timer.py
ª   ª   ª       ª       ª       _win32_console.py
ª   ª   ª       ª       ª       _windows.py
ª   ª   ª       ª       ª       _windows_renderer.py
ª   ª   ª       ª       ª       _wrap.py
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       __main__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---tomli
ª   ª   ª       ª       ª       py.typed
ª   ª   ª       ª       ª       _parser.py
ª   ª   ª       ª       ª       _re.py
ª   ª   ª       ª       ª       _types.py
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---truststore
ª   ª   ª       ª       ª       py.typed
ª   ª   ª       ª       ª       _api.py
ª   ª   ª       ª       ª       _macos.py
ª   ª   ª       ª       ª       _openssl.py
ª   ª   ª       ª       ª       _ssl_constants.py
ª   ª   ª       ª       ª       _windows.py
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---urllib3
ª   ª   ª       ª           ª   connection.py
ª   ª   ª       ª           ª   connectionpool.py
ª   ª   ª       ª           ª   exceptions.py
ª   ª   ª       ª           ª   fields.py
ª   ª   ª       ª           ª   filepost.py
ª   ª   ª       ª           ª   poolmanager.py
ª   ª   ª       ª           ª   request.py
ª   ª   ª       ª           ª   response.py
ª   ª   ª       ª           ª   _collections.py
ª   ª   ª       ª           ª   _version.py
ª   ª   ª       ª           ª   __init__.py
ª   ª   ª       ª           ª   
ª   ª   ª       ª           +---contrib
ª   ª   ª       ª           ª   ª   appengine.py
ª   ª   ª       ª           ª   ª   ntlmpool.py
ª   ª   ª       ª           ª   ª   pyopenssl.py
ª   ª   ª       ª           ª   ª   securetransport.py
ª   ª   ª       ª           ª   ª   socks.py
ª   ª   ª       ª           ª   ª   _appengine_environ.py
ª   ª   ª       ª           ª   ª   __init__.py
ª   ª   ª       ª           ª   ª   
ª   ª   ª       ª           ª   +---_securetransport
ª   ª   ª       ª           ª           bindings.py
ª   ª   ª       ª           ª           low_level.py
ª   ª   ª       ª           ª           __init__.py
ª   ª   ª       ª           ª           
ª   ª   ª       ª           +---packages
ª   ª   ª       ª           ª   ª   six.py
ª   ª   ª       ª           ª   ª   __init__.py
ª   ª   ª       ª           ª   ª   
ª   ª   ª       ª           ª   +---backports
ª   ª   ª       ª           ª           makefile.py
ª   ª   ª       ª           ª           weakref_finalize.py
ª   ª   ª       ª           ª           __init__.py
ª   ª   ª       ª           ª           
ª   ª   ª       ª           +---util
ª   ª   ª       ª                   connection.py
ª   ª   ª       ª                   proxy.py
ª   ª   ª       ª                   queue.py
ª   ª   ª       ª                   request.py
ª   ª   ª       ª                   response.py
ª   ª   ª       ª                   retry.py
ª   ª   ª       ª                   ssltransport.py
ª   ª   ª       ª                   ssl_.py
ª   ª   ª       ª                   ssl_match_hostname.py
ª   ª   ª       ª                   timeout.py
ª   ª   ª       ª                   url.py
ª   ª   ª       ª                   wait.py
ª   ª   ª       ª                   __init__.py
ª   ª   ª       ª                   
ª   ª   ª       +---pip-24.3.1.dist-info
ª   ª   ª       ª       AUTHORS.txt
ª   ª   ª       ª       entry_points.txt
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       REQUESTED
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---pkg_resources
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---extern
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---_vendor
ª   ª   ª       ª       ª   appdirs.py
ª   ª   ª       ª       ª   pyparsing.py
ª   ª   ª       ª       ª   six.py
ª   ª   ª       ª       ª   __init__.py
ª   ª   ª       ª       ª   
ª   ª   ª       ª       +---packaging
ª   ª   ª       ª               markers.py
ª   ª   ª       ª               requirements.py
ª   ª   ª       ª               specifiers.py
ª   ª   ª       ª               tags.py
ª   ª   ª       ª               utils.py
ª   ª   ª       ª               version.py
ª   ª   ª       ª               _compat.py
ª   ª   ª       ª               _structures.py
ª   ª   ª       ª               __about__.py
ª   ª   ª       ª               __init__.py
ª   ª   ª       ª               
ª   ª   ª       +---proto
ª   ª   ª       ª   ª   datetime_helpers.py
ª   ª   ª       ª   ª   enums.py
ª   ª   ª       ª   ª   fields.py
ª   ª   ª       ª   ª   message.py
ª   ª   ª       ª   ª   modules.py
ª   ª   ª       ª   ª   primitives.py
ª   ª   ª       ª   ª   utils.py
ª   ª   ª       ª   ª   version.py
ª   ª   ª       ª   ª   _file_info.py
ª   ª   ª       ª   ª   _package_info.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---marshal
ª   ª   ª       ª       ª   compat.py
ª   ª   ª       ª       ª   marshal.py
ª   ª   ª       ª       ª   __init__.py
ª   ª   ª       ª       ª   
ª   ª   ª       ª       +---collections
ª   ª   ª       ª       ª       maps.py
ª   ª   ª       ª       ª       repeated.py
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---rules
ª   ª   ª       ª               bytes.py
ª   ª   ª       ª               dates.py
ª   ª   ª       ª               enums.py
ª   ª   ª       ª               field_mask.py
ª   ª   ª       ª               message.py
ª   ª   ª       ª               stringy_numbers.py
ª   ª   ª       ª               struct.py
ª   ª   ª       ª               wrappers.py
ª   ª   ª       ª               __init__.py
ª   ª   ª       ª               
ª   ª   ª       +---protobuf-5.28.3.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---proto_plus-1.25.0.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---pyasn1
ª   ª   ª       ª   ª   debug.py
ª   ª   ª       ª   ª   error.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---codec
ª   ª   ª       ª   ª   ª   streaming.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---ber
ª   ª   ª       ª   ª   ª       decoder.py
ª   ª   ª       ª   ª   ª       encoder.py
ª   ª   ª       ª   ª   ª       eoo.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---cer
ª   ª   ª       ª   ª   ª       decoder.py
ª   ª   ª       ª   ª   ª       encoder.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---der
ª   ª   ª       ª   ª   ª       decoder.py
ª   ª   ª       ª   ª   ª       encoder.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---native
ª   ª   ª       ª   ª           decoder.py
ª   ª   ª       ª   ª           encoder.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---compat
ª   ª   ª       ª   ª       integer.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---type
ª   ª   ª       ª           base.py
ª   ª   ª       ª           char.py
ª   ª   ª       ª           constraint.py
ª   ª   ª       ª           error.py
ª   ª   ª       ª           namedtype.py
ª   ª   ª       ª           namedval.py
ª   ª   ª       ª           opentype.py
ª   ª   ª       ª           tag.py
ª   ª   ª       ª           tagmap.py
ª   ª   ª       ª           univ.py
ª   ª   ª       ª           useful.py
ª   ª   ª       ª           __init__.py
ª   ª   ª       ª           
ª   ª   ª       +---pyasn1-0.6.1.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.rst
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       zip-safe
ª   ª   ª       ª       
ª   ª   ª       +---pyasn1_modules
ª   ª   ª       ª       pem.py
ª   ª   ª       ª       rfc1155.py
ª   ª   ª       ª       rfc1157.py
ª   ª   ª       ª       rfc1901.py
ª   ª   ª       ª       rfc1902.py
ª   ª   ª       ª       rfc1905.py
ª   ª   ª       ª       rfc2251.py
ª   ª   ª       ª       rfc2314.py
ª   ª   ª       ª       rfc2315.py
ª   ª   ª       ª       rfc2437.py
ª   ª   ª       ª       rfc2459.py
ª   ª   ª       ª       rfc2511.py
ª   ª   ª       ª       rfc2560.py
ª   ª   ª       ª       rfc2631.py
ª   ª   ª       ª       rfc2634.py
ª   ª   ª       ª       rfc2876.py
ª   ª   ª       ª       rfc2985.py
ª   ª   ª       ª       rfc2986.py
ª   ª   ª       ª       rfc3058.py
ª   ª   ª       ª       rfc3114.py
ª   ª   ª       ª       rfc3125.py
ª   ª   ª       ª       rfc3161.py
ª   ª   ª       ª       rfc3274.py
ª   ª   ª       ª       rfc3279.py
ª   ª   ª       ª       rfc3280.py
ª   ª   ª       ª       rfc3281.py
ª   ª   ª       ª       rfc3370.py
ª   ª   ª       ª       rfc3412.py
ª   ª   ª       ª       rfc3414.py
ª   ª   ª       ª       rfc3447.py
ª   ª   ª       ª       rfc3537.py
ª   ª   ª       ª       rfc3560.py
ª   ª   ª       ª       rfc3565.py
ª   ª   ª       ª       rfc3657.py
ª   ª   ª       ª       rfc3709.py
ª   ª   ª       ª       rfc3739.py
ª   ª   ª       ª       rfc3770.py
ª   ª   ª       ª       rfc3779.py
ª   ª   ª       ª       rfc3820.py
ª   ª   ª       ª       rfc3852.py
ª   ª   ª       ª       rfc4010.py
ª   ª   ª       ª       rfc4043.py
ª   ª   ª       ª       rfc4055.py
ª   ª   ª       ª       rfc4073.py
ª   ª   ª       ª       rfc4108.py
ª   ª   ª       ª       rfc4210.py
ª   ª   ª       ª       rfc4211.py
ª   ª   ª       ª       rfc4334.py
ª   ª   ª       ª       rfc4357.py
ª   ª   ª       ª       rfc4387.py
ª   ª   ª       ª       rfc4476.py
ª   ª   ª       ª       rfc4490.py
ª   ª   ª       ª       rfc4491.py
ª   ª   ª       ª       rfc4683.py
ª   ª   ª       ª       rfc4985.py
ª   ª   ª       ª       rfc5035.py
ª   ª   ª       ª       rfc5083.py
ª   ª   ª       ª       rfc5084.py
ª   ª   ª       ª       rfc5126.py
ª   ª   ª       ª       rfc5208.py
ª   ª   ª       ª       rfc5275.py
ª   ª   ª       ª       rfc5280.py
ª   ª   ª       ª       rfc5480.py
ª   ª   ª       ª       rfc5636.py
ª   ª   ª       ª       rfc5639.py
ª   ª   ª       ª       rfc5649.py
ª   ª   ª       ª       rfc5652.py
ª   ª   ª       ª       rfc5697.py
ª   ª   ª       ª       rfc5751.py
ª   ª   ª       ª       rfc5752.py
ª   ª   ª       ª       rfc5753.py
ª   ª   ª       ª       rfc5755.py
ª   ª   ª       ª       rfc5913.py
ª   ª   ª       ª       rfc5914.py
ª   ª   ª       ª       rfc5915.py
ª   ª   ª       ª       rfc5916.py
ª   ª   ª       ª       rfc5917.py
ª   ª   ª       ª       rfc5924.py
ª   ª   ª       ª       rfc5934.py
ª   ª   ª       ª       rfc5940.py
ª   ª   ª       ª       rfc5958.py
ª   ª   ª       ª       rfc5990.py
ª   ª   ª       ª       rfc6010.py
ª   ª   ª       ª       rfc6019.py
ª   ª   ª       ª       rfc6031.py
ª   ª   ª       ª       rfc6032.py
ª   ª   ª       ª       rfc6120.py
ª   ª   ª       ª       rfc6170.py
ª   ª   ª       ª       rfc6187.py
ª   ª   ª       ª       rfc6210.py
ª   ª   ª       ª       rfc6211.py
ª   ª   ª       ª       rfc6402.py
ª   ª   ª       ª       rfc6482.py
ª   ª   ª       ª       rfc6486.py
ª   ª   ª       ª       rfc6487.py
ª   ª   ª       ª       rfc6664.py
ª   ª   ª       ª       rfc6955.py
ª   ª   ª       ª       rfc6960.py
ª   ª   ª       ª       rfc7030.py
ª   ª   ª       ª       rfc7191.py
ª   ª   ª       ª       rfc7229.py
ª   ª   ª       ª       rfc7292.py
ª   ª   ª       ª       rfc7296.py
ª   ª   ª       ª       rfc7508.py
ª   ª   ª       ª       rfc7585.py
ª   ª   ª       ª       rfc7633.py
ª   ª   ª       ª       rfc7773.py
ª   ª   ª       ª       rfc7894.py
ª   ª   ª       ª       rfc7906.py
ª   ª   ª       ª       rfc7914.py
ª   ª   ª       ª       rfc8017.py
ª   ª   ª       ª       rfc8018.py
ª   ª   ª       ª       rfc8103.py
ª   ª   ª       ª       rfc8209.py
ª   ª   ª       ª       rfc8226.py
ª   ª   ª       ª       rfc8358.py
ª   ª   ª       ª       rfc8360.py
ª   ª   ª       ª       rfc8398.py
ª   ª   ª       ª       rfc8410.py
ª   ª   ª       ª       rfc8418.py
ª   ª   ª       ª       rfc8419.py
ª   ª   ª       ª       rfc8479.py
ª   ª   ª       ª       rfc8494.py
ª   ª   ª       ª       rfc8520.py
ª   ª   ª       ª       rfc8619.py
ª   ª   ª       ª       rfc8649.py
ª   ª   ª       ª       rfc8692.py
ª   ª   ª       ª       rfc8696.py
ª   ª   ª       ª       rfc8702.py
ª   ª   ª       ª       rfc8708.py
ª   ª   ª       ª       rfc8769.py
ª   ª   ª       ª       __init__.py
ª   ª   ª       ª       
ª   ª   ª       +---pyasn1_modules-0.4.1.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       zip-safe
ª   ª   ª       ª       
ª   ª   ª       +---python_dateutil-2.9.0.post0.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       zip-safe
ª   ª   ª       ª       
ª   ª   ª       +---pytz
ª   ª   ª       ª   ª   exceptions.py
ª   ª   ª       ª   ª   lazy.py
ª   ª   ª       ª   ª   reference.py
ª   ª   ª       ª   ª   tzfile.py
ª   ª   ª       ª   ª   tzinfo.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---zoneinfo
ª   ª   ª       ª       ª   CET
ª   ª   ª       ª       ª   CST6CDT
ª   ª   ª       ª       ª   Cuba
ª   ª   ª       ª       ª   EET
ª   ª   ª       ª       ª   Egypt
ª   ª   ª       ª       ª   Eire
ª   ª   ª       ª       ª   EST
ª   ª   ª       ª       ª   EST5EDT
ª   ª   ª       ª       ª   Factory
ª   ª   ª       ª       ª   GB
ª   ª   ª       ª       ª   GB-Eire
ª   ª   ª       ª       ª   GMT
ª   ª   ª       ª       ª   GMT+0
ª   ª   ª       ª       ª   GMT-0
ª   ª   ª       ª       ª   GMT0
ª   ª   ª       ª       ª   Greenwich
ª   ª   ª       ª       ª   Hongkong
ª   ª   ª       ª       ª   HST
ª   ª   ª       ª       ª   Iceland
ª   ª   ª       ª       ª   Iran
ª   ª   ª       ª       ª   iso3166.tab
ª   ª   ª       ª       ª   Israel
ª   ª   ª       ª       ª   Jamaica
ª   ª   ª       ª       ª   Japan
ª   ª   ª       ª       ª   Kwajalein
ª   ª   ª       ª       ª   leapseconds
ª   ª   ª       ª       ª   Libya
ª   ª   ª       ª       ª   MET
ª   ª   ª       ª       ª   MST
ª   ª   ª       ª       ª   MST7MDT
ª   ª   ª       ª       ª   Navajo
ª   ª   ª       ª       ª   NZ
ª   ª   ª       ª       ª   NZ-CHAT
ª   ª   ª       ª       ª   Poland
ª   ª   ª       ª       ª   Portugal
ª   ª   ª       ª       ª   PRC
ª   ª   ª       ª       ª   PST8PDT
ª   ª   ª       ª       ª   ROC
ª   ª   ª       ª       ª   ROK
ª   ª   ª       ª       ª   Singapore
ª   ª   ª       ª       ª   Turkey
ª   ª   ª       ª       ª   tzdata.zi
ª   ª   ª       ª       ª   UCT
ª   ª   ª       ª       ª   Universal
ª   ª   ª       ª       ª   UTC
ª   ª   ª       ª       ª   W-SU
ª   ª   ª       ª       ª   WET
ª   ª   ª       ª       ª   zone.tab
ª   ª   ª       ª       ª   zone1970.tab
ª   ª   ª       ª       ª   zonenow.tab
ª   ª   ª       ª       ª   Zulu
ª   ª   ª       ª       ª   
ª   ª   ª       ª       +---Africa
ª   ª   ª       ª       ª       Abidjan
ª   ª   ª       ª       ª       Accra
ª   ª   ª       ª       ª       Addis_Ababa
ª   ª   ª       ª       ª       Algiers
ª   ª   ª       ª       ª       Asmara
ª   ª   ª       ª       ª       Asmera
ª   ª   ª       ª       ª       Bamako
ª   ª   ª       ª       ª       Bangui
ª   ª   ª       ª       ª       Banjul
ª   ª   ª       ª       ª       Bissau
ª   ª   ª       ª       ª       Blantyre
ª   ª   ª       ª       ª       Brazzaville
ª   ª   ª       ª       ª       Bujumbura
ª   ª   ª       ª       ª       Cairo
ª   ª   ª       ª       ª       Casablanca
ª   ª   ª       ª       ª       Ceuta
ª   ª   ª       ª       ª       Conakry
ª   ª   ª       ª       ª       Dakar
ª   ª   ª       ª       ª       Dar_es_Salaam
ª   ª   ª       ª       ª       Djibouti
ª   ª   ª       ª       ª       Douala
ª   ª   ª       ª       ª       El_Aaiun
ª   ª   ª       ª       ª       Freetown
ª   ª   ª       ª       ª       Gaborone
ª   ª   ª       ª       ª       Harare
ª   ª   ª       ª       ª       Johannesburg
ª   ª   ª       ª       ª       Juba
ª   ª   ª       ª       ª       Kampala
ª   ª   ª       ª       ª       Khartoum
ª   ª   ª       ª       ª       Kigali
ª   ª   ª       ª       ª       Kinshasa
ª   ª   ª       ª       ª       Lagos
ª   ª   ª       ª       ª       Libreville
ª   ª   ª       ª       ª       Lome
ª   ª   ª       ª       ª       Luanda
ª   ª   ª       ª       ª       Lubumbashi
ª   ª   ª       ª       ª       Lusaka
ª   ª   ª       ª       ª       Malabo
ª   ª   ª       ª       ª       Maputo
ª   ª   ª       ª       ª       Maseru
ª   ª   ª       ª       ª       Mbabane
ª   ª   ª       ª       ª       Mogadishu
ª   ª   ª       ª       ª       Monrovia
ª   ª   ª       ª       ª       Nairobi
ª   ª   ª       ª       ª       Ndjamena
ª   ª   ª       ª       ª       Niamey
ª   ª   ª       ª       ª       Nouakchott
ª   ª   ª       ª       ª       Ouagadougou
ª   ª   ª       ª       ª       Porto-Novo
ª   ª   ª       ª       ª       Sao_Tome
ª   ª   ª       ª       ª       Timbuktu
ª   ª   ª       ª       ª       Tripoli
ª   ª   ª       ª       ª       Tunis
ª   ª   ª       ª       ª       Windhoek
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---America
ª   ª   ª       ª       ª   ª   Adak
ª   ª   ª       ª       ª   ª   Anchorage
ª   ª   ª       ª       ª   ª   Anguilla
ª   ª   ª       ª       ª   ª   Antigua
ª   ª   ª       ª       ª   ª   Araguaina
ª   ª   ª       ª       ª   ª   Aruba
ª   ª   ª       ª       ª   ª   Asuncion
ª   ª   ª       ª       ª   ª   Atikokan
ª   ª   ª       ª       ª   ª   Atka
ª   ª   ª       ª       ª   ª   Bahia
ª   ª   ª       ª       ª   ª   Bahia_Banderas
ª   ª   ª       ª       ª   ª   Barbados
ª   ª   ª       ª       ª   ª   Belem
ª   ª   ª       ª       ª   ª   Belize
ª   ª   ª       ª       ª   ª   Blanc-Sablon
ª   ª   ª       ª       ª   ª   Boa_Vista
ª   ª   ª       ª       ª   ª   Bogota
ª   ª   ª       ª       ª   ª   Boise
ª   ª   ª       ª       ª   ª   Buenos_Aires
ª   ª   ª       ª       ª   ª   Cambridge_Bay
ª   ª   ª       ª       ª   ª   Campo_Grande
ª   ª   ª       ª       ª   ª   Cancun
ª   ª   ª       ª       ª   ª   Caracas
ª   ª   ª       ª       ª   ª   Catamarca
ª   ª   ª       ª       ª   ª   Cayenne
ª   ª   ª       ª       ª   ª   Cayman
ª   ª   ª       ª       ª   ª   Chicago
ª   ª   ª       ª       ª   ª   Chihuahua
ª   ª   ª       ª       ª   ª   Ciudad_Juarez
ª   ª   ª       ª       ª   ª   Coral_Harbour
ª   ª   ª       ª       ª   ª   Cordoba
ª   ª   ª       ª       ª   ª   Costa_Rica
ª   ª   ª       ª       ª   ª   Creston
ª   ª   ª       ª       ª   ª   Cuiaba
ª   ª   ª       ª       ª   ª   Curacao
ª   ª   ª       ª       ª   ª   Danmarkshavn
ª   ª   ª       ª       ª   ª   Dawson
ª   ª   ª       ª       ª   ª   Dawson_Creek
ª   ª   ª       ª       ª   ª   Denver
ª   ª   ª       ª       ª   ª   Detroit
ª   ª   ª       ª       ª   ª   Dominica
ª   ª   ª       ª       ª   ª   Edmonton
ª   ª   ª       ª       ª   ª   Eirunepe
ª   ª   ª       ª       ª   ª   El_Salvador
ª   ª   ª       ª       ª   ª   Ensenada
ª   ª   ª       ª       ª   ª   Fortaleza
ª   ª   ª       ª       ª   ª   Fort_Nelson
ª   ª   ª       ª       ª   ª   Fort_Wayne
ª   ª   ª       ª       ª   ª   Glace_Bay
ª   ª   ª       ª       ª   ª   Godthab
ª   ª   ª       ª       ª   ª   Goose_Bay
ª   ª   ª       ª       ª   ª   Grand_Turk
ª   ª   ª       ª       ª   ª   Grenada
ª   ª   ª       ª       ª   ª   Guadeloupe
ª   ª   ª       ª       ª   ª   Guatemala
ª   ª   ª       ª       ª   ª   Guayaquil
ª   ª   ª       ª       ª   ª   Guyana
ª   ª   ª       ª       ª   ª   Halifax
ª   ª   ª       ª       ª   ª   Havana
ª   ª   ª       ª       ª   ª   Hermosillo
ª   ª   ª       ª       ª   ª   Indianapolis
ª   ª   ª       ª       ª   ª   Inuvik
ª   ª   ª       ª       ª   ª   Iqaluit
ª   ª   ª       ª       ª   ª   Jamaica
ª   ª   ª       ª       ª   ª   Jujuy
ª   ª   ª       ª       ª   ª   Juneau
ª   ª   ª       ª       ª   ª   Knox_IN
ª   ª   ª       ª       ª   ª   Kralendijk
ª   ª   ª       ª       ª   ª   La_Paz
ª   ª   ª       ª       ª   ª   Lima
ª   ª   ª       ª       ª   ª   Los_Angeles
ª   ª   ª       ª       ª   ª   Louisville
ª   ª   ª       ª       ª   ª   Lower_Princes
ª   ª   ª       ª       ª   ª   Maceio
ª   ª   ª       ª       ª   ª   Managua
ª   ª   ª       ª       ª   ª   Manaus
ª   ª   ª       ª       ª   ª   Marigot
ª   ª   ª       ª       ª   ª   Martinique
ª   ª   ª       ª       ª   ª   Matamoros
ª   ª   ª       ª       ª   ª   Mazatlan
ª   ª   ª       ª       ª   ª   Mendoza
ª   ª   ª       ª       ª   ª   Menominee
ª   ª   ª       ª       ª   ª   Merida
ª   ª   ª       ª       ª   ª   Metlakatla
ª   ª   ª       ª       ª   ª   Mexico_City
ª   ª   ª       ª       ª   ª   Miquelon
ª   ª   ª       ª       ª   ª   Moncton
ª   ª   ª       ª       ª   ª   Monterrey
ª   ª   ª       ª       ª   ª   Montevideo
ª   ª   ª       ª       ª   ª   Montreal
ª   ª   ª       ª       ª   ª   Montserrat
ª   ª   ª       ª       ª   ª   Nassau
ª   ª   ª       ª       ª   ª   New_York
ª   ª   ª       ª       ª   ª   Nipigon
ª   ª   ª       ª       ª   ª   Nome
ª   ª   ª       ª       ª   ª   Noronha
ª   ª   ª       ª       ª   ª   Nuuk
ª   ª   ª       ª       ª   ª   Ojinaga
ª   ª   ª       ª       ª   ª   Panama
ª   ª   ª       ª       ª   ª   Pangnirtung
ª   ª   ª       ª       ª   ª   Paramaribo
ª   ª   ª       ª       ª   ª   Phoenix
ª   ª   ª       ª       ª   ª   Port-au-Prince
ª   ª   ª       ª       ª   ª   Porto_Acre
ª   ª   ª       ª       ª   ª   Porto_Velho
ª   ª   ª       ª       ª   ª   Port_of_Spain
ª   ª   ª       ª       ª   ª   Puerto_Rico
ª   ª   ª       ª       ª   ª   Punta_Arenas
ª   ª   ª       ª       ª   ª   Rainy_River
ª   ª   ª       ª       ª   ª   Rankin_Inlet
ª   ª   ª       ª       ª   ª   Recife
ª   ª   ª       ª       ª   ª   Regina
ª   ª   ª       ª       ª   ª   Resolute
ª   ª   ª       ª       ª   ª   Rio_Branco
ª   ª   ª       ª       ª   ª   Rosario
ª   ª   ª       ª       ª   ª   Santarem
ª   ª   ª       ª       ª   ª   Santa_Isabel
ª   ª   ª       ª       ª   ª   Santiago
ª   ª   ª       ª       ª   ª   Santo_Domingo
ª   ª   ª       ª       ª   ª   Sao_Paulo
ª   ª   ª       ª       ª   ª   Scoresbysund
ª   ª   ª       ª       ª   ª   Shiprock
ª   ª   ª       ª       ª   ª   Sitka
ª   ª   ª       ª       ª   ª   St_Barthelemy
ª   ª   ª       ª       ª   ª   St_Johns
ª   ª   ª       ª       ª   ª   St_Kitts
ª   ª   ª       ª       ª   ª   St_Lucia
ª   ª   ª       ª       ª   ª   St_Thomas
ª   ª   ª       ª       ª   ª   St_Vincent
ª   ª   ª       ª       ª   ª   Swift_Current
ª   ª   ª       ª       ª   ª   Tegucigalpa
ª   ª   ª       ª       ª   ª   Thule
ª   ª   ª       ª       ª   ª   Thunder_Bay
ª   ª   ª       ª       ª   ª   Tijuana
ª   ª   ª       ª       ª   ª   Toronto
ª   ª   ª       ª       ª   ª   Tortola
ª   ª   ª       ª       ª   ª   Vancouver
ª   ª   ª       ª       ª   ª   Virgin
ª   ª   ª       ª       ª   ª   Whitehorse
ª   ª   ª       ª       ª   ª   Winnipeg
ª   ª   ª       ª       ª   ª   Yakutat
ª   ª   ª       ª       ª   ª   Yellowknife
ª   ª   ª       ª       ª   ª   
ª   ª   ª       ª       ª   +---Argentina
ª   ª   ª       ª       ª   ª       Buenos_Aires
ª   ª   ª       ª       ª   ª       Catamarca
ª   ª   ª       ª       ª   ª       ComodRivadavia
ª   ª   ª       ª       ª   ª       Cordoba
ª   ª   ª       ª       ª   ª       Jujuy
ª   ª   ª       ª       ª   ª       La_Rioja
ª   ª   ª       ª       ª   ª       Mendoza
ª   ª   ª       ª       ª   ª       Rio_Gallegos
ª   ª   ª       ª       ª   ª       Salta
ª   ª   ª       ª       ª   ª       San_Juan
ª   ª   ª       ª       ª   ª       San_Luis
ª   ª   ª       ª       ª   ª       Tucuman
ª   ª   ª       ª       ª   ª       Ushuaia
ª   ª   ª       ª       ª   ª       
ª   ª   ª       ª       ª   +---Indiana
ª   ª   ª       ª       ª   ª       Indianapolis
ª   ª   ª       ª       ª   ª       Knox
ª   ª   ª       ª       ª   ª       Marengo
ª   ª   ª       ª       ª   ª       Petersburg
ª   ª   ª       ª       ª   ª       Tell_City
ª   ª   ª       ª       ª   ª       Vevay
ª   ª   ª       ª       ª   ª       Vincennes
ª   ª   ª       ª       ª   ª       Winamac
ª   ª   ª       ª       ª   ª       
ª   ª   ª       ª       ª   +---Kentucky
ª   ª   ª       ª       ª   ª       Louisville
ª   ª   ª       ª       ª   ª       Monticello
ª   ª   ª       ª       ª   ª       
ª   ª   ª       ª       ª   +---North_Dakota
ª   ª   ª       ª       ª           Beulah
ª   ª   ª       ª       ª           Center
ª   ª   ª       ª       ª           New_Salem
ª   ª   ª       ª       ª           
ª   ª   ª       ª       +---Antarctica
ª   ª   ª       ª       ª       Casey
ª   ª   ª       ª       ª       Davis
ª   ª   ª       ª       ª       DumontDUrville
ª   ª   ª       ª       ª       Macquarie
ª   ª   ª       ª       ª       Mawson
ª   ª   ª       ª       ª       McMurdo
ª   ª   ª       ª       ª       Palmer
ª   ª   ª       ª       ª       Rothera
ª   ª   ª       ª       ª       South_Pole
ª   ª   ª       ª       ª       Syowa
ª   ª   ª       ª       ª       Troll
ª   ª   ª       ª       ª       Vostok
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Arctic
ª   ª   ª       ª       ª       Longyearbyen
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Asia
ª   ª   ª       ª       ª       Aden
ª   ª   ª       ª       ª       Almaty
ª   ª   ª       ª       ª       Amman
ª   ª   ª       ª       ª       Anadyr
ª   ª   ª       ª       ª       Aqtau
ª   ª   ª       ª       ª       Aqtobe
ª   ª   ª       ª       ª       Ashgabat
ª   ª   ª       ª       ª       Ashkhabad
ª   ª   ª       ª       ª       Atyrau
ª   ª   ª       ª       ª       Baghdad
ª   ª   ª       ª       ª       Bahrain
ª   ª   ª       ª       ª       Baku
ª   ª   ª       ª       ª       Bangkok
ª   ª   ª       ª       ª       Barnaul
ª   ª   ª       ª       ª       Beirut
ª   ª   ª       ª       ª       Bishkek
ª   ª   ª       ª       ª       Brunei
ª   ª   ª       ª       ª       Calcutta
ª   ª   ª       ª       ª       Chita
ª   ª   ª       ª       ª       Choibalsan
ª   ª   ª       ª       ª       Chongqing
ª   ª   ª       ª       ª       Chungking
ª   ª   ª       ª       ª       Colombo
ª   ª   ª       ª       ª       Dacca
ª   ª   ª       ª       ª       Damascus
ª   ª   ª       ª       ª       Dhaka
ª   ª   ª       ª       ª       Dili
ª   ª   ª       ª       ª       Dubai
ª   ª   ª       ª       ª       Dushanbe
ª   ª   ª       ª       ª       Famagusta
ª   ª   ª       ª       ª       Gaza
ª   ª   ª       ª       ª       Harbin
ª   ª   ª       ª       ª       Hebron
ª   ª   ª       ª       ª       Hong_Kong
ª   ª   ª       ª       ª       Hovd
ª   ª   ª       ª       ª       Ho_Chi_Minh
ª   ª   ª       ª       ª       Irkutsk
ª   ª   ª       ª       ª       Istanbul
ª   ª   ª       ª       ª       Jakarta
ª   ª   ª       ª       ª       Jayapura
ª   ª   ª       ª       ª       Jerusalem
ª   ª   ª       ª       ª       Kabul
ª   ª   ª       ª       ª       Kamchatka
ª   ª   ª       ª       ª       Karachi
ª   ª   ª       ª       ª       Kashgar
ª   ª   ª       ª       ª       Kathmandu
ª   ª   ª       ª       ª       Katmandu
ª   ª   ª       ª       ª       Khandyga
ª   ª   ª       ª       ª       Kolkata
ª   ª   ª       ª       ª       Krasnoyarsk
ª   ª   ª       ª       ª       Kuala_Lumpur
ª   ª   ª       ª       ª       Kuching
ª   ª   ª       ª       ª       Kuwait
ª   ª   ª       ª       ª       Macao
ª   ª   ª       ª       ª       Macau
ª   ª   ª       ª       ª       Magadan
ª   ª   ª       ª       ª       Makassar
ª   ª   ª       ª       ª       Manila
ª   ª   ª       ª       ª       Muscat
ª   ª   ª       ª       ª       Nicosia
ª   ª   ª       ª       ª       Novokuznetsk
ª   ª   ª       ª       ª       Novosibirsk
ª   ª   ª       ª       ª       Omsk
ª   ª   ª       ª       ª       Oral
ª   ª   ª       ª       ª       Phnom_Penh
ª   ª   ª       ª       ª       Pontianak
ª   ª   ª       ª       ª       Pyongyang
ª   ª   ª       ª       ª       Qatar
ª   ª   ª       ª       ª       Qostanay
ª   ª   ª       ª       ª       Qyzylorda
ª   ª   ª       ª       ª       Rangoon
ª   ª   ª       ª       ª       Riyadh
ª   ª   ª       ª       ª       Saigon
ª   ª   ª       ª       ª       Sakhalin
ª   ª   ª       ª       ª       Samarkand
ª   ª   ª       ª       ª       Seoul
ª   ª   ª       ª       ª       Shanghai
ª   ª   ª       ª       ª       Singapore
ª   ª   ª       ª       ª       Srednekolymsk
ª   ª   ª       ª       ª       Taipei
ª   ª   ª       ª       ª       Tashkent
ª   ª   ª       ª       ª       Tbilisi
ª   ª   ª       ª       ª       Tehran
ª   ª   ª       ª       ª       Tel_Aviv
ª   ª   ª       ª       ª       Thimbu
ª   ª   ª       ª       ª       Thimphu
ª   ª   ª       ª       ª       Tokyo
ª   ª   ª       ª       ª       Tomsk
ª   ª   ª       ª       ª       Ujung_Pandang
ª   ª   ª       ª       ª       Ulaanbaatar
ª   ª   ª       ª       ª       Ulan_Bator
ª   ª   ª       ª       ª       Urumqi
ª   ª   ª       ª       ª       Ust-Nera
ª   ª   ª       ª       ª       Vientiane
ª   ª   ª       ª       ª       Vladivostok
ª   ª   ª       ª       ª       Yakutsk
ª   ª   ª       ª       ª       Yangon
ª   ª   ª       ª       ª       Yekaterinburg
ª   ª   ª       ª       ª       Yerevan
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Atlantic
ª   ª   ª       ª       ª       Azores
ª   ª   ª       ª       ª       Bermuda
ª   ª   ª       ª       ª       Canary
ª   ª   ª       ª       ª       Cape_Verde
ª   ª   ª       ª       ª       Faeroe
ª   ª   ª       ª       ª       Faroe
ª   ª   ª       ª       ª       Jan_Mayen
ª   ª   ª       ª       ª       Madeira
ª   ª   ª       ª       ª       Reykjavik
ª   ª   ª       ª       ª       South_Georgia
ª   ª   ª       ª       ª       Stanley
ª   ª   ª       ª       ª       St_Helena
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Australia
ª   ª   ª       ª       ª       ACT
ª   ª   ª       ª       ª       Adelaide
ª   ª   ª       ª       ª       Brisbane
ª   ª   ª       ª       ª       Broken_Hill
ª   ª   ª       ª       ª       Canberra
ª   ª   ª       ª       ª       Currie
ª   ª   ª       ª       ª       Darwin
ª   ª   ª       ª       ª       Eucla
ª   ª   ª       ª       ª       Hobart
ª   ª   ª       ª       ª       LHI
ª   ª   ª       ª       ª       Lindeman
ª   ª   ª       ª       ª       Lord_Howe
ª   ª   ª       ª       ª       Melbourne
ª   ª   ª       ª       ª       North
ª   ª   ª       ª       ª       NSW
ª   ª   ª       ª       ª       Perth
ª   ª   ª       ª       ª       Queensland
ª   ª   ª       ª       ª       South
ª   ª   ª       ª       ª       Sydney
ª   ª   ª       ª       ª       Tasmania
ª   ª   ª       ª       ª       Victoria
ª   ª   ª       ª       ª       West
ª   ª   ª       ª       ª       Yancowinna
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Brazil
ª   ª   ª       ª       ª       Acre
ª   ª   ª       ª       ª       DeNoronha
ª   ª   ª       ª       ª       East
ª   ª   ª       ª       ª       West
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Canada
ª   ª   ª       ª       ª       Atlantic
ª   ª   ª       ª       ª       Central
ª   ª   ª       ª       ª       Eastern
ª   ª   ª       ª       ª       Mountain
ª   ª   ª       ª       ª       Newfoundland
ª   ª   ª       ª       ª       Pacific
ª   ª   ª       ª       ª       Saskatchewan
ª   ª   ª       ª       ª       Yukon
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Chile
ª   ª   ª       ª       ª       Continental
ª   ª   ª       ª       ª       EasterIsland
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Etc
ª   ª   ª       ª       ª       GMT
ª   ª   ª       ª       ª       GMT+0
ª   ª   ª       ª       ª       GMT+1
ª   ª   ª       ª       ª       GMT+10
ª   ª   ª       ª       ª       GMT+11
ª   ª   ª       ª       ª       GMT+12
ª   ª   ª       ª       ª       GMT+2
ª   ª   ª       ª       ª       GMT+3
ª   ª   ª       ª       ª       GMT+4
ª   ª   ª       ª       ª       GMT+5
ª   ª   ª       ª       ª       GMT+6
ª   ª   ª       ª       ª       GMT+7
ª   ª   ª       ª       ª       GMT+8
ª   ª   ª       ª       ª       GMT+9
ª   ª   ª       ª       ª       GMT-0
ª   ª   ª       ª       ª       GMT-1
ª   ª   ª       ª       ª       GMT-10
ª   ª   ª       ª       ª       GMT-11
ª   ª   ª       ª       ª       GMT-12
ª   ª   ª       ª       ª       GMT-13
ª   ª   ª       ª       ª       GMT-14
ª   ª   ª       ª       ª       GMT-2
ª   ª   ª       ª       ª       GMT-3
ª   ª   ª       ª       ª       GMT-4
ª   ª   ª       ª       ª       GMT-5
ª   ª   ª       ª       ª       GMT-6
ª   ª   ª       ª       ª       GMT-7
ª   ª   ª       ª       ª       GMT-8
ª   ª   ª       ª       ª       GMT-9
ª   ª   ª       ª       ª       GMT0
ª   ª   ª       ª       ª       Greenwich
ª   ª   ª       ª       ª       UCT
ª   ª   ª       ª       ª       Universal
ª   ª   ª       ª       ª       UTC
ª   ª   ª       ª       ª       Zulu
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Europe
ª   ª   ª       ª       ª       Amsterdam
ª   ª   ª       ª       ª       Andorra
ª   ª   ª       ª       ª       Astrakhan
ª   ª   ª       ª       ª       Athens
ª   ª   ª       ª       ª       Belfast
ª   ª   ª       ª       ª       Belgrade
ª   ª   ª       ª       ª       Berlin
ª   ª   ª       ª       ª       Bratislava
ª   ª   ª       ª       ª       Brussels
ª   ª   ª       ª       ª       Bucharest
ª   ª   ª       ª       ª       Budapest
ª   ª   ª       ª       ª       Busingen
ª   ª   ª       ª       ª       Chisinau
ª   ª   ª       ª       ª       Copenhagen
ª   ª   ª       ª       ª       Dublin
ª   ª   ª       ª       ª       Gibraltar
ª   ª   ª       ª       ª       Guernsey
ª   ª   ª       ª       ª       Helsinki
ª   ª   ª       ª       ª       Isle_of_Man
ª   ª   ª       ª       ª       Istanbul
ª   ª   ª       ª       ª       Jersey
ª   ª   ª       ª       ª       Kaliningrad
ª   ª   ª       ª       ª       Kiev
ª   ª   ª       ª       ª       Kirov
ª   ª   ª       ª       ª       Kyiv
ª   ª   ª       ª       ª       Lisbon
ª   ª   ª       ª       ª       Ljubljana
ª   ª   ª       ª       ª       London
ª   ª   ª       ª       ª       Luxembourg
ª   ª   ª       ª       ª       Madrid
ª   ª   ª       ª       ª       Malta
ª   ª   ª       ª       ª       Mariehamn
ª   ª   ª       ª       ª       Minsk
ª   ª   ª       ª       ª       Monaco
ª   ª   ª       ª       ª       Moscow
ª   ª   ª       ª       ª       Nicosia
ª   ª   ª       ª       ª       Oslo
ª   ª   ª       ª       ª       Paris
ª   ª   ª       ª       ª       Podgorica
ª   ª   ª       ª       ª       Prague
ª   ª   ª       ª       ª       Riga
ª   ª   ª       ª       ª       Rome
ª   ª   ª       ª       ª       Samara
ª   ª   ª       ª       ª       San_Marino
ª   ª   ª       ª       ª       Sarajevo
ª   ª   ª       ª       ª       Saratov
ª   ª   ª       ª       ª       Simferopol
ª   ª   ª       ª       ª       Skopje
ª   ª   ª       ª       ª       Sofia
ª   ª   ª       ª       ª       Stockholm
ª   ª   ª       ª       ª       Tallinn
ª   ª   ª       ª       ª       Tirane
ª   ª   ª       ª       ª       Tiraspol
ª   ª   ª       ª       ª       Ulyanovsk
ª   ª   ª       ª       ª       Uzhgorod
ª   ª   ª       ª       ª       Vaduz
ª   ª   ª       ª       ª       Vatican
ª   ª   ª       ª       ª       Vienna
ª   ª   ª       ª       ª       Vilnius
ª   ª   ª       ª       ª       Volgograd
ª   ª   ª       ª       ª       Warsaw
ª   ª   ª       ª       ª       Zagreb
ª   ª   ª       ª       ª       Zaporozhye
ª   ª   ª       ª       ª       Zurich
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Indian
ª   ª   ª       ª       ª       Antananarivo
ª   ª   ª       ª       ª       Chagos
ª   ª   ª       ª       ª       Christmas
ª   ª   ª       ª       ª       Cocos
ª   ª   ª       ª       ª       Comoro
ª   ª   ª       ª       ª       Kerguelen
ª   ª   ª       ª       ª       Mahe
ª   ª   ª       ª       ª       Maldives
ª   ª   ª       ª       ª       Mauritius
ª   ª   ª       ª       ª       Mayotte
ª   ª   ª       ª       ª       Reunion
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Mexico
ª   ª   ª       ª       ª       BajaNorte
ª   ª   ª       ª       ª       BajaSur
ª   ª   ª       ª       ª       General
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Pacific
ª   ª   ª       ª       ª       Apia
ª   ª   ª       ª       ª       Auckland
ª   ª   ª       ª       ª       Bougainville
ª   ª   ª       ª       ª       Chatham
ª   ª   ª       ª       ª       Chuuk
ª   ª   ª       ª       ª       Easter
ª   ª   ª       ª       ª       Efate
ª   ª   ª       ª       ª       Enderbury
ª   ª   ª       ª       ª       Fakaofo
ª   ª   ª       ª       ª       Fiji
ª   ª   ª       ª       ª       Funafuti
ª   ª   ª       ª       ª       Galapagos
ª   ª   ª       ª       ª       Gambier
ª   ª   ª       ª       ª       Guadalcanal
ª   ª   ª       ª       ª       Guam
ª   ª   ª       ª       ª       Honolulu
ª   ª   ª       ª       ª       Johnston
ª   ª   ª       ª       ª       Kanton
ª   ª   ª       ª       ª       Kiritimati
ª   ª   ª       ª       ª       Kosrae
ª   ª   ª       ª       ª       Kwajalein
ª   ª   ª       ª       ª       Majuro
ª   ª   ª       ª       ª       Marquesas
ª   ª   ª       ª       ª       Midway
ª   ª   ª       ª       ª       Nauru
ª   ª   ª       ª       ª       Niue
ª   ª   ª       ª       ª       Norfolk
ª   ª   ª       ª       ª       Noumea
ª   ª   ª       ª       ª       Pago_Pago
ª   ª   ª       ª       ª       Palau
ª   ª   ª       ª       ª       Pitcairn
ª   ª   ª       ª       ª       Pohnpei
ª   ª   ª       ª       ª       Ponape
ª   ª   ª       ª       ª       Port_Moresby
ª   ª   ª       ª       ª       Rarotonga
ª   ª   ª       ª       ª       Saipan
ª   ª   ª       ª       ª       Samoa
ª   ª   ª       ª       ª       Tahiti
ª   ª   ª       ª       ª       Tarawa
ª   ª   ª       ª       ª       Tongatapu
ª   ª   ª       ª       ª       Truk
ª   ª   ª       ª       ª       Wake
ª   ª   ª       ª       ª       Wallis
ª   ª   ª       ª       ª       Yap
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---US
ª   ª   ª       ª               Alaska
ª   ª   ª       ª               Aleutian
ª   ª   ª       ª               Arizona
ª   ª   ª       ª               Central
ª   ª   ª       ª               East-Indiana
ª   ª   ª       ª               Eastern
ª   ª   ª       ª               Hawaii
ª   ª   ª       ª               Indiana-Starke
ª   ª   ª       ª               Michigan
ª   ª   ª       ª               Mountain
ª   ª   ª       ª               Pacific
ª   ª   ª       ª               Samoa
ª   ª   ª       ª               
ª   ª   ª       +---pytz-2024.2.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       zip-safe
ª   ª   ª       ª       
ª   ª   ª       +---requests
ª   ª   ª       ª       adapters.py
ª   ª   ª       ª       api.py
ª   ª   ª       ª       auth.py
ª   ª   ª       ª       certs.py
ª   ª   ª       ª       compat.py
ª   ª   ª       ª       cookies.py
ª   ª   ª       ª       exceptions.py
ª   ª   ª       ª       help.py
ª   ª   ª       ª       hooks.py
ª   ª   ª       ª       models.py
ª   ª   ª       ª       packages.py
ª   ª   ª       ª       sessions.py
ª   ª   ª       ª       status_codes.py
ª   ª   ª       ª       structures.py
ª   ª   ª       ª       utils.py
ª   ª   ª       ª       _internal_utils.py
ª   ª   ª       ª       __init__.py
ª   ª   ª       ª       __version__.py
ª   ª   ª       ª       
ª   ª   ª       +---requests-2.32.3.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---rsa
ª   ª   ª       ª       asn1.py
ª   ª   ª       ª       cli.py
ª   ª   ª       ª       common.py
ª   ª   ª       ª       core.py
ª   ª   ª       ª       key.py
ª   ª   ª       ª       parallel.py
ª   ª   ª       ª       pem.py
ª   ª   ª       ª       pkcs1.py
ª   ª   ª       ª       pkcs1_v2.py
ª   ª   ª       ª       prime.py
ª   ª   ª       ª       py.typed
ª   ª   ª       ª       randnum.py
ª   ª   ª       ª       transform.py
ª   ª   ª       ª       util.py
ª   ª   ª       ª       __init__.py
ª   ª   ª       ª       
ª   ª   ª       +---rsa-4.9.dist-info
ª   ª   ª       ª       entry_points.txt
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---scipy
ª   ª   ª       ª   ª   conftest.py
ª   ª   ª       ª   ª   doc_requirements.txt
ª   ª   ª       ª   ª   HACKING.rst.txt
ª   ª   ª       ª   ª   INSTALL.rst.txt
ª   ª   ª       ª   ª   LICENSE.txt
ª   ª   ª       ª   ª   LICENSES_bundled.txt
ª   ª   ª       ª   ª   linalg.pxd
ª   ª   ª       ª   ª   meson_options.txt
ª   ª   ª       ª   ª   mypy_requirements.txt
ª   ª   ª       ª   ª   optimize.pxd
ª   ª   ª       ª   ª   setup.py
ª   ª   ª       ª   ª   special.pxd
ª   ª   ª       ª   ª   version.py
ª   ª   ª       ª   ª   _distributor_init.py
ª   ª   ª       ª   ª   __config__.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---.libs
ª   ª   ª       ª   ª       libansari.MKH3DK7244XLLQVOJ7IYCVGVQPIB3QTC.gfortran-win32.dll
ª   ª   ª       ª   ª       libbanded5x.X7LAILNEDDW4E5ZH44Z3OQD5WAE5SVLP.gfortran-win32.dll
ª   ª   ª       ª   ª       libbispeu.3BVIZEHSB4TCRQSVUSVJEPQAFV7CXRHP.gfortran-win32.dll
ª   ª   ª       ª   ª       libblkdta00.C5RO7VWB3IWXLE7L57ZPET32CTOFHWRD.gfortran-win32.dll
ª   ª   ª       ª   ª       libchkder.3GB4LLWY3KMIB3XZQCXI54FI6MYSP3C3.gfortran-win32.dll
ª   ª   ª       ª   ª       libcobyla2.U7QIGNKXCGGLA7I2NW6J3FEWV334WBZ4.gfortran-win32.dll
ª   ª   ª       ª   ª       libdcsrch.KHKDWS7VOS5FQTNSLN2VDOCM7JGSZT4J.gfortran-win32.dll
ª   ª   ª       ª   ª       libdet.FMPN2L5V52Z5HGOHABZUWMIRJ32ZST6Y.gfortran-win32.dll
ª   ª   ª       ª   ª       libdfft.BHTK3PXD3CJPQKGV2ODEFWX3MXJSJQUF.gfortran-win32.dll
ª   ª   ª       ª   ª       libdfitpack.2ZAEUU6DHDKF3TCRNFY5V42OZOGACOI3.gfortran-win32.dll
ª   ª   ª       ª   ª       libdgamln.3L7ID2OLBXFD46N2CQ43JWHQ4LZSEP5M.gfortran-win32.dll
ª   ª   ª       ª   ª       libdqag.NMX23FLRSHB434UXHNBDFRDOFLKLRKPR.gfortran-win32.dll
ª   ª   ª       ª   ª       libd_odr.JTAKKU5UHPY3Q6TQMOBE3MYGWOLHB732.gfortran-win32.dll
ª   ª   ª       ª   ª       libgetbreak.FEHTDQBGWG27H7L52CCJ3XQ25QGLREKK.gfortran-win32.dll
ª   ª   ª       ª   ª       liblbfgsb.CKO2XLYJ3YBRLANPRW6ZDCCQMPSCEGOI.gfortran-win32.dll
ª   ª   ª       ª   ª       libmvndst.DLZXPRE6UYN6DB3URVHUJU6VY6GOPG2G.gfortran-win32.dll
ª   ª   ª       ª   ª       libnnls.WLUX2GRNZY6AXH2QF33W4V5VKZSQZ7GZ.gfortran-win32.dll
ª   ª   ª       ª   ª       libopenblas.67LVI5PIPIPYANQC3VAWDE6KZKQBFZON.gfortran-win32.dll
ª   ª   ª       ª   ª       libslsqp_op.YEJXW5WM5WHULDR4LP2CECQWMKYSU2V7.gfortran-win32.dll
ª   ª   ª       ª   ª       libspecfun.LG5CXJHT3OOMEIW3YR7MYFGDVQDBQS2M.gfortran-win32.dll
ª   ª   ª       ª   ª       libwrap_dum.GOB3UPKGKMLLVPOJLDDZVS2EYAG6PBNK.gfortran-win32.dll
ª   ª   ª       ª   ª       libwrap_dum.WPQ6YJENJJZVNFDRKLGZ3AANPKT45Z5L.gfortran-win32.dll
ª   ª   ª       ª   ª       lib_arpack-.XOZWXTEQHT3V7JMZQU6XWV3TIPTK67DD.gfortran-win32.dll
ª   ª   ª       ª   ª       lib_blas_su.52PVEJ2I4FB5JTDBFQXX6WEQ62TNNZED.gfortran-win32.dll
ª   ª   ª       ª   ª       lib_cpropac.BGCDHXLSITJXHYDTF7A2PPEMJW3JYKLF.gfortran-win32.dll
ª   ª   ª       ª   ª       lib_dop-f2p.THTKKRZYXIU6GZK7RWPJSWMKPXS5IWYC.gfortran-win32.dll
ª   ª   ª       ª   ª       lib_dpropac.IB6LQ5YAQODV5EVRQBB7XPEUEXP57BT5.gfortran-win32.dll
ª   ª   ª       ª   ª       lib_lsoda-f.H7WHD7HLB4HQNB2MKWWK5UW35RGZSIEH.gfortran-win32.dll
ª   ª   ª       ª   ª       lib_spropac.QXNBOTKII4PVUJ63XJ3GMQJ3IRGVDYJF.gfortran-win32.dll
ª   ª   ª       ª   ª       lib_test_fo.W4FAHPU7B3LC6AUMSCCF6IZYIYKJAMXJ.gfortran-win32.dll
ª   ª   ª       ª   ª       lib_vode-f2.KRCAOOXXFYOZEOPNR3A7J3A2RRCCFULB.gfortran-win32.dll
ª   ª   ª       ª   ª       lib_zpropac.DI6O24FW7F7PFJ4R2PZ3GG35DKLOUMUN.gfortran-win32.dll
ª   ª   ª       ª   ª       msvcp140.dll
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---cluster
ª   ª   ª       ª   ª   ª   hierarchy.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   vq.py
ª   ª   ª       ª   ª   ª   _hierarchy.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _optimal_leaf_ordering.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _vq.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           hierarchy_test_data.py
ª   ª   ª       ª   ª           meson.build
ª   ª   ª       ª   ª           test_disjoint_set.py
ª   ª   ª       ª   ª           test_hierarchy.py
ª   ª   ª       ª   ª           test_vq.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---constants
ª   ª   ª       ª   ª   ª   codata.py
ª   ª   ª       ª   ª   ª   constants.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _codata.py
ª   ª   ª       ª   ª   ª   _constants.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           meson.build
ª   ª   ª       ª   ª           test_codata.py
ª   ª   ª       ª   ª           test_constants.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---fft
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _backend.py
ª   ª   ª       ª   ª   ª   _basic.py
ª   ª   ª       ª   ª   ª   _debug_backends.py
ª   ª   ª       ª   ª   ª   _fftlog.py
ª   ª   ª       ª   ª   ª   _fftlog_multimethods.py
ª   ª   ª       ª   ª   ª   _helper.py
ª   ª   ª       ª   ª   ª   _realtransforms.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª   ª       meson.build
ª   ª   ª       ª   ª   ª       mock_backend.py
ª   ª   ª       ª   ª   ª       test_backend.py
ª   ª   ª       ª   ª   ª       test_fftlog.py
ª   ª   ª       ª   ª   ª       test_fft_function.py
ª   ª   ª       ª   ª   ª       test_helper.py
ª   ª   ª       ª   ª   ª       test_multithreading.py
ª   ª   ª       ª   ª   ª       test_numpy.py
ª   ª   ª       ª   ª   ª       test_real_transforms.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_pocketfft
ª   ª   ª       ª   ª       ª   basic.py
ª   ª   ª       ª   ª       ª   helper.py
ª   ª   ª       ª   ª       ª   LICENSE.md
ª   ª   ª       ª   ª       ª   pypocketfft.cp39-win32.pyd
ª   ª   ª       ª   ª       ª   realtransforms.py
ª   ª   ª       ª   ª       ª   setup.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---tests
ª   ª   ª       ª   ª               meson.build
ª   ª   ª       ª   ª               test_basic.py
ª   ª   ª       ª   ª               test_real_transforms.py
ª   ª   ª       ª   ª               __init__.py
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---fftpack
ª   ª   ª       ª   ª   ª   basic.py
ª   ª   ª       ª   ª   ª   convolve.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   helper.py
ª   ª   ª       ª   ª   ª   pseudo_diffs.py
ª   ª   ª       ª   ª   ª   realtransforms.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _basic.py
ª   ª   ª       ª   ª   ª   _helper.py
ª   ª   ª       ª   ª   ª   _pseudo_diffs.py
ª   ª   ª       ª   ª   ª   _realtransforms.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           fftw_dct.c
ª   ª   ª       ª   ª           fftw_double_ref.npz
ª   ª   ª       ª   ª           fftw_longdouble_ref.npz
ª   ª   ª       ª   ª           fftw_single_ref.npz
ª   ª   ª       ª   ª           gendata.m
ª   ª   ª       ª   ª           gendata.py
ª   ª   ª       ª   ª           gen_fftw_ref.py
ª   ª   ª       ª   ª           Makefile
ª   ª   ª       ª   ª           meson.build
ª   ª   ª       ª   ª           test.npz
ª   ª   ª       ª   ª           test_basic.py
ª   ª   ª       ª   ª           test_helper.py
ª   ª   ª       ª   ª           test_import.py
ª   ª   ª       ª   ª           test_pseudo_diffs.py
ª   ª   ª       ª   ª           test_real_transforms.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---integrate
ª   ª   ª       ª   ª   ª   dop.py
ª   ª   ª       ª   ª   ª   lsoda.py
ª   ª   ª       ª   ª   ª   odepack.py
ª   ª   ª       ª   ª   ª   quadpack.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   vode.py
ª   ª   ª       ª   ª   ª   _bvp.py
ª   ª   ª       ª   ª   ª   _dop.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _lsoda.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _ode.py
ª   ª   ª       ª   ª   ª   _odepack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _odepack_py.py
ª   ª   ª       ª   ª   ª   _quadpack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _quadpack_py.py
ª   ª   ª       ª   ª   ª   _quadrature.py
ª   ª   ª       ª   ª   ª   _quad_vec.py
ª   ª   ª       ª   ª   ª   _test_multivariate.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _test_odeint_banded.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _vode.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª   ª       banded5x5.f
ª   ª   ª       ª   ª   ª       banded5x5.pyf
ª   ª   ª       ª   ª   ª       meson.build
ª   ª   ª       ª   ª   ª       test_banded_ode_solvers.py
ª   ª   ª       ª   ª   ª       test_bvp.py
ª   ª   ª       ª   ª   ª       test_integrate.py
ª   ª   ª       ª   ª   ª       test_odeint_jac.py
ª   ª   ª       ª   ª   ª       test_quadpack.py
ª   ª   ª       ª   ª   ª       test_quadrature.py
ª   ª   ª       ª   ª   ª       test__quad_vec.py
ª   ª   ª       ª   ª   ª       _test_multivariate.c
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_ivp
ª   ª   ª       ª   ª       ª   base.py
ª   ª   ª       ª   ª       ª   bdf.py
ª   ª   ª       ª   ª       ª   common.py
ª   ª   ª       ª   ª       ª   dop853_coefficients.py
ª   ª   ª       ª   ª       ª   ivp.py
ª   ª   ª       ª   ª       ª   lsoda.py
ª   ª   ª       ª   ª       ª   radau.py
ª   ª   ª       ª   ª       ª   rk.py
ª   ª   ª       ª   ª       ª   setup.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---tests
ª   ª   ª       ª   ª               meson.build
ª   ª   ª       ª   ª               test_ivp.py
ª   ª   ª       ª   ª               test_rk.py
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---interpolate
ª   ª   ª       ª   ª   ª   dfitpack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   fitpack.py
ª   ª   ª       ª   ª   ª   fitpack2.py
ª   ª   ª       ª   ª   ª   interpnd.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   interpolate.py
ª   ª   ª       ª   ª   ª   ndgriddata.py
ª   ª   ª       ª   ª   ª   polyint.py
ª   ª   ª       ª   ª   ª   rbf.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _bspl.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _bsplines.py
ª   ª   ª       ª   ª   ª   _cubic.py
ª   ª   ª       ª   ª   ª   _fitpack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _fitpack2.py
ª   ª   ª       ª   ª   ª   _fitpack_impl.py
ª   ª   ª       ª   ª   ª   _fitpack_py.py
ª   ª   ª       ª   ª   ª   _interpnd_info.py
ª   ª   ª       ª   ª   ª   _interpolate.py
ª   ª   ª       ª   ª   ª   _ndgriddata.py
ª   ª   ª       ª   ª   ª   _pade.py
ª   ª   ª       ª   ª   ª   _polyint.py
ª   ª   ª       ª   ª   ª   _ppoly.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _rbf.py
ª   ª   ª       ª   ª   ª   _rbfinterp.py
ª   ª   ª       ª   ª   ª   _rbfinterp_pythran.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _rbfinterp_pythran.py
ª   ª   ª       ª   ª   ª   _rgi.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª       ª   meson.build
ª   ª   ª       ª   ª       ª   test_bsplines.py
ª   ª   ª       ª   ª       ª   test_fitpack.py
ª   ª   ª       ª   ª       ª   test_fitpack2.py
ª   ª   ª       ª   ª       ª   test_gil.py
ª   ª   ª       ª   ª       ª   test_interpnd.py
ª   ª   ª       ª   ª       ª   test_interpolate.py
ª   ª   ª       ª   ª       ª   test_ndgriddata.py
ª   ª   ª       ª   ª       ª   test_pade.py
ª   ª   ª       ª   ª       ª   test_polyint.py
ª   ª   ª       ª   ª       ª   test_rbf.py
ª   ª   ª       ª   ª       ª   test_rbfinterp.py
ª   ª   ª       ª   ª       ª   test_rgi.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---data
ª   ª   ª       ª   ª               bug-1310.npz
ª   ª   ª       ª   ª               estimate_gradients_hang.npy
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---io
ª   ª   ª       ª   ª   ª   harwell_boeing.py
ª   ª   ª       ª   ª   ª   idl.py
ª   ª   ª       ª   ª   ª   mmio.py
ª   ª   ª       ª   ª   ª   netcdf.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   wavfile.py
ª   ª   ª       ª   ª   ª   _fortran.py
ª   ª   ª       ª   ª   ª   _idl.py
ª   ª   ª       ª   ª   ª   _mmio.py
ª   ª   ª       ª   ª   ª   _netcdf.py
ª   ª   ª       ª   ª   ª   _test_fortran.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---arff
ª   ª   ª       ª   ª   ª   ª   arffread.py
ª   ª   ª       ª   ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   ª   _arffread.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---tests
ª   ª   ª       ª   ª   ª       ª   meson.build
ª   ª   ª       ª   ª   ª       ª   test_arffread.py
ª   ª   ª       ª   ª   ª       ª   __init__.py
ª   ª   ª       ª   ª   ª       ª   
ª   ª   ª       ª   ª   ª       +---data
ª   ª   ª       ª   ª   ª               iris.arff
ª   ª   ª       ª   ª   ª               missing.arff
ª   ª   ª       ª   ª   ª               nodata.arff
ª   ª   ª       ª   ª   ª               quoted_nominal.arff
ª   ª   ª       ª   ª   ª               quoted_nominal_spaces.arff
ª   ª   ª       ª   ª   ª               test1.arff
ª   ª   ª       ª   ª   ª               test10.arff
ª   ª   ª       ª   ª   ª               test11.arff
ª   ª   ª       ª   ª   ª               test2.arff
ª   ª   ª       ª   ª   ª               test3.arff
ª   ª   ª       ª   ª   ª               test4.arff
ª   ª   ª       ª   ª   ª               test5.arff
ª   ª   ª       ª   ª   ª               test6.arff
ª   ª   ª       ª   ª   ª               test7.arff
ª   ª   ª       ª   ª   ª               test8.arff
ª   ª   ª       ª   ª   ª               test9.arff
ª   ª   ª       ª   ª   ª               
ª   ª   ª       ª   ª   +---matlab
ª   ª   ª       ª   ª   ª   ª   byteordercodes.py
ª   ª   ª       ª   ª   ª   ª   mio.py
ª   ª   ª       ª   ª   ª   ª   mio4.py
ª   ª   ª       ª   ª   ª   ª   mio5.py
ª   ª   ª       ª   ª   ª   ª   mio5_params.py
ª   ª   ª       ª   ª   ª   ª   mio5_utils.py
ª   ª   ª       ª   ª   ª   ª   miobase.py
ª   ª   ª       ª   ª   ª   ª   mio_utils.py
ª   ª   ª       ª   ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   ª   streams.py
ª   ª   ª       ª   ª   ª   ª   _byteordercodes.py
ª   ª   ª       ª   ª   ª   ª   _mio.py
ª   ª   ª       ª   ª   ª   ª   _mio4.py
ª   ª   ª       ª   ª   ª   ª   _mio5.py
ª   ª   ª       ª   ª   ª   ª   _mio5_params.py
ª   ª   ª       ª   ª   ª   ª   _mio5_utils.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   _miobase.py
ª   ª   ª       ª   ª   ª   ª   _mio_utils.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   _streams.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---tests
ª   ª   ª       ª   ª   ª       ª   afunc.m
ª   ª   ª       ª   ª   ª       ª   gen_mat4files.m
ª   ª   ª       ª   ª   ª       ª   gen_mat5files.m
ª   ª   ª       ª   ª   ª       ª   meson.build
ª   ª   ª       ª   ª   ª       ª   save_matfile.m
ª   ª   ª       ª   ª   ª       ª   test_byteordercodes.py
ª   ª   ª       ª   ª   ª       ª   test_mio.py
ª   ª   ª       ª   ª   ª       ª   test_mio5_utils.py
ª   ª   ª       ª   ª   ª       ª   test_miobase.py
ª   ª   ª       ª   ª   ª       ª   test_mio_funcs.py
ª   ª   ª       ª   ª   ª       ª   test_mio_utils.py
ª   ª   ª       ª   ª   ª       ª   test_pathological.py
ª   ª   ª       ª   ª   ª       ª   test_streams.py
ª   ª   ª       ª   ª   ª       ª   __init__.py
ª   ª   ª       ª   ª   ª       ª   
ª   ª   ª       ª   ª   ª       +---data
ª   ª   ª       ª   ª   ª               bad_miuint32.mat
ª   ª   ª       ª   ª   ª               bad_miutf8_array_name.mat
ª   ª   ª       ª   ª   ª               big_endian.mat
ª   ª   ª       ª   ª   ª               broken_utf8.mat
ª   ª   ª       ª   ª   ª               corrupted_zlib_checksum.mat
ª   ª   ª       ª   ª   ª               corrupted_zlib_data.mat
ª   ª   ª       ª   ª   ª               japanese_utf8.txt
ª   ª   ª       ª   ª   ª               little_endian.mat
ª   ª   ª       ª   ª   ª               logical_sparse.mat
ª   ª   ª       ª   ª   ª               malformed1.mat
ª   ª   ª       ª   ª   ª               miuint32_for_miint32.mat
ª   ª   ª       ª   ª   ª               miutf8_array_name.mat
ª   ª   ª       ª   ª   ª               nasty_duplicate_fieldnames.mat
ª   ª   ª       ª   ª   ª               one_by_zero_char.mat
ª   ª   ª       ª   ª   ª               parabola.mat
ª   ª   ª       ª   ª   ª               single_empty_string.mat
ª   ª   ª       ª   ª   ª               some_functions.mat
ª   ª   ª       ª   ª   ª               sqr.mat
ª   ª   ª       ª   ª   ª               test3dmatrix_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               test3dmatrix_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               test3dmatrix_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               test3dmatrix_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testbool_8_WIN64.mat
ª   ª   ª       ª   ª   ª               testcellnest_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               testcellnest_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testcellnest_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testcellnest_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testcell_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               testcell_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testcell_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testcell_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testcomplex_4.2c_SOL2.mat
ª   ª   ª       ª   ª   ª               testcomplex_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               testcomplex_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testcomplex_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testcomplex_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testdouble_4.2c_SOL2.mat
ª   ª   ª       ª   ª   ª               testdouble_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               testdouble_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testdouble_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testdouble_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testemptycell_5.3_SOL2.mat
ª   ª   ª       ª   ª   ª               testemptycell_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testemptycell_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testemptycell_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testfunc_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testhdf5_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testmatrix_4.2c_SOL2.mat
ª   ª   ª       ª   ª   ª               testmatrix_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               testmatrix_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testmatrix_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testmatrix_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testminus_4.2c_SOL2.mat
ª   ª   ª       ª   ª   ª               testminus_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               testminus_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testminus_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testminus_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testmulti_4.2c_SOL2.mat
ª   ª   ª       ª   ª   ª               testmulti_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testmulti_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testobject_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               testobject_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testobject_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testobject_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testonechar_4.2c_SOL2.mat
ª   ª   ª       ª   ª   ª               testonechar_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               testonechar_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testonechar_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testonechar_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testscalarcell_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testsimplecell.mat
ª   ª   ª       ª   ª   ª               testsparsecomplex_4.2c_SOL2.mat
ª   ª   ª       ª   ª   ª               testsparsecomplex_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               testsparsecomplex_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testsparsecomplex_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testsparsecomplex_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testsparsefloat_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testsparse_4.2c_SOL2.mat
ª   ª   ª       ª   ª   ª               testsparse_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               testsparse_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testsparse_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testsparse_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststringarray_4.2c_SOL2.mat
ª   ª   ª       ª   ª   ª               teststringarray_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               teststringarray_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststringarray_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststringarray_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststring_4.2c_SOL2.mat
ª   ª   ª       ª   ª   ª               teststring_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               teststring_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststring_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststring_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststructarr_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               teststructarr_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststructarr_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststructarr_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststructnest_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               teststructnest_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststructnest_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststructnest_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststruct_6.1_SOL2.mat
ª   ª   ª       ª   ª   ª               teststruct_6.5.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststruct_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               teststruct_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testunicode_7.1_GLNX86.mat
ª   ª   ª       ª   ª   ª               testunicode_7.4_GLNX86.mat
ª   ª   ª       ª   ª   ª               testvec_4_GLNX86.mat
ª   ª   ª       ª   ª   ª               test_empty_struct.mat
ª   ª   ª       ª   ª   ª               test_mat4_le_floats.mat
ª   ª   ª       ª   ª   ª               test_skip_variable.mat
ª   ª   ª       ª   ª   ª               
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª   ª   ª   meson.build
ª   ª   ª       ª   ª   ª   ª   test_fortran.py
ª   ª   ª       ª   ª   ª   ª   test_idl.py
ª   ª   ª       ª   ª   ª   ª   test_mmio.py
ª   ª   ª       ª   ª   ª   ª   test_netcdf.py
ª   ª   ª       ª   ª   ª   ª   test_paths.py
ª   ª   ª       ª   ª   ª   ª   test_wavfile.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---data
ª   ª   ª       ª   ª   ª           array_float32_1d.sav
ª   ª   ª       ª   ª   ª           array_float32_2d.sav
ª   ª   ª       ª   ª   ª           array_float32_3d.sav
ª   ª   ª       ª   ª   ª           array_float32_4d.sav
ª   ª   ª       ª   ª   ª           array_float32_5d.sav
ª   ª   ª       ª   ª   ª           array_float32_6d.sav
ª   ª   ª       ª   ª   ª           array_float32_7d.sav
ª   ª   ª       ª   ª   ª           array_float32_8d.sav
ª   ª   ª       ª   ª   ª           array_float32_pointer_1d.sav
ª   ª   ª       ª   ª   ª           array_float32_pointer_2d.sav
ª   ª   ª       ª   ª   ª           array_float32_pointer_3d.sav
ª   ª   ª       ª   ª   ª           array_float32_pointer_4d.sav
ª   ª   ª       ª   ª   ª           array_float32_pointer_5d.sav
ª   ª   ª       ª   ª   ª           array_float32_pointer_6d.sav
ª   ª   ª       ª   ª   ª           array_float32_pointer_7d.sav
ª   ª   ª       ª   ª   ª           array_float32_pointer_8d.sav
ª   ª   ª       ª   ª   ª           example_1.nc
ª   ª   ª       ª   ª   ª           example_2.nc
ª   ª   ª       ª   ª   ª           example_3_maskedvals.nc
ª   ª   ª       ª   ª   ª           fortran-3x3d-2i.dat
ª   ª   ª       ª   ª   ª           fortran-mixed.dat
ª   ª   ª       ª   ª   ª           fortran-sf8-11x1x10.dat
ª   ª   ª       ª   ª   ª           fortran-sf8-15x10x22.dat
ª   ª   ª       ª   ª   ª           fortran-sf8-1x1x1.dat
ª   ª   ª       ª   ª   ª           fortran-sf8-1x1x5.dat
ª   ª   ª       ª   ª   ª           fortran-sf8-1x1x7.dat
ª   ª   ª       ª   ª   ª           fortran-sf8-1x3x5.dat
ª   ª   ª       ª   ª   ª           fortran-si4-11x1x10.dat
ª   ª   ª       ª   ª   ª           fortran-si4-15x10x22.dat
ª   ª   ª       ª   ª   ª           fortran-si4-1x1x1.dat
ª   ª   ª       ª   ª   ª           fortran-si4-1x1x5.dat
ª   ª   ª       ª   ª   ª           fortran-si4-1x1x7.dat
ª   ª   ª       ª   ª   ª           fortran-si4-1x3x5.dat
ª   ª   ª       ª   ª   ª           invalid_pointer.sav
ª   ª   ª       ª   ª   ª           null_pointer.sav
ª   ª   ª       ª   ª   ª           scalar_byte.sav
ª   ª   ª       ª   ª   ª           scalar_byte_descr.sav
ª   ª   ª       ª   ª   ª           scalar_complex32.sav
ª   ª   ª       ª   ª   ª           scalar_complex64.sav
ª   ª   ª       ª   ª   ª           scalar_float32.sav
ª   ª   ª       ª   ª   ª           scalar_float64.sav
ª   ª   ª       ª   ª   ª           scalar_heap_pointer.sav
ª   ª   ª       ª   ª   ª           scalar_int16.sav
ª   ª   ª       ª   ª   ª           scalar_int32.sav
ª   ª   ª       ª   ª   ª           scalar_int64.sav
ª   ª   ª       ª   ª   ª           scalar_string.sav
ª   ª   ª       ª   ª   ª           scalar_uint16.sav
ª   ª   ª       ª   ª   ª           scalar_uint32.sav
ª   ª   ª       ª   ª   ª           scalar_uint64.sav
ª   ª   ª       ª   ª   ª           struct_arrays.sav
ª   ª   ª       ª   ª   ª           struct_arrays_byte_idl80.sav
ª   ª   ª       ª   ª   ª           struct_arrays_replicated.sav
ª   ª   ª       ª   ª   ª           struct_arrays_replicated_3d.sav
ª   ª   ª       ª   ª   ª           struct_inherit.sav
ª   ª   ª       ª   ª   ª           struct_pointers.sav
ª   ª   ª       ª   ª   ª           struct_pointers_replicated.sav
ª   ª   ª       ª   ª   ª           struct_pointers_replicated_3d.sav
ª   ª   ª       ª   ª   ª           struct_pointer_arrays.sav
ª   ª   ª       ª   ª   ª           struct_pointer_arrays_replicated.sav
ª   ª   ª       ª   ª   ª           struct_pointer_arrays_replicated_3d.sav
ª   ª   ª       ª   ª   ª           struct_scalars.sav
ª   ª   ª       ª   ª   ª           struct_scalars_replicated.sav
ª   ª   ª       ª   ª   ª           struct_scalars_replicated_3d.sav
ª   ª   ª       ª   ª   ª           test-44100Hz-2ch-32bit-float-be.wav
ª   ª   ª       ª   ª   ª           test-44100Hz-2ch-32bit-float-le.wav
ª   ª   ª       ª   ª   ª           test-44100Hz-be-1ch-4bytes.wav
ª   ª   ª       ª   ª   ª           test-44100Hz-le-1ch-4bytes-early-eof-no-data.wav
ª   ª   ª       ª   ª   ª           test-44100Hz-le-1ch-4bytes-early-eof.wav
ª   ª   ª       ª   ª   ª           test-44100Hz-le-1ch-4bytes-incomplete-chunk.wav
ª   ª   ª       ª   ª   ª           test-44100Hz-le-1ch-4bytes.wav
ª   ª   ª       ª   ª   ª           test-48000Hz-2ch-64bit-float-le-wavex.wav
ª   ª   ª       ª   ª   ª           test-8000Hz-be-3ch-5S-24bit.wav
ª   ª   ª       ª   ª   ª           test-8000Hz-le-1ch-10S-20bit-extra.wav
ª   ª   ª       ª   ª   ª           test-8000Hz-le-1ch-1byte-ulaw.wav
ª   ª   ª       ª   ª   ª           test-8000Hz-le-2ch-1byteu.wav
ª   ª   ª       ª   ª   ª           test-8000Hz-le-3ch-5S-24bit-inconsistent.wav
ª   ª   ª       ª   ª   ª           test-8000Hz-le-3ch-5S-24bit.wav
ª   ª   ª       ª   ª   ª           test-8000Hz-le-3ch-5S-36bit.wav
ª   ª   ª       ª   ª   ª           test-8000Hz-le-3ch-5S-45bit.wav
ª   ª   ª       ª   ª   ª           test-8000Hz-le-3ch-5S-53bit.wav
ª   ª   ª       ª   ª   ª           test-8000Hz-le-3ch-5S-64bit.wav
ª   ª   ª       ª   ª   ª           test-8000Hz-le-4ch-9S-12bit.wav
ª   ª   ª       ª   ª   ª           test-8000Hz-le-5ch-9S-5bit.wav
ª   ª   ª       ª   ª   ª           Transparent Busy.ani
ª   ª   ª       ª   ª   ª           various_compressed.sav
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---_harwell_boeing
ª   ª   ª       ª   ª       ª   hb.py
ª   ª   ª       ª   ª       ª   setup.py
ª   ª   ª       ª   ª       ª   _fortran_format_parser.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---tests
ª   ª   ª       ª   ª               meson.build
ª   ª   ª       ª   ª               test_fortran_format.py
ª   ª   ª       ª   ª               test_hb.py
ª   ª   ª       ª   ª               __init__.py
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---linalg
ª   ª   ª       ª   ª   ª   basic.py
ª   ª   ª       ª   ª   ª   blas.py
ª   ª   ª       ª   ª   ª   cython_blas.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   cython_blas.pxd
ª   ª   ª       ª   ª   ª   cython_lapack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   cython_lapack.pxd
ª   ª   ª       ª   ª   ª   decomp.py
ª   ª   ª       ª   ª   ª   decomp_cholesky.py
ª   ª   ª       ª   ª   ª   decomp_lu.py
ª   ª   ª       ª   ª   ª   decomp_qr.py
ª   ª   ª       ª   ª   ª   decomp_schur.py
ª   ª   ª       ª   ª   ª   decomp_svd.py
ª   ª   ª       ª   ª   ª   flinalg.py
ª   ª   ª       ª   ª   ª   interpolative.py
ª   ª   ª       ª   ª   ª   lapack.py
ª   ª   ª       ª   ª   ª   matfuncs.py
ª   ª   ª       ª   ª   ª   misc.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   special_matrices.py
ª   ª   ª       ª   ª   ª   _basic.py
ª   ª   ª       ª   ª   ª   _cythonized_array_utils.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _cythonized_array_utils.pxd
ª   ª   ª       ª   ª   ª   _cythonized_array_utils.pyi
ª   ª   ª       ª   ª   ª   _cython_signature_generator.py
ª   ª   ª       ª   ª   ª   _decomp.py
ª   ª   ª       ª   ª   ª   _decomp_cholesky.py
ª   ª   ª       ª   ª   ª   _decomp_cossin.py
ª   ª   ª       ª   ª   ª   _decomp_ldl.py
ª   ª   ª       ª   ª   ª   _decomp_lu.py
ª   ª   ª       ª   ª   ª   _decomp_polar.py
ª   ª   ª       ª   ª   ª   _decomp_qr.py
ª   ª   ª       ª   ª   ª   _decomp_qz.py
ª   ª   ª       ª   ª   ª   _decomp_schur.py
ª   ª   ª       ª   ª   ª   _decomp_svd.py
ª   ª   ª       ª   ª   ª   _decomp_update.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _expm_frechet.py
ª   ª   ª       ª   ª   ª   _fblas.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _flapack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _flinalg.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _flinalg_py.py
ª   ª   ª       ª   ª   ª   _generate_pyx.py
ª   ª   ª       ª   ª   ª   _interpolative.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _interpolative_backend.py
ª   ª   ª       ª   ª   ª   _matfuncs.py
ª   ª   ª       ª   ª   ª   _matfuncs_expm.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _matfuncs_expm.pyi
ª   ª   ª       ª   ª   ª   _matfuncs_inv_ssq.py
ª   ª   ª       ª   ª   ª   _matfuncs_sqrtm.py
ª   ª   ª       ª   ª   ª   _matfuncs_sqrtm_triu.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _matfuncs_sqrtm_triu.py
ª   ª   ª       ª   ª   ª   _misc.py
ª   ª   ª       ª   ª   ª   _procrustes.py
ª   ª   ª       ª   ª   ª   _sketches.py
ª   ª   ª       ª   ª   ª   _solvers.py
ª   ª   ª       ª   ª   ª   _solve_toeplitz.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _special_matrices.py
ª   ª   ª       ª   ª   ª   _testutils.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---src
ª   ª   ª       ª   ª   ª   +---id_dist
ª   ª   ª       ª   ª   ª       +---doc
ª   ª   ª       ª   ª   ª               doc.tex
ª   ª   ª       ª   ª   ª               
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª       ª   meson.build
ª   ª   ª       ª   ª       ª   test_basic.py
ª   ª   ª       ª   ª       ª   test_blas.py
ª   ª   ª       ª   ª       ª   test_cythonized_array_utils.py
ª   ª   ª       ª   ª       ª   test_cython_blas.py
ª   ª   ª       ª   ª       ª   test_cython_lapack.py
ª   ª   ª       ª   ª       ª   test_decomp.py
ª   ª   ª       ª   ª       ª   test_decomp_cholesky.py
ª   ª   ª       ª   ª       ª   test_decomp_cossin.py
ª   ª   ª       ª   ª       ª   test_decomp_ldl.py
ª   ª   ª       ª   ª       ª   test_decomp_polar.py
ª   ª   ª       ª   ª       ª   test_decomp_update.py
ª   ª   ª       ª   ª       ª   test_fblas.py
ª   ª   ª       ª   ª       ª   test_interpolative.py
ª   ª   ª       ª   ª       ª   test_lapack.py
ª   ª   ª       ª   ª       ª   test_matfuncs.py
ª   ª   ª       ª   ª       ª   test_matmul_toeplitz.py
ª   ª   ª       ª   ª       ª   test_misc.py
ª   ª   ª       ª   ª       ª   test_procrustes.py
ª   ª   ª       ª   ª       ª   test_sketches.py
ª   ª   ª       ª   ª       ª   test_solvers.py
ª   ª   ª       ª   ª       ª   test_solve_toeplitz.py
ª   ª   ª       ª   ª       ª   test_special_matrices.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---data
ª   ª   ª       ª   ª               carex_15_data.npz
ª   ª   ª       ª   ª               carex_18_data.npz
ª   ª   ª       ª   ª               carex_19_data.npz
ª   ª   ª       ª   ª               carex_20_data.npz
ª   ª   ª       ª   ª               carex_6_data.npz
ª   ª   ª       ª   ª               gendare_20170120_data.npz
ª   ª   ª       ª   ª               meson.build
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---misc
ª   ª   ª       ª   ª   ª   ascent.dat
ª   ª   ª       ª   ª   ª   common.py
ª   ª   ª       ª   ª   ª   doccer.py
ª   ª   ª       ª   ª   ª   ecg.dat
ª   ª   ª       ª   ª   ª   face.dat
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _common.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           meson.build
ª   ª   ª       ª   ª           test_common.py
ª   ª   ª       ª   ª           test_doccer.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---ndimage
ª   ª   ª       ª   ª   ª   filters.py
ª   ª   ª       ª   ª   ª   fourier.py
ª   ª   ª       ª   ª   ª   interpolation.py
ª   ª   ª       ª   ª   ª   measurements.py
ª   ª   ª       ª   ª   ª   morphology.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _ctest.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _cytest.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _filters.py
ª   ª   ª       ª   ª   ª   _fourier.py
ª   ª   ª       ª   ª   ª   _interpolation.py
ª   ª   ª       ª   ª   ª   _measurements.py
ª   ª   ª       ª   ª   ª   _morphology.py
ª   ª   ª       ª   ª   ª   _nd_image.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _ni_docstrings.py
ª   ª   ª       ª   ª   ª   _ni_label.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _ni_support.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª       ª   dots.png
ª   ª   ª       ª   ª       ª   meson.build
ª   ª   ª       ª   ª       ª   test_c_api.py
ª   ª   ª       ª   ª       ª   test_datatypes.py
ª   ª   ª       ª   ª       ª   test_filters.py
ª   ª   ª       ª   ª       ª   test_fourier.py
ª   ª   ª       ª   ª       ª   test_interpolation.py
ª   ª   ª       ª   ª       ª   test_measurements.py
ª   ª   ª       ª   ª       ª   test_morphology.py
ª   ª   ª       ª   ª       ª   test_splines.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---data
ª   ª   ª       ª   ª               label_inputs.txt
ª   ª   ª       ª   ª               label_results.txt
ª   ª   ª       ª   ª               label_strels.txt
ª   ª   ª       ª   ª               README.txt
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---odr
ª   ª   ª       ª   ª   ª   models.py
ª   ª   ª       ª   ª   ª   odrpack.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _add_newdocs.py
ª   ª   ª       ª   ª   ª   _models.py
ª   ª   ª       ª   ª   ª   _odrpack.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __odrpack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           meson.build
ª   ª   ª       ª   ª           test_odr.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---optimize
ª   ª   ª       ª   ª   ª   cobyla.py
ª   ª   ª       ª   ª   ª   cython_optimize.pxd
ª   ª   ª       ª   ª   ª   lbfgsb.py
ª   ª   ª       ª   ª   ª   linesearch.py
ª   ª   ª       ª   ª   ª   minpack.py
ª   ª   ª       ª   ª   ª   minpack2.py
ª   ª   ª       ª   ª   ª   moduleTNC.py
ª   ª   ª       ª   ª   ª   nonlin.py
ª   ª   ª       ª   ª   ª   optimize.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   slsqp.py
ª   ª   ª       ª   ª   ª   tnc.py
ª   ª   ª       ª   ª   ª   zeros.py
ª   ª   ª       ª   ª   ª   _basinhopping.py
ª   ª   ª       ª   ª   ª   _bglu_dense.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _cobyla.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _cobyla_py.py
ª   ª   ª       ª   ª   ª   _constraints.py
ª   ª   ª       ª   ª   ª   _differentiable_functions.py
ª   ª   ª       ª   ª   ª   _differentialevolution.py
ª   ª   ª       ª   ª   ª   _direct.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _direct_py.py
ª   ª   ª       ª   ª   ª   _dual_annealing.py
ª   ª   ª       ª   ª   ª   _group_columns.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _group_columns.py
ª   ª   ª       ª   ª   ª   _hessian_update_strategy.py
ª   ª   ª       ª   ª   ª   _lbfgsb.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _lbfgsb_py.py
ª   ª   ª       ª   ª   ª   _linesearch.py
ª   ª   ª       ª   ª   ª   _linprog.py
ª   ª   ª       ª   ª   ª   _linprog_doc.py
ª   ª   ª       ª   ª   ª   _linprog_highs.py
ª   ª   ª       ª   ª   ª   _linprog_ip.py
ª   ª   ª       ª   ª   ª   _linprog_rs.py
ª   ª   ª       ª   ª   ª   _linprog_simplex.py
ª   ª   ª       ª   ª   ª   _linprog_util.py
ª   ª   ª       ª   ª   ª   _lsap.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _milp.py
ª   ª   ª       ª   ª   ª   _minimize.py
ª   ª   ª       ª   ª   ª   _minpack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _minpack2.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _minpack_py.py
ª   ª   ª       ª   ª   ª   _moduleTNC.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _nnls.py
ª   ª   ª       ª   ª   ª   _nonlin.py
ª   ª   ª       ª   ª   ª   _numdiff.py
ª   ª   ª       ª   ª   ª   _optimize.py
ª   ª   ª       ª   ª   ª   _qap.py
ª   ª   ª       ª   ª   ª   _remove_redundancy.py
ª   ª   ª       ª   ª   ª   _root.py
ª   ª   ª       ª   ª   ª   _root_scalar.py
ª   ª   ª       ª   ª   ª   _shgo.py
ª   ª   ª       ª   ª   ª   _slsqp.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _slsqp_py.py
ª   ª   ª       ª   ª   ª   _spectral.py
ª   ª   ª       ª   ª   ª   _tnc.py
ª   ª   ª       ª   ª   ª   _trustregion.py
ª   ª   ª       ª   ª   ª   _trustregion_dogleg.py
ª   ª   ª       ª   ª   ª   _trustregion_exact.py
ª   ª   ª       ª   ª   ª   _trustregion_krylov.py
ª   ª   ª       ª   ª   ª   _trustregion_ncg.py
ª   ª   ª       ª   ª   ª   _tstutils.py
ª   ª   ª       ª   ª   ª   _zeros.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _zeros_py.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   __nnls.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   __nnls.pyi
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---cython_optimize
ª   ª   ª       ª   ª   ª       c_zeros.pxd
ª   ª   ª       ª   ª   ª       setup.py
ª   ª   ª       ª   ª   ª       _zeros.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       _zeros.pxd
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---lbfgsb_src
ª   ª   ª       ª   ª   ª       README
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª   ª       meson.build
ª   ª   ª       ª   ª   ª       test_cobyla.py
ª   ª   ª       ª   ª   ª       test_constraints.py
ª   ª   ª       ª   ª   ª       test_constraint_conversion.py
ª   ª   ª       ª   ª   ª       test_cython_optimize.py
ª   ª   ª       ª   ª   ª       test_differentiable_functions.py
ª   ª   ª       ª   ª   ª       test_direct.py
ª   ª   ª       ª   ª   ª       test_hessian_update_strategy.py
ª   ª   ª       ª   ª   ª       test_lbfgsb_hessinv.py
ª   ª   ª       ª   ª   ª       test_lbfgsb_setulb.py
ª   ª   ª       ª   ª   ª       test_least_squares.py
ª   ª   ª       ª   ª   ª       test_linear_assignment.py
ª   ª   ª       ª   ª   ª       test_linesearch.py
ª   ª   ª       ª   ª   ª       test_linprog.py
ª   ª   ª       ª   ª   ª       test_lsq_common.py
ª   ª   ª       ª   ª   ª       test_lsq_linear.py
ª   ª   ª       ª   ª   ª       test_milp.py
ª   ª   ª       ª   ª   ª       test_minimize_constrained.py
ª   ª   ª       ª   ª   ª       test_minpack.py
ª   ª   ª       ª   ª   ª       test_nnls.py
ª   ª   ª       ª   ª   ª       test_nonlin.py
ª   ª   ª       ª   ª   ª       test_optimize.py
ª   ª   ª       ª   ª   ª       test_quadratic_assignment.py
ª   ª   ª       ª   ª   ª       test_regression.py
ª   ª   ª       ª   ª   ª       test_slsqp.py
ª   ª   ª       ª   ª   ª       test_tnc.py
ª   ª   ª       ª   ª   ª       test_trustregion.py
ª   ª   ª       ª   ª   ª       test_trustregion_exact.py
ª   ª   ª       ª   ª   ª       test_trustregion_krylov.py
ª   ª   ª       ª   ª   ª       test_zeros.py
ª   ª   ª       ª   ª   ª       test__basinhopping.py
ª   ª   ª       ª   ª   ª       test__differential_evolution.py
ª   ª   ª       ª   ª   ª       test__dual_annealing.py
ª   ª   ª       ª   ª   ª       test__linprog_clean_inputs.py
ª   ª   ª       ª   ª   ª       test__numdiff.py
ª   ª   ª       ª   ª   ª       test__remove_redundancy.py
ª   ª   ª       ª   ª   ª       test__root.py
ª   ª   ª       ª   ª   ª       test__shgo.py
ª   ª   ª       ª   ª   ª       test__spectral.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_highs
ª   ª   ª       ª   ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   ª   _highs_constants.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   _highs_wrapper.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---cython
ª   ª   ª       ª   ª   ª       +---src
ª   ª   ª       ª   ª   ª               HConst.pxd
ª   ª   ª       ª   ª   ª               Highs.pxd
ª   ª   ª       ª   ª   ª               HighsInfo.pxd
ª   ª   ª       ª   ª   ª               HighsIO.pxd
ª   ª   ª       ª   ª   ª               HighsLp.pxd
ª   ª   ª       ª   ª   ª               HighsLpUtils.pxd
ª   ª   ª       ª   ª   ª               HighsModelUtils.pxd
ª   ª   ª       ª   ª   ª               HighsOptions.pxd
ª   ª   ª       ª   ª   ª               HighsRuntimeOptions.pxd
ª   ª   ª       ª   ª   ª               HighsSparseMatrix.pxd
ª   ª   ª       ª   ª   ª               HighsStatus.pxd
ª   ª   ª       ª   ª   ª               highs_c_api.pxd
ª   ª   ª       ª   ª   ª               SimplexConst.pxd
ª   ª   ª       ª   ª   ª               
ª   ª   ª       ª   ª   +---_lsq
ª   ª   ª       ª   ª   ª       bvls.py
ª   ª   ª       ª   ª   ª       common.py
ª   ª   ª       ª   ª   ª       dogbox.py
ª   ª   ª       ª   ª   ª       givens_elimination.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       least_squares.py
ª   ª   ª       ª   ª   ª       lsq_linear.py
ª   ª   ª       ª   ª   ª       setup.py
ª   ª   ª       ª   ª   ª       trf.py
ª   ª   ª       ª   ª   ª       trf_linear.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_shgo_lib
ª   ª   ª       ª   ª   ª       meson.build
ª   ª   ª       ª   ª   ª       triangulation.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_trlib
ª   ª   ª       ª   ª   ª       setup.py
ª   ª   ª       ª   ª   ª       _trlib.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_trustregion_constr
ª   ª   ª       ª   ª       ª   canonical_constraint.py
ª   ª   ª       ª   ª       ª   equality_constrained_sqp.py
ª   ª   ª       ª   ª       ª   minimize_trustregion_constr.py
ª   ª   ª       ª   ª       ª   projections.py
ª   ª   ª       ª   ª       ª   qp_subproblem.py
ª   ª   ª       ª   ª       ª   report.py
ª   ª   ª       ª   ª       ª   setup.py
ª   ª   ª       ª   ª       ª   tr_interior_point.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---tests
ª   ª   ª       ª   ª               meson.build
ª   ª   ª       ª   ª               test_canonical_constraint.py
ª   ª   ª       ª   ª               test_projections.py
ª   ª   ª       ª   ª               test_qp_subproblem.py
ª   ª   ª       ª   ª               test_report.py
ª   ª   ª       ª   ª               __init__.py
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---signal
ª   ª   ª       ª   ª   ª   bsplines.py
ª   ª   ª       ª   ª   ª   filter_design.py
ª   ª   ª       ª   ª   ª   fir_filter_design.py
ª   ª   ª       ª   ª   ª   ltisys.py
ª   ª   ª       ª   ª   ª   lti_conversion.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   signaltools.py
ª   ª   ª       ª   ª   ª   spectral.py
ª   ª   ª       ª   ª   ª   spline.py
ª   ª   ª       ª   ª   ª   waveforms.py
ª   ª   ª       ª   ª   ª   wavelets.py
ª   ª   ª       ª   ª   ª   _arraytools.py
ª   ª   ª       ª   ª   ª   _bsplines.py
ª   ª   ª       ª   ª   ª   _czt.py
ª   ª   ª       ª   ª   ª   _filter_design.py
ª   ª   ª       ª   ª   ª   _fir_filter_design.py
ª   ª   ª       ª   ª   ª   _ltisys.py
ª   ª   ª       ª   ª   ª   _lti_conversion.py
ª   ª   ª       ª   ª   ª   _max_len_seq.py
ª   ª   ª       ª   ª   ª   _max_len_seq_inner.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _max_len_seq_inner.py
ª   ª   ª       ª   ª   ª   _peak_finding.py
ª   ª   ª       ª   ª   ª   _peak_finding_utils.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _savitzky_golay.py
ª   ª   ª       ª   ª   ª   _signaltools.py
ª   ª   ª       ª   ª   ª   _sigtools.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _sosfilt.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _spectral.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _spectral.py
ª   ª   ª       ª   ª   ª   _spectral_py.py
ª   ª   ª       ª   ª   ª   _spline.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _upfirdn.py
ª   ª   ª       ª   ª   ª   _upfirdn_apply.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _waveforms.py
ª   ª   ª       ª   ª   ª   _wavelets.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª   ª       meson.build
ª   ª   ª       ª   ª   ª       mpsig.py
ª   ª   ª       ª   ª   ª       test_array_tools.py
ª   ª   ª       ª   ª   ª       test_bsplines.py
ª   ª   ª       ª   ª   ª       test_cont2discrete.py
ª   ª   ª       ª   ª   ª       test_czt.py
ª   ª   ª       ª   ª   ª       test_dltisys.py
ª   ª   ª       ª   ª   ª       test_filter_design.py
ª   ª   ª       ª   ª   ª       test_fir_filter_design.py
ª   ª   ª       ª   ª   ª       test_ltisys.py
ª   ª   ª       ª   ª   ª       test_max_len_seq.py
ª   ª   ª       ª   ª   ª       test_peak_finding.py
ª   ª   ª       ª   ª   ª       test_result_type.py
ª   ª   ª       ª   ª   ª       test_savitzky_golay.py
ª   ª   ª       ª   ª   ª       test_signaltools.py
ª   ª   ª       ª   ª   ª       test_spectral.py
ª   ª   ª       ª   ª   ª       test_upfirdn.py
ª   ª   ª       ª   ª   ª       test_waveforms.py
ª   ª   ª       ª   ª   ª       test_wavelets.py
ª   ª   ª       ª   ª   ª       test_windows.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---windows
ª   ª   ª       ª   ª           setup.py
ª   ª   ª       ª   ª           windows.py
ª   ª   ª       ª   ª           _windows.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---sparse
ª   ª   ª       ª   ª   ª   base.py
ª   ª   ª       ª   ª   ª   bsr.py
ª   ª   ª       ª   ª   ª   compressed.py
ª   ª   ª       ª   ª   ª   construct.py
ª   ª   ª       ª   ª   ª   coo.py
ª   ª   ª       ª   ª   ª   csc.py
ª   ª   ª       ª   ª   ª   csr.py
ª   ª   ª       ª   ª   ª   data.py
ª   ª   ª       ª   ª   ª   dia.py
ª   ª   ª       ª   ª   ª   dok.py
ª   ª   ª       ª   ª   ª   extract.py
ª   ª   ª       ª   ª   ª   lil.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   sparsetools.py
ª   ª   ª       ª   ª   ª   spfuncs.py
ª   ª   ª       ª   ª   ª   sputils.py
ª   ª   ª       ª   ª   ª   _arrays.py
ª   ª   ª       ª   ª   ª   _base.py
ª   ª   ª       ª   ª   ª   _bsr.py
ª   ª   ª       ª   ª   ª   _compressed.py
ª   ª   ª       ª   ª   ª   _construct.py
ª   ª   ª       ª   ª   ª   _coo.py
ª   ª   ª       ª   ª   ª   _csc.py
ª   ª   ª       ª   ª   ª   _csparsetools.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _csr.py
ª   ª   ª       ª   ª   ª   _data.py
ª   ª   ª       ª   ª   ª   _dia.py
ª   ª   ª       ª   ª   ª   _dok.py
ª   ª   ª       ª   ª   ª   _extract.py
ª   ª   ª       ª   ª   ª   _generate_sparsetools.py
ª   ª   ª       ª   ª   ª   _index.py
ª   ª   ª       ª   ª   ª   _lil.py
ª   ª   ª       ª   ª   ª   _matrix_io.py
ª   ª   ª       ª   ª   ª   _sparsetools.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _spfuncs.py
ª   ª   ª       ª   ª   ª   _sputils.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---csgraph
ª   ª   ª       ª   ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   ª   _flow.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   _laplacian.py
ª   ª   ª       ª   ª   ª   ª   _matching.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   _min_spanning_tree.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   _reordering.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   _shortest_path.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   _tools.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   _traversal.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   _validation.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---tests
ª   ª   ª       ª   ª   ª           meson.build
ª   ª   ª       ª   ª   ª           test_connected_components.py
ª   ª   ª       ª   ª   ª           test_conversions.py
ª   ª   ª       ª   ª   ª           test_flow.py
ª   ª   ª       ª   ª   ª           test_graph_laplacian.py
ª   ª   ª       ª   ª   ª           test_matching.py
ª   ª   ª       ª   ª   ª           test_reordering.py
ª   ª   ª       ª   ª   ª           test_shortest_path.py
ª   ª   ª       ª   ª   ª           test_spanning_tree.py
ª   ª   ª       ª   ª   ª           test_traversal.py
ª   ª   ª       ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---linalg
ª   ª   ª       ª   ª   ª   ª   dsolve.py
ª   ª   ª       ª   ª   ª   ª   eigen.py
ª   ª   ª       ª   ª   ª   ª   interface.py
ª   ª   ª       ª   ª   ª   ª   isolve.py
ª   ª   ª       ª   ª   ª   ª   matfuncs.py
ª   ª   ª       ª   ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   ª   _expm_multiply.py
ª   ª   ª       ª   ª   ª   ª   _interface.py
ª   ª   ª       ª   ª   ª   ª   _matfuncs.py
ª   ª   ª       ª   ª   ª   ª   _norm.py
ª   ª   ª       ª   ª   ª   ª   _onenormest.py
ª   ª   ª       ª   ª   ª   ª   _svdp.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---tests
ª   ª   ª       ª   ª   ª   ª       meson.build
ª   ª   ª       ª   ª   ª   ª       propack_test_data.npz
ª   ª   ª       ª   ª   ª   ª       test_expm_multiply.py
ª   ª   ª       ª   ª   ª   ª       test_interface.py
ª   ª   ª       ª   ª   ª   ª       test_matfuncs.py
ª   ª   ª       ª   ª   ª   ª       test_norm.py
ª   ª   ª       ª   ª   ª   ª       test_onenormest.py
ª   ª   ª       ª   ª   ª   ª       test_propack.py
ª   ª   ª       ª   ª   ª   ª       test_pydata_sparse.py
ª   ª   ª       ª   ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   +---_dsolve
ª   ª   ª       ª   ª   ª   ª   ª   linsolve.py
ª   ª   ª       ª   ª   ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   ª   ª   _add_newdocs.py
ª   ª   ª       ª   ª   ª   ª   ª   _superlu.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   ª   +---SuperLU
ª   ª   ª       ª   ª   ª   ª   ª       License.txt
ª   ª   ª       ª   ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   ª   +---tests
ª   ª   ª       ª   ª   ª   ª           meson.build
ª   ª   ª       ª   ª   ª   ª           test_linsolve.py
ª   ª   ª       ª   ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª   ª           
ª   ª   ª       ª   ª   ª   +---_eigen
ª   ª   ª       ª   ª   ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   ª   ª   _svds.py
ª   ª   ª       ª   ª   ª   ª   ª   _svds_doc.py
ª   ª   ª       ª   ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   ª   +---arpack
ª   ª   ª       ª   ª   ª   ª   ª   ª   arpack.py
ª   ª   ª       ª   ª   ª   ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   ª   ª   ª   _arpack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   ª   ª   +---ARPACK
ª   ª   ª       ª   ª   ª   ª   ª   ª       COPYING
ª   ª   ª       ª   ª   ª   ª   ª   ª       
ª   ª   ª       ª   ª   ª   ª   ª   +---tests
ª   ª   ª       ª   ª   ª   ª   ª           meson.build
ª   ª   ª       ª   ª   ª   ª   ª           test_arpack.py
ª   ª   ª       ª   ª   ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª   ª   ª           
ª   ª   ª       ª   ª   ª   ª   +---lobpcg
ª   ª   ª       ª   ª   ª   ª   ª   ª   lobpcg.py
ª   ª   ª       ª   ª   ª   ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   ª   ª   +---tests
ª   ª   ª       ª   ª   ª   ª   ª           meson.build
ª   ª   ª       ª   ª   ª   ª   ª           test_lobpcg.py
ª   ª   ª       ª   ª   ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª   ª   ª           
ª   ª   ª       ª   ª   ª   ª   +---tests
ª   ª   ª       ª   ª   ª   ª           meson.build
ª   ª   ª       ª   ª   ª   ª           test_svds.py
ª   ª   ª       ª   ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª   ª           
ª   ª   ª       ª   ª   ª   +---_isolve
ª   ª   ª       ª   ª   ª   ª   ª   iterative.py
ª   ª   ª       ª   ª   ª   ª   ª   lgmres.py
ª   ª   ª       ª   ª   ª   ª   ª   lsmr.py
ª   ª   ª       ª   ª   ª   ª   ª   lsqr.py
ª   ª   ª       ª   ª   ª   ª   ª   minres.py
ª   ª   ª       ª   ª   ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   ª   ª   tfqmr.py
ª   ª   ª       ª   ª   ª   ª   ª   utils.py
ª   ª   ª       ª   ª   ª   ª   ª   _gcrotmk.py
ª   ª   ª       ª   ª   ª   ª   ª   _iterative.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   ª   +---tests
ª   ª   ª       ª   ª   ª   ª           demo_lgmres.py
ª   ª   ª       ª   ª   ª   ª           meson.build
ª   ª   ª       ª   ª   ª   ª           test_gcrotmk.py
ª   ª   ª       ª   ª   ª   ª           test_iterative.py
ª   ª   ª       ª   ª   ª   ª           test_lgmres.py
ª   ª   ª       ª   ª   ª   ª           test_lsmr.py
ª   ª   ª       ª   ª   ª   ª           test_lsqr.py
ª   ª   ª       ª   ª   ª   ª           test_minres.py
ª   ª   ª       ª   ª   ª   ª           test_utils.py
ª   ª   ª       ª   ª   ª   ª           __init__.py
ª   ª   ª       ª   ª   ª   ª           
ª   ª   ª       ª   ª   ª   +---_propack
ª   ª   ª       ª   ª   ª           _cpropack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª           _dpropack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª           _spropack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª           _zpropack.cp39-win32.pyd
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª       ª   meson.build
ª   ª   ª       ª   ª       ª   test_array_api.py
ª   ª   ª       ª   ª       ª   test_base.py
ª   ª   ª       ª   ª       ª   test_construct.py
ª   ª   ª       ª   ª       ª   test_csc.py
ª   ª   ª       ª   ª       ª   test_csr.py
ª   ª   ª       ª   ª       ª   test_extract.py
ª   ª   ª       ª   ª       ª   test_matrix_io.py
ª   ª   ª       ª   ª       ª   test_sparsetools.py
ª   ª   ª       ª   ª       ª   test_spfuncs.py
ª   ª   ª       ª   ª       ª   test_sputils.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---data
ª   ª   ª       ª   ª               csc_py2.npz
ª   ª   ª       ª   ª               csc_py3.npz
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---spatial
ª   ª   ª       ª   ª   ª   ckdtree.py
ª   ª   ª       ª   ª   ª   distance.py
ª   ª   ª       ª   ª   ª   distance.pyi
ª   ª   ª       ª   ª   ª   kdtree.py
ª   ª   ª       ª   ª   ª   qhull.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   _ckdtree.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _ckdtree.pyi
ª   ª   ª       ª   ª   ª   _distance_pybind.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _distance_wrap.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _geometric_slerp.py
ª   ª   ª       ª   ª   ª   _hausdorff.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _kdtree.py
ª   ª   ª       ª   ª   ª   _plotutils.py
ª   ª   ª       ª   ª   ª   _procrustes.py
ª   ª   ª       ª   ª   ª   _qhull.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _qhull.pyi
ª   ª   ª       ª   ª   ª   _spherical_voronoi.py
ª   ª   ª       ª   ª   ª   _voronoi.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _voronoi.pyi
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---qhull_src
ª   ª   ª       ª   ª   ª       COPYING.txt
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª   ª   ª   meson.build
ª   ª   ª       ª   ª   ª   ª   test_distance.py
ª   ª   ª       ª   ª   ª   ª   test_hausdorff.py
ª   ª   ª       ª   ª   ª   ª   test_kdtree.py
ª   ª   ª       ª   ª   ª   ª   test_qhull.py
ª   ª   ª       ª   ª   ª   ª   test_slerp.py
ª   ª   ª       ª   ª   ª   ª   test_spherical_voronoi.py
ª   ª   ª       ª   ª   ª   ª   test__plotutils.py
ª   ª   ª       ª   ª   ª   ª   test__procrustes.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---data
ª   ª   ª       ª   ª   ª           cdist-X1.txt
ª   ª   ª       ª   ª   ª           cdist-X2.txt
ª   ª   ª       ª   ª   ª           degenerate_pointset.npz
ª   ª   ª       ª   ª   ª           iris.txt
ª   ª   ª       ª   ª   ª           pdist-boolean-inp.txt
ª   ª   ª       ª   ª   ª           pdist-chebyshev-ml-iris.txt
ª   ª   ª       ª   ª   ª           pdist-chebyshev-ml.txt
ª   ª   ª       ª   ª   ª           pdist-cityblock-ml-iris.txt
ª   ª   ª       ª   ª   ª           pdist-cityblock-ml.txt
ª   ª   ª       ª   ª   ª           pdist-correlation-ml-iris.txt
ª   ª   ª       ª   ª   ª           pdist-correlation-ml.txt
ª   ª   ª       ª   ª   ª           pdist-cosine-ml-iris.txt
ª   ª   ª       ª   ª   ª           pdist-cosine-ml.txt
ª   ª   ª       ª   ª   ª           pdist-double-inp.txt
ª   ª   ª       ª   ª   ª           pdist-euclidean-ml-iris.txt
ª   ª   ª       ª   ª   ª           pdist-euclidean-ml.txt
ª   ª   ª       ª   ª   ª           pdist-hamming-ml.txt
ª   ª   ª       ª   ª   ª           pdist-jaccard-ml.txt
ª   ª   ª       ª   ª   ª           pdist-jensenshannon-ml-iris.txt
ª   ª   ª       ª   ª   ª           pdist-jensenshannon-ml.txt
ª   ª   ª       ª   ª   ª           pdist-minkowski-3.2-ml-iris.txt
ª   ª   ª       ª   ª   ª           pdist-minkowski-3.2-ml.txt
ª   ª   ª       ª   ª   ª           pdist-minkowski-5.8-ml-iris.txt
ª   ª   ª       ª   ª   ª           pdist-seuclidean-ml-iris.txt
ª   ª   ª       ª   ª   ª           pdist-seuclidean-ml.txt
ª   ª   ª       ª   ª   ª           pdist-spearman-ml.txt
ª   ª   ª       ª   ª   ª           random-bool-data.txt
ª   ª   ª       ª   ª   ª           random-double-data.txt
ª   ª   ª       ª   ª   ª           random-int-data.txt
ª   ª   ª       ª   ª   ª           random-uint-data.txt
ª   ª   ª       ª   ª   ª           selfdual-4d-polytope.txt
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---transform
ª   ª   ª       ª   ª       ª   rotation.py
ª   ª   ª       ª   ª       ª   setup.py
ª   ª   ª       ª   ª       ª   _rotation.cp39-win32.pyd
ª   ª   ª       ª   ª       ª   _rotation.pyi
ª   ª   ª       ª   ª       ª   _rotation_groups.py
ª   ª   ª       ª   ª       ª   _rotation_spline.py
ª   ª   ª       ª   ª       ª   __init__.py
ª   ª   ª       ª   ª       ª   
ª   ª   ª       ª   ª       +---tests
ª   ª   ª       ª   ª               meson.build
ª   ª   ª       ª   ª               test_rotation.py
ª   ª   ª       ª   ª               test_rotation_groups.py
ª   ª   ª       ª   ª               test_rotation_spline.py
ª   ª   ª       ª   ª               __init__.py
ª   ª   ª       ª   ª               
ª   ª   ª       ª   +---special
ª   ª   ª       ª   ª   ª   add_newdocs.py
ª   ª   ª       ª   ª   ª   basic.py
ª   ª   ª       ª   ª   ª   cython_special.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   cython_special.pxd
ª   ª   ª       ª   ª   ª   cython_special.pyi
ª   ª   ª       ª   ª   ª   orthogonal.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   sf_error.py
ª   ª   ª       ª   ª   ª   specfun.py
ª   ª   ª       ª   ª   ª   spfun_stats.py
ª   ª   ª       ª   ª   ª   _add_newdocs.py
ª   ª   ª       ª   ª   ª   _basic.py
ª   ª   ª       ª   ª   ª   _comb.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _ellip_harm.py
ª   ª   ª       ª   ª   ª   _ellip_harm_2.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _generate_pyx.py
ª   ª   ª       ª   ª   ª   _lambertw.py
ª   ª   ª       ª   ª   ª   _logsumexp.py
ª   ª   ª       ª   ª   ª   _mptestutils.py
ª   ª   ª       ª   ª   ª   _orthogonal.py
ª   ª   ª       ª   ª   ª   _orthogonal.pyi
ª   ª   ª       ª   ª   ª   _sf_error.py
ª   ª   ª       ª   ª   ª   _specfun.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _spfun_stats.py
ª   ª   ª       ª   ª   ª   _spherical_bessel.py
ª   ª   ª       ª   ª   ª   _testutils.py
ª   ª   ª       ª   ª   ª   _test_round.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _test_round.pyi
ª   ª   ª       ª   ª   ª   _ufuncs.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _ufuncs.pyi
ª   ª   ª       ª   ª   ª   _ufuncs_cxx.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª   ª   ª   test_basic.py
ª   ª   ª       ª   ª   ª   ª   test_bdtr.py
ª   ª   ª       ª   ª   ª   ª   test_boxcox.py
ª   ª   ª       ª   ª   ª   ª   test_cdflib.py
ª   ª   ª       ª   ª   ª   ª   test_cdft_asymptotic.py
ª   ª   ª       ª   ª   ª   ª   test_cosine_distr.py
ª   ª   ª       ª   ª   ª   ª   test_cython_special.py
ª   ª   ª       ª   ª   ª   ª   test_data.py
ª   ª   ª       ª   ª   ª   ª   test_digamma.py
ª   ª   ª       ª   ª   ª   ª   test_ellip_harm.py
ª   ª   ª       ª   ª   ª   ª   test_erfinv.py
ª   ª   ª       ª   ª   ª   ª   test_exponential_integrals.py
ª   ª   ª       ª   ª   ª   ª   test_faddeeva.py
ª   ª   ª       ª   ª   ª   ª   test_gamma.py
ª   ª   ª       ª   ª   ª   ª   test_gammainc.py
ª   ª   ª       ª   ª   ª   ª   test_hyp2f1.py
ª   ª   ª       ª   ª   ª   ª   test_hypergeometric.py
ª   ª   ª       ª   ª   ª   ª   test_kolmogorov.py
ª   ª   ª       ª   ª   ª   ª   test_lambertw.py
ª   ª   ª       ª   ª   ª   ª   test_loggamma.py
ª   ª   ª       ª   ª   ª   ª   test_logit.py
ª   ª   ª       ª   ª   ª   ª   test_logsumexp.py
ª   ª   ª       ª   ª   ª   ª   test_log_softmax.py
ª   ª   ª       ª   ª   ª   ª   test_mpmath.py
ª   ª   ª       ª   ª   ª   ª   test_nan_inputs.py
ª   ª   ª       ª   ª   ª   ª   test_ndtr.py
ª   ª   ª       ª   ª   ª   ª   test_ndtri_exp.py
ª   ª   ª       ª   ª   ª   ª   test_orthogonal.py
ª   ª   ª       ª   ª   ª   ª   test_orthogonal_eval.py
ª   ª   ª       ª   ª   ª   ª   test_owens_t.py
ª   ª   ª       ª   ª   ª   ª   test_pcf.py
ª   ª   ª       ª   ª   ª   ª   test_pdtr.py
ª   ª   ª       ª   ª   ª   ª   test_precompute_expn_asy.py
ª   ª   ª       ª   ª   ª   ª   test_precompute_gammainc.py
ª   ª   ª       ª   ª   ª   ª   test_precompute_utils.py
ª   ª   ª       ª   ª   ª   ª   test_round.py
ª   ª   ª       ª   ª   ª   ª   test_sf_error.py
ª   ª   ª       ª   ª   ª   ª   test_sici.py
ª   ª   ª       ª   ª   ª   ª   test_spence.py
ª   ª   ª       ª   ª   ª   ª   test_spfun_stats.py
ª   ª   ª       ª   ª   ª   ª   test_spherical_bessel.py
ª   ª   ª       ª   ª   ª   ª   test_sph_harm.py
ª   ª   ª       ª   ª   ª   ª   test_trig.py
ª   ª   ª       ª   ª   ª   ª   test_wrightomega.py
ª   ª   ª       ª   ª   ª   ª   test_wright_bessel.py
ª   ª   ª       ª   ª   ª   ª   test_zeta.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---data
ª   ª   ª       ª   ª   ª           boost.npz
ª   ª   ª       ª   ª   ª           gsl.npz
ª   ª   ª       ª   ª   ª           local.npz
ª   ª   ª       ª   ª   ª           README
ª   ª   ª       ª   ª   ª           
ª   ª   ª       ª   ª   +---_precompute
ª   ª   ª       ª   ª           cosine_cdf.py
ª   ª   ª       ª   ª           expn_asy.py
ª   ª   ª       ª   ª           gammainc_asy.py
ª   ª   ª       ª   ª           gammainc_data.py
ª   ª   ª       ª   ª           hyp2f1_data.py
ª   ª   ª       ª   ª           lambertw.py
ª   ª   ª       ª   ª           loggamma.py
ª   ª   ª       ª   ª           setup.py
ª   ª   ª       ª   ª           struve_convergence.py
ª   ª   ª       ª   ª           utils.py
ª   ª   ª       ª   ª           wrightomega.py
ª   ª   ª       ª   ª           wright_bessel.py
ª   ª   ª       ª   ª           wright_bessel_data.py
ª   ª   ª       ª   ª           zetac.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---stats
ª   ª   ª       ª   ª   ª   biasedurn.py
ª   ª   ª       ª   ª   ª   contingency.py
ª   ª   ª       ª   ª   ª   distributions.py
ª   ª   ª       ª   ª   ª   kde.py
ª   ª   ª       ª   ª   ª   morestats.py
ª   ª   ª       ª   ª   ª   mstats.py
ª   ª   ª       ª   ª   ª   mstats_basic.py
ª   ª   ª       ª   ª   ª   mstats_extras.py
ª   ª   ª       ª   ª   ª   mvn.py
ª   ª   ª       ª   ª   ª   qmc.py
ª   ª   ª       ª   ª   ª   sampling.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   statlib.py
ª   ª   ª       ª   ª   ª   stats.py
ª   ª   ª       ª   ª   ª   _axis_nan_policy.py
ª   ª   ª       ª   ª   ª   _biasedurn.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _biasedurn.pxd
ª   ª   ª       ª   ª   ª   _binned_statistic.py
ª   ª   ª       ª   ª   ª   _binomtest.py
ª   ª   ª       ª   ª   ª   _common.py
ª   ª   ª       ª   ª   ª   _constants.py
ª   ª   ª       ª   ª   ª   _continuous_distns.py
ª   ª   ª       ª   ª   ª   _crosstab.py
ª   ª   ª       ª   ª   ª   _discrete_distns.py
ª   ª   ª       ª   ª   ª   _distn_infrastructure.py
ª   ª   ª       ª   ª   ª   _distr_params.py
ª   ª   ª       ª   ª   ª   _entropy.py
ª   ª   ª       ª   ª   ª   _fit.py
ª   ª   ª       ª   ª   ª   _generate_pyx.py
ª   ª   ª       ª   ª   ª   _hypotests.py
ª   ª   ª       ª   ª   ª   _hypotests_pythran.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _hypotests_pythran.py
ª   ª   ª       ª   ª   ª   _kde.py
ª   ª   ª       ª   ª   ª   _ksstats.py
ª   ª   ª       ª   ª   ª   _mannwhitneyu.py
ª   ª   ª       ª   ª   ª   _morestats.py
ª   ª   ª       ª   ª   ª   _mstats_basic.py
ª   ª   ª       ª   ª   ª   _mstats_extras.py
ª   ª   ª       ª   ª   ª   _multivariate.py
ª   ª   ª       ª   ª   ª   _mvn.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _page_trend_test.py
ª   ª   ª       ª   ª   ª   _qmc.py
ª   ª   ª       ª   ª   ª   _qmc_cy.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _qmc_cy.pyi
ª   ª   ª       ª   ª   ª   _relative_risk.py
ª   ª   ª       ª   ª   ª   _resampling.py
ª   ª   ª       ª   ª   ª   _result_classes.py
ª   ª   ª       ª   ª   ª   _rvs_sampling.py
ª   ª   ª       ª   ª   ª   _sobol.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _sobol.pyi
ª   ª   ª       ª   ª   ª   _sobol_direction_numbers.npz
ª   ª   ª       ª   ª   ª   _statlib.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _stats.cp39-win32.pyd
ª   ª   ª       ª   ª   ª   _stats_mstats_common.py
ª   ª   ª       ª   ª   ª   _stats_py.py
ª   ª   ª       ª   ª   ª   _tukeylambda_stats.py
ª   ª   ª       ª   ª   ª   _variation.py
ª   ª   ª       ª   ª   ª   _warnings_errors.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª   ª   ª   common_tests.py
ª   ª   ª       ª   ª   ª   ª   meson.build
ª   ª   ª       ª   ª   ª   ª   studentized_range_mpmath_ref.py
ª   ª   ª       ª   ª   ª   ª   test_axis_nan_policy.py
ª   ª   ª       ª   ª   ª   ª   test_binned_statistic.py
ª   ª   ª       ª   ª   ª   ª   test_boost_ufuncs.py
ª   ª   ª       ª   ª   ª   ª   test_contingency.py
ª   ª   ª       ª   ª   ª   ª   test_continuous_basic.py
ª   ª   ª       ª   ª   ª   ª   test_crosstab.py
ª   ª   ª       ª   ª   ª   ª   test_discrete_basic.py
ª   ª   ª       ª   ª   ª   ª   test_discrete_distns.py
ª   ª   ª       ª   ª   ª   ª   test_distributions.py
ª   ª   ª       ª   ª   ª   ª   test_entropy.py
ª   ª   ª       ª   ª   ª   ª   test_fit.py
ª   ª   ª       ª   ª   ª   ª   test_hypotests.py
ª   ª   ª       ª   ª   ª   ª   test_kdeoth.py
ª   ª   ª       ª   ª   ª   ª   test_morestats.py
ª   ª   ª       ª   ª   ª   ª   test_mstats_basic.py
ª   ª   ª       ª   ª   ª   ª   test_mstats_extras.py
ª   ª   ª       ª   ª   ª   ª   test_multivariate.py
ª   ª   ª       ª   ª   ª   ª   test_qmc.py
ª   ª   ª       ª   ª   ª   ª   test_rank.py
ª   ª   ª       ª   ª   ª   ª   test_relative_risk.py
ª   ª   ª       ª   ª   ª   ª   test_resampling.py
ª   ª   ª       ª   ª   ª   ª   test_sampling.py
ª   ª   ª       ª   ª   ª   ª   test_stats.py
ª   ª   ª       ª   ª   ª   ª   test_tukeylambda_stats.py
ª   ª   ª       ª   ª   ª   ª   test_variation.py
ª   ª   ª       ª   ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   ª   
ª   ª   ª       ª   ª   ª   +---data
ª   ª   ª       ª   ª   ª       ª   meson.build
ª   ª   ª       ª   ª   ª       ª   studentized_range_mpmath_ref.json
ª   ª   ª       ª   ª   ª       ª   
ª   ª   ª       ª   ª   ª       +---levy_stable
ª   ª   ª       ª   ª   ª       ª       meson.build
ª   ª   ª       ª   ª   ª       ª       stable-loc-scale-sample-data.npy
ª   ª   ª       ª   ª   ª       ª       stable-Z1-cdf-sample-data.npy
ª   ª   ª       ª   ª   ª       ª       stable-Z1-pdf-sample-data.npy
ª   ª   ª       ª   ª   ª       ª       
ª   ª   ª       ª   ª   ª       +---nist_anova
ª   ª   ª       ª   ª   ª       ª       AtmWtAg.dat
ª   ª   ª       ª   ª   ª       ª       meson.build
ª   ª   ª       ª   ª   ª       ª       SiRstv.dat
ª   ª   ª       ª   ª   ª       ª       SmLs01.dat
ª   ª   ª       ª   ª   ª       ª       SmLs02.dat
ª   ª   ª       ª   ª   ª       ª       SmLs03.dat
ª   ª   ª       ª   ª   ª       ª       SmLs04.dat
ª   ª   ª       ª   ª   ª       ª       SmLs05.dat
ª   ª   ª       ª   ª   ª       ª       SmLs06.dat
ª   ª   ª       ª   ª   ª       ª       SmLs07.dat
ª   ª   ª       ª   ª   ª       ª       SmLs08.dat
ª   ª   ª       ª   ª   ª       ª       SmLs09.dat
ª   ª   ª       ª   ª   ª       ª       
ª   ª   ª       ª   ª   ª       +---nist_linregress
ª   ª   ª       ª   ª   ª               meson.build
ª   ª   ª       ª   ª   ª               Norris.dat
ª   ª   ª       ª   ª   ª               
ª   ª   ª       ª   ª   +---_boost
ª   ª   ª       ª   ª   ª       beta_ufunc.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       binom_ufunc.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       hypergeom_ufunc.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       nbinom_ufunc.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       ncf_ufunc.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       setup.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_levy_stable
ª   ª   ª       ª   ª   ª       levyst.cp39-win32.pyd
ª   ª   ª       ª   ª   ª       setup.py
ª   ª   ª       ª   ª   ª       __init__.py
ª   ª   ª       ª   ª   ª       
ª   ª   ª       ª   ª   +---_unuran
ª   ª   ª       ª   ª           setup.py
ª   ª   ª       ª   ª           unuran.pxd
ª   ª   ª       ª   ª           unuran_wrapper.cp39-win32.pyd
ª   ª   ª       ª   ª           unuran_wrapper.pyi
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---_build_utils
ª   ª   ª       ª   ª   ª   compiler_helper.py
ª   ª   ª       ª   ª   ª   copyfiles.py
ª   ª   ª       ª   ª   ª   cythoner.py
ª   ª   ª       ª   ª   ª   gcc_build_bitness.py
ª   ª   ª       ª   ª   ª   setup.py
ª   ª   ª       ª   ª   ª   system_info.py
ª   ª   ª       ª   ª   ª   tempita.py
ª   ª   ª       ª   ª   ª   _fortran.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---tests
ª   ª   ª       ª   ª           test_scipy_version.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---_lib
ª   ª   ª       ª       ª   decorator.py
ª   ª   ª       ª       ª   deprecation.py
ª   ª   ª       ª       ª   doccer.py
ª   ª   ª       ª       ª   messagestream.cp39-win32.pyd
ª   ª   ª       ª       ª   setup.py
ª   ª   ª       ª       ª   uarray.py
ª   ª   ª       ª       ª   _boost_utils.py
ª   ª   ª       ª       ª   _bunch.py
ª   ª   ª       ª       ª   _ccallback.py
ª   ª   ª       ª       ª   _ccallback_c.cp39-win32.pyd
ª   ª   ª       ª       ª   _disjoint_set.py
ª   ª   ª       ª       ª   _docscrape.py
ª   ª   ª       ª       ª   _fpumode.cp39-win32.pyd
ª   ª   ª       ª       ª   _gcutils.py
ª   ª   ª       ª       ª   _highs_utils.py
ª   ª   ª       ª       ª   _pep440.py
ª   ª   ª       ª       ª   _testutils.py
ª   ª   ª       ª       ª   _test_ccallback.cp39-win32.pyd
ª   ª   ª       ª       ª   _test_deprecation_call.cp39-win32.pyd
ª   ª   ª       ª       ª   _test_deprecation_def.cp39-win32.pyd
ª   ª   ª       ª       ª   _threadsafety.py
ª   ª   ª       ª       ª   _tmpdirs.py
ª   ª   ª       ª       ª   _unuran_utils.py
ª   ª   ª       ª       ª   _util.py
ª   ª   ª       ª       ª   __init__.py
ª   ª   ª       ª       ª   
ª   ª   ª       ª       +---tests
ª   ª   ª       ª       ª       test_bunch.py
ª   ª   ª       ª       ª       test_ccallback.py
ª   ª   ª       ª       ª       test_deprecation.py
ª   ª   ª       ª       ª       test_import_cycles.py
ª   ª   ª       ª       ª       test_public_api.py
ª   ª   ª       ª       ª       test_tmpdirs.py
ª   ª   ª       ª       ª       test_warnings.py
ª   ª   ª       ª       ª       test__gcutils.py
ª   ª   ª       ª       ª       test__pep440.py
ª   ª   ª       ª       ª       test__testutils.py
ª   ª   ª       ª       ª       test__threadsafety.py
ª   ª   ª       ª       ª       test__util.py
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---_uarray
ª   ª   ª       ª               LICENSE
ª   ª   ª       ª               setup.py
ª   ª   ª       ª               _backend.py
ª   ª   ª       ª               _uarray.cp39-win32.pyd
ª   ª   ª       ª               __init__.py
ª   ª   ª       ª               
ª   ª   ª       +---scipy-1.9.1.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª       LICENSES_bundled.txt
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---setuptools
ª   ª   ª       ª   ª   archive_util.py
ª   ª   ª       ª   ª   build_meta.py
ª   ª   ª       ª   ª   cli-32.exe
ª   ª   ª       ª   ª   cli-64.exe
ª   ª   ª       ª   ª   cli.exe
ª   ª   ª       ª   ª   config.py
ª   ª   ª       ª   ª   depends.py
ª   ª   ª       ª   ª   dep_util.py
ª   ª   ª       ª   ª   dist.py
ª   ª   ª       ª   ª   distutils_patch.py
ª   ª   ª       ª   ª   errors.py
ª   ª   ª       ª   ª   extension.py
ª   ª   ª       ª   ª   glob.py
ª   ª   ª       ª   ª   gui-32.exe
ª   ª   ª       ª   ª   gui-64.exe
ª   ª   ª       ª   ª   gui.exe
ª   ª   ª       ª   ª   installer.py
ª   ª   ª       ª   ª   launch.py
ª   ª   ª       ª   ª   lib2to3_ex.py
ª   ª   ª       ª   ª   monkey.py
ª   ª   ª       ª   ª   msvc.py
ª   ª   ª       ª   ª   namespaces.py
ª   ª   ª       ª   ª   package_index.py
ª   ª   ª       ª   ª   py27compat.py
ª   ª   ª       ª   ª   py31compat.py
ª   ª   ª       ª   ª   py33compat.py
ª   ª   ª       ª   ª   py34compat.py
ª   ª   ª       ª   ª   sandbox.py
ª   ª   ª       ª   ª   script (dev).tmpl
ª   ª   ª       ª   ª   script.tmpl
ª   ª   ª       ª   ª   ssl_support.py
ª   ª   ª       ª   ª   unicode_utils.py
ª   ª   ª       ª   ª   version.py
ª   ª   ª       ª   ª   wheel.py
ª   ª   ª       ª   ª   windows_support.py
ª   ª   ª       ª   ª   _deprecation_warning.py
ª   ª   ª       ª   ª   _imp.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---command
ª   ª   ª       ª   ª       alias.py
ª   ª   ª       ª   ª       bdist_egg.py
ª   ª   ª       ª   ª       bdist_rpm.py
ª   ª   ª       ª   ª       bdist_wininst.py
ª   ª   ª       ª   ª       build_clib.py
ª   ª   ª       ª   ª       build_ext.py
ª   ª   ª       ª   ª       build_py.py
ª   ª   ª       ª   ª       develop.py
ª   ª   ª       ª   ª       dist_info.py
ª   ª   ª       ª   ª       easy_install.py
ª   ª   ª       ª   ª       egg_info.py
ª   ª   ª       ª   ª       install.py
ª   ª   ª       ª   ª       install_egg_info.py
ª   ª   ª       ª   ª       install_lib.py
ª   ª   ª       ª   ª       install_scripts.py
ª   ª   ª       ª   ª       launcher manifest.xml
ª   ª   ª       ª   ª       py36compat.py
ª   ª   ª       ª   ª       register.py
ª   ª   ª       ª   ª       rotate.py
ª   ª   ª       ª   ª       saveopts.py
ª   ª   ª       ª   ª       sdist.py
ª   ª   ª       ª   ª       setopt.py
ª   ª   ª       ª   ª       test.py
ª   ª   ª       ª   ª       upload.py
ª   ª   ª       ª   ª       upload_docs.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---extern
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---_distutils
ª   ª   ª       ª   ª   ª   archive_util.py
ª   ª   ª       ª   ª   ª   bcppcompiler.py
ª   ª   ª       ª   ª   ª   ccompiler.py
ª   ª   ª       ª   ª   ª   cmd.py
ª   ª   ª       ª   ª   ª   config.py
ª   ª   ª       ª   ª   ª   core.py
ª   ª   ª       ª   ª   ª   cygwinccompiler.py
ª   ª   ª       ª   ª   ª   debug.py
ª   ª   ª       ª   ª   ª   dep_util.py
ª   ª   ª       ª   ª   ª   dir_util.py
ª   ª   ª       ª   ª   ª   dist.py
ª   ª   ª       ª   ª   ª   errors.py
ª   ª   ª       ª   ª   ª   extension.py
ª   ª   ª       ª   ª   ª   fancy_getopt.py
ª   ª   ª       ª   ª   ª   filelist.py
ª   ª   ª       ª   ª   ª   file_util.py
ª   ª   ª       ª   ª   ª   log.py
ª   ª   ª       ª   ª   ª   msvc9compiler.py
ª   ª   ª       ª   ª   ª   msvccompiler.py
ª   ª   ª       ª   ª   ª   spawn.py
ª   ª   ª       ª   ª   ª   sysconfig.py
ª   ª   ª       ª   ª   ª   text_file.py
ª   ª   ª       ª   ª   ª   unixccompiler.py
ª   ª   ª       ª   ª   ª   util.py
ª   ª   ª       ª   ª   ª   version.py
ª   ª   ª       ª   ª   ª   versionpredicate.py
ª   ª   ª       ª   ª   ª   _msvccompiler.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---command
ª   ª   ª       ª   ª           bdist.py
ª   ª   ª       ª   ª           bdist_dumb.py
ª   ª   ª       ª   ª           bdist_msi.py
ª   ª   ª       ª   ª           bdist_rpm.py
ª   ª   ª       ª   ª           bdist_wininst.py
ª   ª   ª       ª   ª           build.py
ª   ª   ª       ª   ª           build_clib.py
ª   ª   ª       ª   ª           build_ext.py
ª   ª   ª       ª   ª           build_py.py
ª   ª   ª       ª   ª           build_scripts.py
ª   ª   ª       ª   ª           check.py
ª   ª   ª       ª   ª           clean.py
ª   ª   ª       ª   ª           config.py
ª   ª   ª       ª   ª           install.py
ª   ª   ª       ª   ª           install_data.py
ª   ª   ª       ª   ª           install_egg_info.py
ª   ª   ª       ª   ª           install_headers.py
ª   ª   ª       ª   ª           install_lib.py
ª   ª   ª       ª   ª           install_scripts.py
ª   ª   ª       ª   ª           register.py
ª   ª   ª       ª   ª           sdist.py
ª   ª   ª       ª   ª           upload.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---_vendor
ª   ª   ª       ª       ª   ordered_set.py
ª   ª   ª       ª       ª   pyparsing.py
ª   ª   ª       ª       ª   six.py
ª   ª   ª       ª       ª   __init__.py
ª   ª   ª       ª       ª   
ª   ª   ª       ª       +---packaging
ª   ª   ª       ª               markers.py
ª   ª   ª       ª               requirements.py
ª   ª   ª       ª               specifiers.py
ª   ª   ª       ª               tags.py
ª   ª   ª       ª               utils.py
ª   ª   ª       ª               version.py
ª   ª   ª       ª               _compat.py
ª   ª   ª       ª               _structures.py
ª   ª   ª       ª               __about__.py
ª   ª   ª       ª               __init__.py
ª   ª   ª       ª               
ª   ª   ª       +---setuptools-49.2.1.dist-info
ª   ª   ª       ª       dependency_links.txt
ª   ª   ª       ª       entry_points.txt
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       REQUESTED
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       zip-safe
ª   ª   ª       ª       
ª   ª   ª       +---six-1.16.0.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---threadpoolctl-3.5.0.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---tzdata
ª   ª   ª       ª   ª   zones
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---zoneinfo
ª   ª   ª       ª       ª   CET
ª   ª   ª       ª       ª   CST6CDT
ª   ª   ª       ª       ª   Cuba
ª   ª   ª       ª       ª   EET
ª   ª   ª       ª       ª   Egypt
ª   ª   ª       ª       ª   Eire
ª   ª   ª       ª       ª   EST
ª   ª   ª       ª       ª   EST5EDT
ª   ª   ª       ª       ª   Factory
ª   ª   ª       ª       ª   GB
ª   ª   ª       ª       ª   GB-Eire
ª   ª   ª       ª       ª   GMT
ª   ª   ª       ª       ª   GMT+0
ª   ª   ª       ª       ª   GMT-0
ª   ª   ª       ª       ª   GMT0
ª   ª   ª       ª       ª   Greenwich
ª   ª   ª       ª       ª   Hongkong
ª   ª   ª       ª       ª   HST
ª   ª   ª       ª       ª   Iceland
ª   ª   ª       ª       ª   Iran
ª   ª   ª       ª       ª   iso3166.tab
ª   ª   ª       ª       ª   Israel
ª   ª   ª       ª       ª   Jamaica
ª   ª   ª       ª       ª   Japan
ª   ª   ª       ª       ª   Kwajalein
ª   ª   ª       ª       ª   leapseconds
ª   ª   ª       ª       ª   Libya
ª   ª   ª       ª       ª   MET
ª   ª   ª       ª       ª   MST
ª   ª   ª       ª       ª   MST7MDT
ª   ª   ª       ª       ª   Navajo
ª   ª   ª       ª       ª   NZ
ª   ª   ª       ª       ª   NZ-CHAT
ª   ª   ª       ª       ª   Poland
ª   ª   ª       ª       ª   Portugal
ª   ª   ª       ª       ª   PRC
ª   ª   ª       ª       ª   PST8PDT
ª   ª   ª       ª       ª   ROC
ª   ª   ª       ª       ª   ROK
ª   ª   ª       ª       ª   Singapore
ª   ª   ª       ª       ª   Turkey
ª   ª   ª       ª       ª   tzdata.zi
ª   ª   ª       ª       ª   UCT
ª   ª   ª       ª       ª   Universal
ª   ª   ª       ª       ª   UTC
ª   ª   ª       ª       ª   W-SU
ª   ª   ª       ª       ª   WET
ª   ª   ª       ª       ª   zone.tab
ª   ª   ª       ª       ª   zone1970.tab
ª   ª   ª       ª       ª   zonenow.tab
ª   ª   ª       ª       ª   Zulu
ª   ª   ª       ª       ª   __init__.py
ª   ª   ª       ª       ª   
ª   ª   ª       ª       +---Africa
ª   ª   ª       ª       ª       Abidjan
ª   ª   ª       ª       ª       Accra
ª   ª   ª       ª       ª       Addis_Ababa
ª   ª   ª       ª       ª       Algiers
ª   ª   ª       ª       ª       Asmara
ª   ª   ª       ª       ª       Asmera
ª   ª   ª       ª       ª       Bamako
ª   ª   ª       ª       ª       Bangui
ª   ª   ª       ª       ª       Banjul
ª   ª   ª       ª       ª       Bissau
ª   ª   ª       ª       ª       Blantyre
ª   ª   ª       ª       ª       Brazzaville
ª   ª   ª       ª       ª       Bujumbura
ª   ª   ª       ª       ª       Cairo
ª   ª   ª       ª       ª       Casablanca
ª   ª   ª       ª       ª       Ceuta
ª   ª   ª       ª       ª       Conakry
ª   ª   ª       ª       ª       Dakar
ª   ª   ª       ª       ª       Dar_es_Salaam
ª   ª   ª       ª       ª       Djibouti
ª   ª   ª       ª       ª       Douala
ª   ª   ª       ª       ª       El_Aaiun
ª   ª   ª       ª       ª       Freetown
ª   ª   ª       ª       ª       Gaborone
ª   ª   ª       ª       ª       Harare
ª   ª   ª       ª       ª       Johannesburg
ª   ª   ª       ª       ª       Juba
ª   ª   ª       ª       ª       Kampala
ª   ª   ª       ª       ª       Khartoum
ª   ª   ª       ª       ª       Kigali
ª   ª   ª       ª       ª       Kinshasa
ª   ª   ª       ª       ª       Lagos
ª   ª   ª       ª       ª       Libreville
ª   ª   ª       ª       ª       Lome
ª   ª   ª       ª       ª       Luanda
ª   ª   ª       ª       ª       Lubumbashi
ª   ª   ª       ª       ª       Lusaka
ª   ª   ª       ª       ª       Malabo
ª   ª   ª       ª       ª       Maputo
ª   ª   ª       ª       ª       Maseru
ª   ª   ª       ª       ª       Mbabane
ª   ª   ª       ª       ª       Mogadishu
ª   ª   ª       ª       ª       Monrovia
ª   ª   ª       ª       ª       Nairobi
ª   ª   ª       ª       ª       Ndjamena
ª   ª   ª       ª       ª       Niamey
ª   ª   ª       ª       ª       Nouakchott
ª   ª   ª       ª       ª       Ouagadougou
ª   ª   ª       ª       ª       Porto-Novo
ª   ª   ª       ª       ª       Sao_Tome
ª   ª   ª       ª       ª       Timbuktu
ª   ª   ª       ª       ª       Tripoli
ª   ª   ª       ª       ª       Tunis
ª   ª   ª       ª       ª       Windhoek
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---America
ª   ª   ª       ª       ª   ª   Adak
ª   ª   ª       ª       ª   ª   Anchorage
ª   ª   ª       ª       ª   ª   Anguilla
ª   ª   ª       ª       ª   ª   Antigua
ª   ª   ª       ª       ª   ª   Araguaina
ª   ª   ª       ª       ª   ª   Aruba
ª   ª   ª       ª       ª   ª   Asuncion
ª   ª   ª       ª       ª   ª   Atikokan
ª   ª   ª       ª       ª   ª   Atka
ª   ª   ª       ª       ª   ª   Bahia
ª   ª   ª       ª       ª   ª   Bahia_Banderas
ª   ª   ª       ª       ª   ª   Barbados
ª   ª   ª       ª       ª   ª   Belem
ª   ª   ª       ª       ª   ª   Belize
ª   ª   ª       ª       ª   ª   Blanc-Sablon
ª   ª   ª       ª       ª   ª   Boa_Vista
ª   ª   ª       ª       ª   ª   Bogota
ª   ª   ª       ª       ª   ª   Boise
ª   ª   ª       ª       ª   ª   Buenos_Aires
ª   ª   ª       ª       ª   ª   Cambridge_Bay
ª   ª   ª       ª       ª   ª   Campo_Grande
ª   ª   ª       ª       ª   ª   Cancun
ª   ª   ª       ª       ª   ª   Caracas
ª   ª   ª       ª       ª   ª   Catamarca
ª   ª   ª       ª       ª   ª   Cayenne
ª   ª   ª       ª       ª   ª   Cayman
ª   ª   ª       ª       ª   ª   Chicago
ª   ª   ª       ª       ª   ª   Chihuahua
ª   ª   ª       ª       ª   ª   Ciudad_Juarez
ª   ª   ª       ª       ª   ª   Coral_Harbour
ª   ª   ª       ª       ª   ª   Cordoba
ª   ª   ª       ª       ª   ª   Costa_Rica
ª   ª   ª       ª       ª   ª   Creston
ª   ª   ª       ª       ª   ª   Cuiaba
ª   ª   ª       ª       ª   ª   Curacao
ª   ª   ª       ª       ª   ª   Danmarkshavn
ª   ª   ª       ª       ª   ª   Dawson
ª   ª   ª       ª       ª   ª   Dawson_Creek
ª   ª   ª       ª       ª   ª   Denver
ª   ª   ª       ª       ª   ª   Detroit
ª   ª   ª       ª       ª   ª   Dominica
ª   ª   ª       ª       ª   ª   Edmonton
ª   ª   ª       ª       ª   ª   Eirunepe
ª   ª   ª       ª       ª   ª   El_Salvador
ª   ª   ª       ª       ª   ª   Ensenada
ª   ª   ª       ª       ª   ª   Fortaleza
ª   ª   ª       ª       ª   ª   Fort_Nelson
ª   ª   ª       ª       ª   ª   Fort_Wayne
ª   ª   ª       ª       ª   ª   Glace_Bay
ª   ª   ª       ª       ª   ª   Godthab
ª   ª   ª       ª       ª   ª   Goose_Bay
ª   ª   ª       ª       ª   ª   Grand_Turk
ª   ª   ª       ª       ª   ª   Grenada
ª   ª   ª       ª       ª   ª   Guadeloupe
ª   ª   ª       ª       ª   ª   Guatemala
ª   ª   ª       ª       ª   ª   Guayaquil
ª   ª   ª       ª       ª   ª   Guyana
ª   ª   ª       ª       ª   ª   Halifax
ª   ª   ª       ª       ª   ª   Havana
ª   ª   ª       ª       ª   ª   Hermosillo
ª   ª   ª       ª       ª   ª   Indianapolis
ª   ª   ª       ª       ª   ª   Inuvik
ª   ª   ª       ª       ª   ª   Iqaluit
ª   ª   ª       ª       ª   ª   Jamaica
ª   ª   ª       ª       ª   ª   Jujuy
ª   ª   ª       ª       ª   ª   Juneau
ª   ª   ª       ª       ª   ª   Knox_IN
ª   ª   ª       ª       ª   ª   Kralendijk
ª   ª   ª       ª       ª   ª   La_Paz
ª   ª   ª       ª       ª   ª   Lima
ª   ª   ª       ª       ª   ª   Los_Angeles
ª   ª   ª       ª       ª   ª   Louisville
ª   ª   ª       ª       ª   ª   Lower_Princes
ª   ª   ª       ª       ª   ª   Maceio
ª   ª   ª       ª       ª   ª   Managua
ª   ª   ª       ª       ª   ª   Manaus
ª   ª   ª       ª       ª   ª   Marigot
ª   ª   ª       ª       ª   ª   Martinique
ª   ª   ª       ª       ª   ª   Matamoros
ª   ª   ª       ª       ª   ª   Mazatlan
ª   ª   ª       ª       ª   ª   Mendoza
ª   ª   ª       ª       ª   ª   Menominee
ª   ª   ª       ª       ª   ª   Merida
ª   ª   ª       ª       ª   ª   Metlakatla
ª   ª   ª       ª       ª   ª   Mexico_City
ª   ª   ª       ª       ª   ª   Miquelon
ª   ª   ª       ª       ª   ª   Moncton
ª   ª   ª       ª       ª   ª   Monterrey
ª   ª   ª       ª       ª   ª   Montevideo
ª   ª   ª       ª       ª   ª   Montreal
ª   ª   ª       ª       ª   ª   Montserrat
ª   ª   ª       ª       ª   ª   Nassau
ª   ª   ª       ª       ª   ª   New_York
ª   ª   ª       ª       ª   ª   Nipigon
ª   ª   ª       ª       ª   ª   Nome
ª   ª   ª       ª       ª   ª   Noronha
ª   ª   ª       ª       ª   ª   Nuuk
ª   ª   ª       ª       ª   ª   Ojinaga
ª   ª   ª       ª       ª   ª   Panama
ª   ª   ª       ª       ª   ª   Pangnirtung
ª   ª   ª       ª       ª   ª   Paramaribo
ª   ª   ª       ª       ª   ª   Phoenix
ª   ª   ª       ª       ª   ª   Port-au-Prince
ª   ª   ª       ª       ª   ª   Porto_Acre
ª   ª   ª       ª       ª   ª   Porto_Velho
ª   ª   ª       ª       ª   ª   Port_of_Spain
ª   ª   ª       ª       ª   ª   Puerto_Rico
ª   ª   ª       ª       ª   ª   Punta_Arenas
ª   ª   ª       ª       ª   ª   Rainy_River
ª   ª   ª       ª       ª   ª   Rankin_Inlet
ª   ª   ª       ª       ª   ª   Recife
ª   ª   ª       ª       ª   ª   Regina
ª   ª   ª       ª       ª   ª   Resolute
ª   ª   ª       ª       ª   ª   Rio_Branco
ª   ª   ª       ª       ª   ª   Rosario
ª   ª   ª       ª       ª   ª   Santarem
ª   ª   ª       ª       ª   ª   Santa_Isabel
ª   ª   ª       ª       ª   ª   Santiago
ª   ª   ª       ª       ª   ª   Santo_Domingo
ª   ª   ª       ª       ª   ª   Sao_Paulo
ª   ª   ª       ª       ª   ª   Scoresbysund
ª   ª   ª       ª       ª   ª   Shiprock
ª   ª   ª       ª       ª   ª   Sitka
ª   ª   ª       ª       ª   ª   St_Barthelemy
ª   ª   ª       ª       ª   ª   St_Johns
ª   ª   ª       ª       ª   ª   St_Kitts
ª   ª   ª       ª       ª   ª   St_Lucia
ª   ª   ª       ª       ª   ª   St_Thomas
ª   ª   ª       ª       ª   ª   St_Vincent
ª   ª   ª       ª       ª   ª   Swift_Current
ª   ª   ª       ª       ª   ª   Tegucigalpa
ª   ª   ª       ª       ª   ª   Thule
ª   ª   ª       ª       ª   ª   Thunder_Bay
ª   ª   ª       ª       ª   ª   Tijuana
ª   ª   ª       ª       ª   ª   Toronto
ª   ª   ª       ª       ª   ª   Tortola
ª   ª   ª       ª       ª   ª   Vancouver
ª   ª   ª       ª       ª   ª   Virgin
ª   ª   ª       ª       ª   ª   Whitehorse
ª   ª   ª       ª       ª   ª   Winnipeg
ª   ª   ª       ª       ª   ª   Yakutat
ª   ª   ª       ª       ª   ª   Yellowknife
ª   ª   ª       ª       ª   ª   __init__.py
ª   ª   ª       ª       ª   ª   
ª   ª   ª       ª       ª   +---Argentina
ª   ª   ª       ª       ª   ª       Buenos_Aires
ª   ª   ª       ª       ª   ª       Catamarca
ª   ª   ª       ª       ª   ª       ComodRivadavia
ª   ª   ª       ª       ª   ª       Cordoba
ª   ª   ª       ª       ª   ª       Jujuy
ª   ª   ª       ª       ª   ª       La_Rioja
ª   ª   ª       ª       ª   ª       Mendoza
ª   ª   ª       ª       ª   ª       Rio_Gallegos
ª   ª   ª       ª       ª   ª       Salta
ª   ª   ª       ª       ª   ª       San_Juan
ª   ª   ª       ª       ª   ª       San_Luis
ª   ª   ª       ª       ª   ª       Tucuman
ª   ª   ª       ª       ª   ª       Ushuaia
ª   ª   ª       ª       ª   ª       __init__.py
ª   ª   ª       ª       ª   ª       
ª   ª   ª       ª       ª   +---Indiana
ª   ª   ª       ª       ª   ª       Indianapolis
ª   ª   ª       ª       ª   ª       Knox
ª   ª   ª       ª       ª   ª       Marengo
ª   ª   ª       ª       ª   ª       Petersburg
ª   ª   ª       ª       ª   ª       Tell_City
ª   ª   ª       ª       ª   ª       Vevay
ª   ª   ª       ª       ª   ª       Vincennes
ª   ª   ª       ª       ª   ª       Winamac
ª   ª   ª       ª       ª   ª       __init__.py
ª   ª   ª       ª       ª   ª       
ª   ª   ª       ª       ª   +---Kentucky
ª   ª   ª       ª       ª   ª       Louisville
ª   ª   ª       ª       ª   ª       Monticello
ª   ª   ª       ª       ª   ª       __init__.py
ª   ª   ª       ª       ª   ª       
ª   ª   ª       ª       ª   +---North_Dakota
ª   ª   ª       ª       ª           Beulah
ª   ª   ª       ª       ª           Center
ª   ª   ª       ª       ª           New_Salem
ª   ª   ª       ª       ª           __init__.py
ª   ª   ª       ª       ª           
ª   ª   ª       ª       +---Antarctica
ª   ª   ª       ª       ª       Casey
ª   ª   ª       ª       ª       Davis
ª   ª   ª       ª       ª       DumontDUrville
ª   ª   ª       ª       ª       Macquarie
ª   ª   ª       ª       ª       Mawson
ª   ª   ª       ª       ª       McMurdo
ª   ª   ª       ª       ª       Palmer
ª   ª   ª       ª       ª       Rothera
ª   ª   ª       ª       ª       South_Pole
ª   ª   ª       ª       ª       Syowa
ª   ª   ª       ª       ª       Troll
ª   ª   ª       ª       ª       Vostok
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Arctic
ª   ª   ª       ª       ª       Longyearbyen
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Asia
ª   ª   ª       ª       ª       Aden
ª   ª   ª       ª       ª       Almaty
ª   ª   ª       ª       ª       Amman
ª   ª   ª       ª       ª       Anadyr
ª   ª   ª       ª       ª       Aqtau
ª   ª   ª       ª       ª       Aqtobe
ª   ª   ª       ª       ª       Ashgabat
ª   ª   ª       ª       ª       Ashkhabad
ª   ª   ª       ª       ª       Atyrau
ª   ª   ª       ª       ª       Baghdad
ª   ª   ª       ª       ª       Bahrain
ª   ª   ª       ª       ª       Baku
ª   ª   ª       ª       ª       Bangkok
ª   ª   ª       ª       ª       Barnaul
ª   ª   ª       ª       ª       Beirut
ª   ª   ª       ª       ª       Bishkek
ª   ª   ª       ª       ª       Brunei
ª   ª   ª       ª       ª       Calcutta
ª   ª   ª       ª       ª       Chita
ª   ª   ª       ª       ª       Choibalsan
ª   ª   ª       ª       ª       Chongqing
ª   ª   ª       ª       ª       Chungking
ª   ª   ª       ª       ª       Colombo
ª   ª   ª       ª       ª       Dacca
ª   ª   ª       ª       ª       Damascus
ª   ª   ª       ª       ª       Dhaka
ª   ª   ª       ª       ª       Dili
ª   ª   ª       ª       ª       Dubai
ª   ª   ª       ª       ª       Dushanbe
ª   ª   ª       ª       ª       Famagusta
ª   ª   ª       ª       ª       Gaza
ª   ª   ª       ª       ª       Harbin
ª   ª   ª       ª       ª       Hebron
ª   ª   ª       ª       ª       Hong_Kong
ª   ª   ª       ª       ª       Hovd
ª   ª   ª       ª       ª       Ho_Chi_Minh
ª   ª   ª       ª       ª       Irkutsk
ª   ª   ª       ª       ª       Istanbul
ª   ª   ª       ª       ª       Jakarta
ª   ª   ª       ª       ª       Jayapura
ª   ª   ª       ª       ª       Jerusalem
ª   ª   ª       ª       ª       Kabul
ª   ª   ª       ª       ª       Kamchatka
ª   ª   ª       ª       ª       Karachi
ª   ª   ª       ª       ª       Kashgar
ª   ª   ª       ª       ª       Kathmandu
ª   ª   ª       ª       ª       Katmandu
ª   ª   ª       ª       ª       Khandyga
ª   ª   ª       ª       ª       Kolkata
ª   ª   ª       ª       ª       Krasnoyarsk
ª   ª   ª       ª       ª       Kuala_Lumpur
ª   ª   ª       ª       ª       Kuching
ª   ª   ª       ª       ª       Kuwait
ª   ª   ª       ª       ª       Macao
ª   ª   ª       ª       ª       Macau
ª   ª   ª       ª       ª       Magadan
ª   ª   ª       ª       ª       Makassar
ª   ª   ª       ª       ª       Manila
ª   ª   ª       ª       ª       Muscat
ª   ª   ª       ª       ª       Nicosia
ª   ª   ª       ª       ª       Novokuznetsk
ª   ª   ª       ª       ª       Novosibirsk
ª   ª   ª       ª       ª       Omsk
ª   ª   ª       ª       ª       Oral
ª   ª   ª       ª       ª       Phnom_Penh
ª   ª   ª       ª       ª       Pontianak
ª   ª   ª       ª       ª       Pyongyang
ª   ª   ª       ª       ª       Qatar
ª   ª   ª       ª       ª       Qostanay
ª   ª   ª       ª       ª       Qyzylorda
ª   ª   ª       ª       ª       Rangoon
ª   ª   ª       ª       ª       Riyadh
ª   ª   ª       ª       ª       Saigon
ª   ª   ª       ª       ª       Sakhalin
ª   ª   ª       ª       ª       Samarkand
ª   ª   ª       ª       ª       Seoul
ª   ª   ª       ª       ª       Shanghai
ª   ª   ª       ª       ª       Singapore
ª   ª   ª       ª       ª       Srednekolymsk
ª   ª   ª       ª       ª       Taipei
ª   ª   ª       ª       ª       Tashkent
ª   ª   ª       ª       ª       Tbilisi
ª   ª   ª       ª       ª       Tehran
ª   ª   ª       ª       ª       Tel_Aviv
ª   ª   ª       ª       ª       Thimbu
ª   ª   ª       ª       ª       Thimphu
ª   ª   ª       ª       ª       Tokyo
ª   ª   ª       ª       ª       Tomsk
ª   ª   ª       ª       ª       Ujung_Pandang
ª   ª   ª       ª       ª       Ulaanbaatar
ª   ª   ª       ª       ª       Ulan_Bator
ª   ª   ª       ª       ª       Urumqi
ª   ª   ª       ª       ª       Ust-Nera
ª   ª   ª       ª       ª       Vientiane
ª   ª   ª       ª       ª       Vladivostok
ª   ª   ª       ª       ª       Yakutsk
ª   ª   ª       ª       ª       Yangon
ª   ª   ª       ª       ª       Yekaterinburg
ª   ª   ª       ª       ª       Yerevan
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Atlantic
ª   ª   ª       ª       ª       Azores
ª   ª   ª       ª       ª       Bermuda
ª   ª   ª       ª       ª       Canary
ª   ª   ª       ª       ª       Cape_Verde
ª   ª   ª       ª       ª       Faeroe
ª   ª   ª       ª       ª       Faroe
ª   ª   ª       ª       ª       Jan_Mayen
ª   ª   ª       ª       ª       Madeira
ª   ª   ª       ª       ª       Reykjavik
ª   ª   ª       ª       ª       South_Georgia
ª   ª   ª       ª       ª       Stanley
ª   ª   ª       ª       ª       St_Helena
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Australia
ª   ª   ª       ª       ª       ACT
ª   ª   ª       ª       ª       Adelaide
ª   ª   ª       ª       ª       Brisbane
ª   ª   ª       ª       ª       Broken_Hill
ª   ª   ª       ª       ª       Canberra
ª   ª   ª       ª       ª       Currie
ª   ª   ª       ª       ª       Darwin
ª   ª   ª       ª       ª       Eucla
ª   ª   ª       ª       ª       Hobart
ª   ª   ª       ª       ª       LHI
ª   ª   ª       ª       ª       Lindeman
ª   ª   ª       ª       ª       Lord_Howe
ª   ª   ª       ª       ª       Melbourne
ª   ª   ª       ª       ª       North
ª   ª   ª       ª       ª       NSW
ª   ª   ª       ª       ª       Perth
ª   ª   ª       ª       ª       Queensland
ª   ª   ª       ª       ª       South
ª   ª   ª       ª       ª       Sydney
ª   ª   ª       ª       ª       Tasmania
ª   ª   ª       ª       ª       Victoria
ª   ª   ª       ª       ª       West
ª   ª   ª       ª       ª       Yancowinna
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Brazil
ª   ª   ª       ª       ª       Acre
ª   ª   ª       ª       ª       DeNoronha
ª   ª   ª       ª       ª       East
ª   ª   ª       ª       ª       West
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Canada
ª   ª   ª       ª       ª       Atlantic
ª   ª   ª       ª       ª       Central
ª   ª   ª       ª       ª       Eastern
ª   ª   ª       ª       ª       Mountain
ª   ª   ª       ª       ª       Newfoundland
ª   ª   ª       ª       ª       Pacific
ª   ª   ª       ª       ª       Saskatchewan
ª   ª   ª       ª       ª       Yukon
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Chile
ª   ª   ª       ª       ª       Continental
ª   ª   ª       ª       ª       EasterIsland
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Etc
ª   ª   ª       ª       ª       GMT
ª   ª   ª       ª       ª       GMT+0
ª   ª   ª       ª       ª       GMT+1
ª   ª   ª       ª       ª       GMT+10
ª   ª   ª       ª       ª       GMT+11
ª   ª   ª       ª       ª       GMT+12
ª   ª   ª       ª       ª       GMT+2
ª   ª   ª       ª       ª       GMT+3
ª   ª   ª       ª       ª       GMT+4
ª   ª   ª       ª       ª       GMT+5
ª   ª   ª       ª       ª       GMT+6
ª   ª   ª       ª       ª       GMT+7
ª   ª   ª       ª       ª       GMT+8
ª   ª   ª       ª       ª       GMT+9
ª   ª   ª       ª       ª       GMT-0
ª   ª   ª       ª       ª       GMT-1
ª   ª   ª       ª       ª       GMT-10
ª   ª   ª       ª       ª       GMT-11
ª   ª   ª       ª       ª       GMT-12
ª   ª   ª       ª       ª       GMT-13
ª   ª   ª       ª       ª       GMT-14
ª   ª   ª       ª       ª       GMT-2
ª   ª   ª       ª       ª       GMT-3
ª   ª   ª       ª       ª       GMT-4
ª   ª   ª       ª       ª       GMT-5
ª   ª   ª       ª       ª       GMT-6
ª   ª   ª       ª       ª       GMT-7
ª   ª   ª       ª       ª       GMT-8
ª   ª   ª       ª       ª       GMT-9
ª   ª   ª       ª       ª       GMT0
ª   ª   ª       ª       ª       Greenwich
ª   ª   ª       ª       ª       UCT
ª   ª   ª       ª       ª       Universal
ª   ª   ª       ª       ª       UTC
ª   ª   ª       ª       ª       Zulu
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Europe
ª   ª   ª       ª       ª       Amsterdam
ª   ª   ª       ª       ª       Andorra
ª   ª   ª       ª       ª       Astrakhan
ª   ª   ª       ª       ª       Athens
ª   ª   ª       ª       ª       Belfast
ª   ª   ª       ª       ª       Belgrade
ª   ª   ª       ª       ª       Berlin
ª   ª   ª       ª       ª       Bratislava
ª   ª   ª       ª       ª       Brussels
ª   ª   ª       ª       ª       Bucharest
ª   ª   ª       ª       ª       Budapest
ª   ª   ª       ª       ª       Busingen
ª   ª   ª       ª       ª       Chisinau
ª   ª   ª       ª       ª       Copenhagen
ª   ª   ª       ª       ª       Dublin
ª   ª   ª       ª       ª       Gibraltar
ª   ª   ª       ª       ª       Guernsey
ª   ª   ª       ª       ª       Helsinki
ª   ª   ª       ª       ª       Isle_of_Man
ª   ª   ª       ª       ª       Istanbul
ª   ª   ª       ª       ª       Jersey
ª   ª   ª       ª       ª       Kaliningrad
ª   ª   ª       ª       ª       Kiev
ª   ª   ª       ª       ª       Kirov
ª   ª   ª       ª       ª       Kyiv
ª   ª   ª       ª       ª       Lisbon
ª   ª   ª       ª       ª       Ljubljana
ª   ª   ª       ª       ª       London
ª   ª   ª       ª       ª       Luxembourg
ª   ª   ª       ª       ª       Madrid
ª   ª   ª       ª       ª       Malta
ª   ª   ª       ª       ª       Mariehamn
ª   ª   ª       ª       ª       Minsk
ª   ª   ª       ª       ª       Monaco
ª   ª   ª       ª       ª       Moscow
ª   ª   ª       ª       ª       Nicosia
ª   ª   ª       ª       ª       Oslo
ª   ª   ª       ª       ª       Paris
ª   ª   ª       ª       ª       Podgorica
ª   ª   ª       ª       ª       Prague
ª   ª   ª       ª       ª       Riga
ª   ª   ª       ª       ª       Rome
ª   ª   ª       ª       ª       Samara
ª   ª   ª       ª       ª       San_Marino
ª   ª   ª       ª       ª       Sarajevo
ª   ª   ª       ª       ª       Saratov
ª   ª   ª       ª       ª       Simferopol
ª   ª   ª       ª       ª       Skopje
ª   ª   ª       ª       ª       Sofia
ª   ª   ª       ª       ª       Stockholm
ª   ª   ª       ª       ª       Tallinn
ª   ª   ª       ª       ª       Tirane
ª   ª   ª       ª       ª       Tiraspol
ª   ª   ª       ª       ª       Ulyanovsk
ª   ª   ª       ª       ª       Uzhgorod
ª   ª   ª       ª       ª       Vaduz
ª   ª   ª       ª       ª       Vatican
ª   ª   ª       ª       ª       Vienna
ª   ª   ª       ª       ª       Vilnius
ª   ª   ª       ª       ª       Volgograd
ª   ª   ª       ª       ª       Warsaw
ª   ª   ª       ª       ª       Zagreb
ª   ª   ª       ª       ª       Zaporozhye
ª   ª   ª       ª       ª       Zurich
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Indian
ª   ª   ª       ª       ª       Antananarivo
ª   ª   ª       ª       ª       Chagos
ª   ª   ª       ª       ª       Christmas
ª   ª   ª       ª       ª       Cocos
ª   ª   ª       ª       ª       Comoro
ª   ª   ª       ª       ª       Kerguelen
ª   ª   ª       ª       ª       Mahe
ª   ª   ª       ª       ª       Maldives
ª   ª   ª       ª       ª       Mauritius
ª   ª   ª       ª       ª       Mayotte
ª   ª   ª       ª       ª       Reunion
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Mexico
ª   ª   ª       ª       ª       BajaNorte
ª   ª   ª       ª       ª       BajaSur
ª   ª   ª       ª       ª       General
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---Pacific
ª   ª   ª       ª       ª       Apia
ª   ª   ª       ª       ª       Auckland
ª   ª   ª       ª       ª       Bougainville
ª   ª   ª       ª       ª       Chatham
ª   ª   ª       ª       ª       Chuuk
ª   ª   ª       ª       ª       Easter
ª   ª   ª       ª       ª       Efate
ª   ª   ª       ª       ª       Enderbury
ª   ª   ª       ª       ª       Fakaofo
ª   ª   ª       ª       ª       Fiji
ª   ª   ª       ª       ª       Funafuti
ª   ª   ª       ª       ª       Galapagos
ª   ª   ª       ª       ª       Gambier
ª   ª   ª       ª       ª       Guadalcanal
ª   ª   ª       ª       ª       Guam
ª   ª   ª       ª       ª       Honolulu
ª   ª   ª       ª       ª       Johnston
ª   ª   ª       ª       ª       Kanton
ª   ª   ª       ª       ª       Kiritimati
ª   ª   ª       ª       ª       Kosrae
ª   ª   ª       ª       ª       Kwajalein
ª   ª   ª       ª       ª       Majuro
ª   ª   ª       ª       ª       Marquesas
ª   ª   ª       ª       ª       Midway
ª   ª   ª       ª       ª       Nauru
ª   ª   ª       ª       ª       Niue
ª   ª   ª       ª       ª       Norfolk
ª   ª   ª       ª       ª       Noumea
ª   ª   ª       ª       ª       Pago_Pago
ª   ª   ª       ª       ª       Palau
ª   ª   ª       ª       ª       Pitcairn
ª   ª   ª       ª       ª       Pohnpei
ª   ª   ª       ª       ª       Ponape
ª   ª   ª       ª       ª       Port_Moresby
ª   ª   ª       ª       ª       Rarotonga
ª   ª   ª       ª       ª       Saipan
ª   ª   ª       ª       ª       Samoa
ª   ª   ª       ª       ª       Tahiti
ª   ª   ª       ª       ª       Tarawa
ª   ª   ª       ª       ª       Tongatapu
ª   ª   ª       ª       ª       Truk
ª   ª   ª       ª       ª       Wake
ª   ª   ª       ª       ª       Wallis
ª   ª   ª       ª       ª       Yap
ª   ª   ª       ª       ª       __init__.py
ª   ª   ª       ª       ª       
ª   ª   ª       ª       +---US
ª   ª   ª       ª               Alaska
ª   ª   ª       ª               Aleutian
ª   ª   ª       ª               Arizona
ª   ª   ª       ª               Central
ª   ª   ª       ª               East-Indiana
ª   ª   ª       ª               Eastern
ª   ª   ª       ª               Hawaii
ª   ª   ª       ª               Indiana-Starke
ª   ª   ª       ª               Michigan
ª   ª   ª       ª               Mountain
ª   ª   ª       ª               Pacific
ª   ª   ª       ª               Samoa
ª   ª   ª       ª               __init__.py
ª   ª   ª       ª               
ª   ª   ª       +---tzdata-2024.2.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE
ª   ª   ª       ª       LICENSE_APACHE
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       top_level.txt
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---urllib3
ª   ª   ª       ª   ª   connection.py
ª   ª   ª       ª   ª   connectionpool.py
ª   ª   ª       ª   ª   exceptions.py
ª   ª   ª       ª   ª   fields.py
ª   ª   ª       ª   ª   filepost.py
ª   ª   ª       ª   ª   poolmanager.py
ª   ª   ª       ª   ª   py.typed
ª   ª   ª       ª   ª   response.py
ª   ª   ª       ª   ª   _base_connection.py
ª   ª   ª       ª   ª   _collections.py
ª   ª   ª       ª   ª   _request_methods.py
ª   ª   ª       ª   ª   _version.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---contrib
ª   ª   ª       ª   ª   ª   pyopenssl.py
ª   ª   ª       ª   ª   ª   socks.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---emscripten
ª   ª   ª       ª   ª           connection.py
ª   ª   ª       ª   ª           emscripten_fetch_worker.js
ª   ª   ª       ª   ª           fetch.py
ª   ª   ª       ª   ª           request.py
ª   ª   ª       ª   ª           response.py
ª   ª   ª       ª   ª           __init__.py
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---http2
ª   ª   ª       ª   ª       connection.py
ª   ª   ª       ª   ª       probe.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---util
ª   ª   ª       ª           connection.py
ª   ª   ª       ª           proxy.py
ª   ª   ª       ª           request.py
ª   ª   ª       ª           response.py
ª   ª   ª       ª           retry.py
ª   ª   ª       ª           ssltransport.py
ª   ª   ª       ª           ssl_.py
ª   ª   ª       ª           ssl_match_hostname.py
ª   ª   ª       ª           timeout.py
ª   ª   ª       ª           url.py
ª   ª   ª       ª           util.py
ª   ª   ª       ª           wait.py
ª   ª   ª       ª           __init__.py
ª   ª   ª       ª           
ª   ª   ª       +---urllib3-2.2.3.dist-info
ª   ª   ª       ª   ª   INSTALLER
ª   ª   ª       ª   ª   METADATA
ª   ª   ª       ª   ª   RECORD
ª   ª   ª       ª   ª   WHEEL
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---licenses
ª   ª   ª       ª           LICENSE.txt
ª   ª   ª       ª           
ª   ª   ª       +---werkzeug
ª   ª   ª       ª   ª   exceptions.py
ª   ª   ª       ª   ª   formparser.py
ª   ª   ª       ª   ª   http.py
ª   ª   ª       ª   ª   local.py
ª   ª   ª       ª   ª   py.typed
ª   ª   ª       ª   ª   security.py
ª   ª   ª       ª   ª   serving.py
ª   ª   ª       ª   ª   test.py
ª   ª   ª       ª   ª   testapp.py
ª   ª   ª       ª   ª   urls.py
ª   ª   ª       ª   ª   user_agent.py
ª   ª   ª       ª   ª   utils.py
ª   ª   ª       ª   ª   wsgi.py
ª   ª   ª       ª   ª   _internal.py
ª   ª   ª       ª   ª   _reloader.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---datastructures
ª   ª   ª       ª   ª       accept.py
ª   ª   ª       ª   ª       auth.py
ª   ª   ª       ª   ª       cache_control.py
ª   ª   ª       ª   ª       csp.py
ª   ª   ª       ª   ª       etag.py
ª   ª   ª       ª   ª       file_storage.py
ª   ª   ª       ª   ª       headers.py
ª   ª   ª       ª   ª       mixins.py
ª   ª   ª       ª   ª       range.py
ª   ª   ª       ª   ª       structures.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---debug
ª   ª   ª       ª   ª   ª   console.py
ª   ª   ª       ª   ª   ª   repr.py
ª   ª   ª       ª   ª   ª   tbtools.py
ª   ª   ª       ª   ª   ª   __init__.py
ª   ª   ª       ª   ª   ª   
ª   ª   ª       ª   ª   +---shared
ª   ª   ª       ª   ª           console.png
ª   ª   ª       ª   ª           debugger.js
ª   ª   ª       ª   ª           ICON_LICENSE.md
ª   ª   ª       ª   ª           less.png
ª   ª   ª       ª   ª           more.png
ª   ª   ª       ª   ª           style.css
ª   ª   ª       ª   ª           
ª   ª   ª       ª   +---middleware
ª   ª   ª       ª   ª       dispatcher.py
ª   ª   ª       ª   ª       http_proxy.py
ª   ª   ª       ª   ª       lint.py
ª   ª   ª       ª   ª       profiler.py
ª   ª   ª       ª   ª       proxy_fix.py
ª   ª   ª       ª   ª       shared_data.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---routing
ª   ª   ª       ª   ª       converters.py
ª   ª   ª       ª   ª       exceptions.py
ª   ª   ª       ª   ª       map.py
ª   ª   ª       ª   ª       matcher.py
ª   ª   ª       ª   ª       rules.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---sansio
ª   ª   ª       ª   ª       http.py
ª   ª   ª       ª   ª       multipart.py
ª   ª   ª       ª   ª       request.py
ª   ª   ª       ª   ª       response.py
ª   ª   ª       ª   ª       utils.py
ª   ª   ª       ª   ª       __init__.py
ª   ª   ª       ª   ª       
ª   ª   ª       ª   +---wrappers
ª   ª   ª       ª           request.py
ª   ª   ª       ª           response.py
ª   ª   ª       ª           __init__.py
ª   ª   ª       ª           
ª   ª   ª       +---werkzeug-3.1.3.dist-info
ª   ª   ª       ª       INSTALLER
ª   ª   ª       ª       LICENSE.txt
ª   ª   ª       ª       METADATA
ª   ª   ª       ª       RECORD
ª   ª   ª       ª       WHEEL
ª   ª   ª       ª       
ª   ª   ª       +---zipp
ª   ª   ª       ª   ª   glob.py
ª   ª   ª       ª   ª   _functools.py
ª   ª   ª       ª   ª   __init__.py
ª   ª   ª       ª   ª   
ª   ª   ª       ª   +---compat
ª   ª   ª       ª           overlay.py
ª   ª   ª       ª           py310.py
ª   ª   ª       ª           __init__.py
ª   ª   ª       ª           
ª   ª   ª       +---zipp-3.21.0.dist-info
ª   ª   ª               INSTALLER
ª   ª   ª               LICENSE
ª   ª   ª               METADATA
ª   ª   ª               RECORD
ª   ª   ª               top_level.txt
ª   ª   ª               WHEEL
ª   ª   ª               
ª   ª   +---Scripts
ª   ª           activate
ª   ª           activate.bat
ª   ª           Activate.ps1
ª   ª           deactivate.bat
ª   ª           easy_install-3.9.exe
ª   ª           easy_install.exe
ª   ª           f2py.exe
ª   ª           flask.exe
ª   ª           normalizer.exe
ª   ª           pip.exe
ª   ª           pip3.9.exe
ª   ª           pip3.exe
ª   ª           pyrsa-decrypt.exe
ª   ª           pyrsa-encrypt.exe
ª   ª           pyrsa-keygen.exe
ª   ª           pyrsa-priv2pub.exe
ª   ª           pyrsa-sign.exe
ª   ª           pyrsa-verify.exe
ª   ª           python.exe
ª   ª           pythonw.exe
ª   ª           
ª   +---templates
ª           index.html
ª           
+---logs
ª   ª   download_data.log
ª   ª   outlier_handling.log
ª   ª   process_data.log
ª   ª   
ª   +---dag_id=DataPipeline
ª   ª   +---run_id=manual__2024-11-12T22?38?50.207970+00?00
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           attempt=2.log
ª   ª   ª           attempt=3.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-12T23?03?26.665725+00?00
ª   ª   ª   +---task_id=anomaly_detection
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=correlation_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=encode_categorical_variables
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=perform_eda
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task1
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task2
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_alert_if_anomalies
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_email_notification
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=smote_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-12T23?47?32.430261+00?00
ª   ª   ª   +---task_id=anomaly_detection
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=correlation_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=encode_categorical_variables
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=perform_eda
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task1
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task2
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_alert_if_anomalies
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_email_notification
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=smote_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-12T23?47?39.632532+00?00
ª   ª   ª   +---task_id=anomaly_detection
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=correlation_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=encode_categorical_variables
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=perform_eda
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task1
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task2
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_alert_if_anomalies
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_email_notification
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=smote_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T00?37?25.094356+00?00
ª   ª   ª   +---task_id=anomaly_detection
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=correlation_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=encode_categorical_variables
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=perform_eda
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task1
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task2
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_alert_if_anomalies
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_email_notification
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=smote_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T01?52?06.915145+00?00
ª   ª   ª   +---task_id=anomaly_detection
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=correlation_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=encode_categorical_variables
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=perform_eda
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task1
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task2
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_alert_if_anomalies
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_email_notification
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=smote_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T02?27?34.971318+00?00
ª   ª   ª   +---task_id=anomaly_detection
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=correlation_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=encode_categorical_variables
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=perform_eda
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task1
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task2
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_alert_if_anomalies
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_email_notification
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=smote_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T03?05?10.820837+00?00
ª   ª   ª   +---task_id=anomaly_detection
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=correlation_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=encode_categorical_variables
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=perform_eda
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task1
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task2
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_alert_if_anomalies
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_email_notification
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=smote_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T19?51?12.763358+00?00
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           attempt=2.log
ª   ª   ª           attempt=3.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T20?04?27.724329+00?00
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           attempt=2.log
ª   ª   ª           attempt=3.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T20?29?55.315384+00?00
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           attempt=2.log
ª   ª   ª           attempt=3.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-14T01?40?43.888608+00?00
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log.SchedulerJob.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-14T06?48?32.060157+00?00
ª   ª   ª   +---task_id=anomaly_detection
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=correlation_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log.SchedulerJob.log
ª   ª   ª   ª       attempt=2.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=encode_categorical_variables
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=perform_eda
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task1
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task2
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_alert_if_anomalies
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_email_notification
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=smote_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-23T07?19?01.570527+00?00
ª   ª   ª   +---task_id=anomaly_detection
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=correlation_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=encode_categorical_variables
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=perform_eda
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task1
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task2
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_alert_if_anomalies
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_email_notification
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       attempt=2.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=smote_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-23T08?09?06.240775+00?00
ª   ª   ª   +---task_id=anomaly_detection
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=correlation_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=encode_categorical_variables
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=perform_eda
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task1
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task2
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_alert_if_anomalies
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_email_notification
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=smote_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-23T10?31?35.988168+00?00
ª   ª   ª   +---task_id=anomaly_detection
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=correlation_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=download_data_from_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=encode_categorical_variables
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=load_data
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=perform_eda
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task1
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=pre_process_task2
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_alert_if_anomalies
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_email_notification
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=smote_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=trigger_model_pipeline_task
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=validate_data_schema
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-23T10?38?45.544500+00?00
ª   ª       +---task_id=anomaly_detection
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=correlation_analysis
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=download_data_from_gcp
ª   ª       ª       attempt=1.log.SchedulerJob.log
ª   ª       ª       attempt=2.log
ª   ª       ª       
ª   ª       +---task_id=encode_categorical_variables
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=load_data
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=perform_eda
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=pre_process_task1
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=pre_process_task2
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=send_alert_if_anomalies
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=send_email_notification
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=smote_analysis
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=trigger_model_pipeline_task
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=validate_data_schema
ª   ª               attempt=1.log
ª   ª               
ª   +---dag_id=ModelDevelopmentPipeline
ª   ª   +---run_id=manual__2024-11-13T00?41?45.642098+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T01?09?27.866921+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T01?35?03.984580+00?00
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           attempt=1.log.SchedulerJob.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T01?57?10.077980+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T02?30?49.866771+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T03?08?50.547838+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T04?33?53.463270+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T05?38?45.387466+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T06?03?22.160235+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T06?09?58.166768+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T06?11?58.613137+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T06?33?07.799755+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T06?34?46.744729+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T06?41?49.723730+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T07?11?12.863087+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T18?23?39.751290+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T19?12?55.629422+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T19?27?01.661220+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T19?45?44.265613+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T20?02?51.080238+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T20?19?08.676406+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T20?36?08.649380+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-13T20?41?00.369410+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-14T01?46?33.120567+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-14T02?25?25.701050+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-23T06?24?31.814520+00?00
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-23T07?38?39.163603+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_bias_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_completion_email
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=sensitivity_analysis_task
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-23T08?12?59.780259+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_bias_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_completion_email
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=sensitivity_analysis_task
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-23T10?35?16.876670+00?00
ª   ª   ª   +---task_id=compare_best_models
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=push_to_gcp
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_bias_analysis
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=run_model_development
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=send_completion_email
ª   ª   ª   ª       attempt=1.log
ª   ª   ª   ª       
ª   ª   ª   +---task_id=sensitivity_analysis_task
ª   ª   ª           attempt=1.log
ª   ª   ª           
ª   ª   +---run_id=manual__2024-11-23T10?48?52.641852+00?00
ª   ª       +---task_id=compare_best_models
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=push_to_gcp
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=run_bias_analysis
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=run_model_development
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=send_completion_email
ª   ª       ª       attempt=1.log
ª   ª       ª       
ª   ª       +---task_id=sensitivity_analysis_task
ª   ª               attempt=1.log
ª   ª               
ª   +---dag_processor_manager
ª   ª       dag_processor_manager.log
ª   ª       
ª   +---scheduler
ª       +---2024-11-01
ª       ª   +---src
ª       ª       ª   airflow.py.log
ª       ª       ª   eda.py.log
ª       ª       ª   
ª       ª       +---data_preprocessing
ª       ª       ª       correlation_analysis.py.log
ª       ª       ª       encoding.py.log
ª       ª       ª       smote.py.log
ª       ª       ª       
ª       ª       +---Data_validation
ª       ª               anomaly_detection_alerts.py.log
ª       ª               data_schema_statistics_generation.py.log
ª       ª               
ª       +---2024-11-02
ª       ª   +---src
ª       ª       ª   airflow.py.log
ª       ª       ª   eda.py.log
ª       ª       ª   
ª       ª       +---data_preprocessing
ª       ª       ª       correlation_analysis.py.log
ª       ª       ª       encoding.py.log
ª       ª       ª       smote.py.log
ª       ª       ª       
ª       ª       +---Data_validation
ª       ª               anomaly_detection.py.log
ª       ª               anomaly_detection_alerts.py.log
ª       ª               data_schema_statistics_generation.py.log
ª       ª               
ª       +---2024-11-03
ª       ª   +---src
ª       ª       ª   airflow.py.log
ª       ª       ª   eda.py.log
ª       ª       ª   
ª       ª       +---data_preprocessing
ª       ª       ª       correlation_analysis.py.log
ª       ª       ª       encoding.py.log
ª       ª       ª       smote.py.log
ª       ª       ª       
ª       ª       +---Data_validation
ª       ª               anomaly_detection.py.log
ª       ª               data_schema_statistics_generation.py.log
ª       ª               
ª       +---2024-11-05
ª       ª   +---src
ª       ª       ª   airflow.py.log
ª       ª       ª   eda.py.log
ª       ª       ª   
ª       ª       +---data_preprocessing
ª       ª       ª       correlation_analysis.py.log
ª       ª       ª       encoding.py.log
ª       ª       ª       smote.py.log
ª       ª       ª       
ª       ª       +---Data_validation
ª       ª               anomaly_detection.py.log
ª       ª               data_schema_statistics_generation.py.log
ª       ª               
ª       +---2024-11-07
ª       ª   +---src
ª       ª       ª   airflow.py.log
ª       ª       ª   eda.py.log
ª       ª       ª   
ª       ª       +---data_preprocessing
ª       ª       ª       correlation_analysis.py.log
ª       ª       ª       encoding.py.log
ª       ª       ª       smote.py.log
ª       ª       ª       
ª       ª       +---Data_validation
ª       ª               data_schema_statistics_generation.py.log
ª       ª               
ª       +---2024-11-09
ª       ª   +---src
ª       ª       ª   airflow.py.log
ª       ª       ª   data_pipeline.py.log
ª       ª       ª   eda.py.log
ª       ª       ª   model_development_pipeline.py.log
ª       ª       ª   
ª       ª       +---data_preprocessing
ª       ª       ª       correlation_analysis.py.log
ª       ª       ª       encoding.py.log
ª       ª       ª       smote.py.log
ª       ª       ª       
ª       ª       +---Data_validation
ª       ª               c.py.log
ª       ª               data_schema_statistics_generation.py.log
ª       ª               
ª       +---2024-11-10
ª       ª   +---src
ª       ª       ª   airflow.py.log
ª       ª       ª   data_pipeline.py.log
ª       ª       ª   eda.py.log
ª       ª       ª   model_development_pipeline.py.log
ª       ª       ª   
ª       ª       +---data_preprocessing
ª       ª       ª       correlation_analysis.py.log
ª       ª       ª       encoding.py.log
ª       ª       ª       smote.py.log
ª       ª       ª       
ª       ª       +---Data_validation
ª       ª               data_schema_statistics_generation.py.log
ª       ª               
ª       +---2024-11-12
ª       ª   +---src
ª       ª       ª   data_pipeline.py.log
ª       ª       ª   eda.py.log
ª       ª       ª   model_development_pipeline.py.log
ª       ª       ª   
ª       ª       +---data_preprocessing
ª       ª       ª       correlation_analysis.py.log
ª       ª       ª       encoding.py.log
ª       ª       ª       smote.py.log
ª       ª       ª       
ª       ª       +---Data_validation
ª       ª       ª       data_schema_statistics_generation.py.log
ª       ª       ª       
ª       ª       +---Model_Pipeline
ª       ª               compare_best_models.py.log
ª       ª               
ª       +---2024-11-13
ª       ª   +---src
ª       ª       ª   data_pipeline.py.log
ª       ª       ª   eda.py.log
ª       ª       ª   model_development_pipeline.py.log
ª       ª       ª   
ª       ª       +---data_preprocessing
ª       ª       ª       correlation_analysis.py.log
ª       ª       ª       encoding.py.log
ª       ª       ª       smote.py.log
ª       ª       ª       
ª       ª       +---Data_validation
ª       ª       ª       data_schema_statistics_generation.py.log
ª       ª       ª       
ª       ª       +---Model_Pipeline
ª       ª               compare_best_models.py.log
ª       ª               push_to_gcp.py.log
ª       ª               
ª       +---2024-11-14
ª       ª   +---src
ª       ª       ª   airflow.py.log
ª       ª       ª   data_pipeline.py.log
ª       ª       ª   eda.py.log
ª       ª       ª   model_development_pipeline.py.log
ª       ª       ª   
ª       ª       +---data_preprocessing
ª       ª       ª       correlation_analysis.py.log
ª       ª       ª       encoding.py.log
ª       ª       ª       smote.py.log
ª       ª       ª       
ª       ª       +---Data_validation
ª       ª       ª       anomaly_detection.py.log
ª       ª       ª       data_schema_statistics_generation.py.log
ª       ª       ª       
ª       ª       +---Model_Pipeline
ª       ª               push_to_gcp.py.log
ª       ª               
ª       +---2024-11-23
ª       ª   +---src
ª       ª       ª   data_pipeline.py.log
ª       ª       ª   eda.py.log
ª       ª       ª   model_development_pipeline.py.log
ª       ª       ª   
ª       ª       +---data_preprocessing
ª       ª       ª       correlation_analysis.py.log
ª       ª       ª       encoding.py.log
ª       ª       ª       
ª       ª       +---Data_validation
ª       ª       ª       data_schema_statistics_generation.py.log
ª       ª       ª       
ª       ª       +---Model_Pipeline
ª       ª               push_to_gcp.py.log
ª       ª               
ª       +---2024-11-26
ª       ª   +---src
ª       ª       ª   data_pipeline.py.log
ª       ª       ª   eda.py.log
ª       ª       ª   model_development_pipeline.py.log
ª       ª       ª   
ª       ª       +---data_preprocessing
ª       ª       ª       correlation_analysis.py.log
ª       ª       ª       encoding.py.log
ª       ª       ª       
ª       ª       +---Data_validation
ª       ª       ª       data_schema_statistics_generation.py.log
ª       ª       ª       
ª       ª       +---Model_Pipeline
ª       ª               push_to_gcp.py.log
ª       ª               
ª       +---latest
+---logstash
+---mlruns
+---plugins
+---src
ª   ª   __init__.py
ª   ª   
ª   +---utils
ª           __init__.py
ª           
+---tests
        test_CorrelationAndEncoding.py
        test_data_format.py
        test_DownloadAndLoadData.py
        test_eda.py
        test_HandleNullValues.py
        test_outliers_handling.py
        test_Smote.py
        

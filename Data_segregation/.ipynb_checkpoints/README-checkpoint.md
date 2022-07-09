# Instructions
In this exercise you will complete a MLflow component that divides the data into training and 
test sample.

## Run Steps

To run the project, execute the script:

```bash
mlflow run . -P input_artifact="trabalho_2_preprocessing/preprocessed_data.csv:latest" \
             -P artifact_root="data" \
             -P artifact_type="trainvaltest_data" \
             -P test_size=0.3 \
             -P stratify="instant_bookable" \
             -P random_state="13"
```
This command splits the dataset in train/test with the test accounting for 30% of the original
dataset. The split is stratified according to the target, to keep the same label balance. 
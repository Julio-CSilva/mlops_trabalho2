import pytest
import wandb
import pandas as pd

# This is global so all tests are collected under the same
# run
run = wandb.init(project="trabalho_2_data_checks", job_type="data_checks")


@pytest.fixture(scope="session")
def data():

    local_path = run.use_artifact("trabalho_2_preprocessing/preprocessed_data.csv:latest").file()
    df = pd.read_csv(local_path)

    return df


def test_column_presence_and_type(data):

    required_columns = {
        "id": pd.api.types.is_int64_dtype,
        "listing_url": pd.api.types.is_object_dtype,
        "last_scraped": pd.api.types.is_object_dtype,
        "name": pd.api.types.is_object_dtype,
        "description": pd.api.types.is_object_dtype,
        "neighborhood_overview": pd.api.types.is_object_dtype,
        "picture_url": pd.api.types.is_object_dtype,
        "host_id": pd.api.types.is_int64_dtype,
        "host_url": pd.api.types.is_object_dtype,
        "host_name": pd.api.types.is_object_dtype,
        "host_since": pd.api.types.is_object_dtype,
        "host_location": pd.api.types.is_object_dtype,  
        "host_about": pd.api.types.is_object_dtype,
        "host_response_time": pd.api.types.is_object_dtype,
        "host_response_rate": pd.api.types.is_object_dtype,
        "host_acceptance_rate": pd.api.types.is_object_dtype,
        "host_is_superhost": pd.api.types.is_object_dtype,
        "host_thumbnail_url": pd.api.types.is_object_dtype,
        "host_picture_url": pd.api.types.is_object_dtype,
        "host_neighbourhood": pd.api.types.is_object_dtype,
        "host_total_listings_count": pd.api.types.is_float_dtype,
        "host_verifications": pd.api.types.is_object_dtype,
        "neighbourhood": pd.api.types.is_object_dtype,
        "neighbourhood_cleansed": pd.api.types.is_object_dtype,
        "property_type": pd.api.types.is_object_dtype,
        "room_type": pd.api.types.is_object_dtype,
        "accommodates": pd.api.types.is_int64_dtype,
        "bathrooms_text": pd.api.types.is_object_dtype,
        "bedrooms": pd.api.types.is_float_dtype,
        "beds": pd.api.types.is_float_dtype,
        "amenities": pd.api.types.is_object_dtype,
        "price": pd.api.types.is_object_dtype,
        "minimum_nights": pd.api.types.is_int64_dtype,
        "maximum_nights": pd.api.types.is_int64_dtype,
        "minimum_minimum_nights": pd.api.types.is_int64_dtype,
        "maximum_minimum_nights": pd.api.types.is_int64_dtype,
        "minimum_maximum_nights": pd.api.types.is_int64_dtype,
        "maximum_maximum_nights": pd.api.types.is_int64_dtype,
        "has_availability": pd.api.types.is_object_dtype,
        "availability_30": pd.api.types.is_int64_dtype,
        "availability_60": pd.api.types.is_int64_dtype,
        "availability_90": pd.api.types.is_int64_dtype,
        "availability_365": pd.api.types.is_int64_dtype,
        "calendar_last_scraped": pd.api.types.is_object_dtype,
        "number_of_reviews": pd.api.types.is_int64_dtype,
        "number_of_reviews_ltm": pd.api.types.is_int64_dtype,
        "number_of_reviews_l30d": pd.api.types.is_int64_dtype,
        "first_review": pd.api.types.is_object_dtype,
        "last_review": pd.api.types.is_object_dtype,
        "instant_bookable": pd.api.types.is_object_dtype,
        "calculated_host_listings_count": pd.api.types.is_int64_dtype,
        "calculated_host_listings_count_entire_homes": pd.api.types.is_int64_dtype,
        "calculated_host_listings_count_private_rooms": pd.api.types.is_int64_dtype,
        "calculated_host_listings_count_shared_rooms": pd.api.types.is_int64_dtype
    }

    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"


def test_class_names(data):

    # Check that only the known classes are present
    known_classes = [
        "f",
        "t"
    ]

    assert data["instant_bookable"].isin(known_classes).all()


def test_column_ranges(data):

    ranges = {
        "id": (10000, 54000000),
        "host_listings_count": (0.0, 500.484705e+06),
        "host_total_listings_count": (0.0, 670.0),
        "accommodates": (0, 30),
        "bedrooms": (1.0, 50.0),
        "beds": (1.0, 100.0)
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={data[col_name].min()} and max={data[col_name].max()}"
        )

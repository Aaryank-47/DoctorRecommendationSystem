import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder 
import re
# import xgboost as xgb # Removed XGBoost as it's not used for initial training with single class

class DoctorRecommender:
    def __init__(self, data_path):
        """
        Initialize the Doctor Recommender system.

        Args:
            data_path (str): Path to the doctor dataset CSV file
        """
        self.df = self.load_data(data_path)
        self.preprocess_data()
        # self.train_model() # Removed initial XGBoost training
        self.symptom_vectorizer = TfidfVectorizer(stop_words='english')
        self.fit_symptom_vectorizer()

    def load_data(self, data_path):
        """Load and preprocess the initial dataset"""
        # Ensure the file path is correct. It seems you intended to read an Excel file.
        # If the file is truly an Excel file, pd.read_excel is correct.
        # If it's a CSV with .xlsx extension, change to pd.read_csv.
        try:
            df = pd.read_excel('DoctorsDatasets.xlsx')
        except FileNotFoundError:
             print(f"Error: Dataset file not found at {data_path}")
             raise # Re-raise the exception if the file is not found
        except Exception as e:
             print(f"Error reading file {data_path}: {e}")
             raise # Re-raise other exceptions during file reading


        # Ensure required columns exist
        required_columns = ['Doctor_ID', 'Doctor_Name', 'Specialty', 'Location',
                          'Experience_Years', 'Rating', 'Availability',
                          'Symptoms_Handled', 'fees', 'email', 'phoneNumber', 'gender']
        assert all(col in df.columns for col in required_columns), f"Missing required columns: {list(set(required_columns) - set(df.columns))}"
        return df

    def preprocess_data(self):
        """Preprocess the data for modeling"""
        # Clean text data
        self.df['Symptoms_Handled'] = self.df['Symptoms_Handled'].apply(self.clean_text)
        self.df['Specialty'] = self.df['Specialty'].apply(self.clean_text)
        self.df['Location'] = self.df['Location'].apply(self.clean_text)
        self.df['gender'] = self.df['gender'].apply(self.clean_text)

        # The 'target' variable and encoding are kept for potential future use (e.g., training after user feedback)
        # Create binary target variable (will be modified based on user interactions)
        self.df['target'] = 1  # All doctors are initially considered good matches

        # Encode categorical features
        self.label_encoders = {}
        categorical_cols = ['Specialty', 'Location', 'gender']
        for col in categorical_cols:
            le = LabelEncoder()
            # Use .astype(str) before encoding to handle potential mixed types or NaNs gracefully
            self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le

        # Convert fees and experience to numeric, handling potential errors
        self.df['fees'] = pd.to_numeric(self.df['fees'], errors='coerce').fillna(0) # Fill NaNs with 0 or another appropriate value
        self.df['Experience_Years'] = pd.to_numeric(self.df['Experience_Years'], errors='coerce').fillna(0)
        self.df['Rating'] = pd.to_numeric(self.df['Rating'], errors='coerce').fillna(0)


    def clean_text(self, text):
        """Clean text data by removing special characters and lowercasing"""
        # Handle potential non-string types explicitly before applying lower() and regex
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.strip()


    def fit_symptom_vectorizer(self):
        """Fit TF-IDF vectorizer on symptoms"""
        # Ensure 'Symptoms_Handled' column is present before fitting
        if 'Symptoms_Handled' in self.df.columns:
            self.symptom_vectorizer.fit(self.df['Symptoms_Handled'])
        else:
             print("Warning: 'Symptoms_Handled' column not found. Symptom filtering will be disabled.")


    # Removed the train_model method as initial training with target=1 is not meaningful

    def apply_filters(self, df, symptoms=None, location=None,
                    min_fee=None, max_fee=None, gender=None, specialty=None):
        """
        Apply all filters to the dataframe sequentially.
        Returns filtered dataframe and any inferred specialty.
        """
        filtered_df = df.copy()
        inferred_specialty = None

        # 1. Filter by symptoms
        if symptoms and 'Symptoms_Handled' in filtered_df.columns and hasattr(self, 'symptom_vectorizer'):
            symptoms_clean = self.clean_text(symptoms)
            try:
                symptom_vec = self.symptom_vectorizer.transform([symptoms_clean])
                # Ensure 'Symptoms_Handled' is treated as string for TF-IDF transform
                doc_symptoms_vec = self.symptom_vectorizer.transform(filtered_df['Symptoms_Handled'].astype(str))

                similarity_scores = cosine_similarity(symptom_vec, doc_symptoms_vec).flatten()
                filtered_df['symptom_similarity'] = similarity_scores
                # Apply a threshold to filter out irrelevant doctors based on symptoms
                filtered_df = filtered_df[filtered_df['symptom_similarity'] > 0] # Keep only those with some similarity

                if len(filtered_df) > 0:
                    # Find the most common specialty among top symptom matches
                    if 'Specialty' in filtered_df.columns:
                        # Use the symptom_similarity to find relevant doctors, not just top N
                        top_matches = filtered_df[filtered_df['symptom_similarity'] > 0].copy()
                        if not top_matches.empty:
                            # Only consider the top symptom matches for inferring specialty
                            inferred_specialty_candidates = top_matches.sort_values('symptom_similarity', ascending=False)['Specialty'].head(10) # Consider top 10 similarity matches
                            if not inferred_specialty_candidates.empty:
                                inferred_specialty = inferred_specialty_candidates.mode()
                                if not inferred_specialty.empty:
                                    inferred_specialty = inferred_specialty[0]

            except Exception as e:
                print(f"Warning: Error during symptom filtering: {e}")
                # If symptom filtering fails, continue without it but warn the user
                if 'symptom_similarity' in filtered_df.columns:
                    filtered_df = filtered_df.drop(columns=['symptom_similarity'])


        # 2. Filter by gender
        if gender and 'gender' in filtered_df.columns:
            gender_clean = self.clean_text(gender)
            filtered_df = filtered_df[filtered_df['gender'] == gender_clean]


        # 3. Filter by location
        if location and 'Location' in filtered_df.columns:
            location_clean = self.clean_text(location)
            # Use .astype(str) to avoid errors with non-string types
            filtered_df = filtered_df[filtered_df['Location'].astype(str).str.contains(location_clean, case=False, na=False)] # na=False handles NaN values


        # 4. Filter by fee range
        if (min_fee is not None or max_fee is not None) and 'fees' in filtered_df.columns:
            if min_fee is None:
                min_fee = 0
            if max_fee is None:
                max_fee = float('inf')
            # Ensure 'fees' is numeric before filtering (handled in preprocess_data)
            filtered_df = filtered_df[
               (filtered_df['fees'] >= min_fee) &
               (filtered_df['fees'] <= max_fee)
            ]


        # 5. Filter by specialty (optional)
        if specialty and 'Specialty' in filtered_df.columns:
            specialty_clean = self.clean_text(specialty)
            # Use .astype(str) to avoid errors with non-string types
            filtered_df = filtered_df[filtered_df['Specialty'].astype(str).str.contains(specialty_clean, case=False, na=False)] # na=False handles NaN values

        # If no specific specialty was requested, but we inferred one from symptoms,
        # consider suggesting it or filtering by it if there are many results.
        # For simplicity here, we just print the suggestion.
        elif inferred_specialty and len(filtered_df) > 0:
            # If no specialty specified but we inferred one, inform user
            # Check if 'Specialty' column exists before printing inferred specialty
            if 'Specialty' in filtered_df.columns:
                 print(f"Note: Based on your symptoms, {inferred_specialty} specialists are highly relevant.")


        return filtered_df, inferred_specialty

    def recommend_doctors(self, symptoms=None, location=None,
                         min_fee=None, max_fee=None,
                         gender=None, specialty=None):
        """
        Recommend doctors based on multiple filters.
        Returns all matching doctors, ranked by relevance.

        Args:
            symptoms (str): Symptoms the patient is experiencing
            location (str): Preferred location for the doctor
            min_fee (float): Minimum acceptable fee
            max_fee (float): Maximum acceptable fee
            gender (str): Preferred doctor gender
            specialty (str): Preferred doctor specialty

        Returns:
            pd.DataFrame: DataFrame of recommended doctors, sorted by relevance
            str: Inferred specialty (if any)
        """
        # Apply all filters
        filtered_df, inferred_specialty = self.apply_filters(
            self.df.copy(), # Pass a copy to ensure filters don't modify the original self.df permanently
            symptoms, location, min_fee, max_fee, gender, specialty
        )


        if len(filtered_df) == 0:
            print("No doctors found matching all criteria.")
            return pd.DataFrame(), None  # Return empty if no matches

        # We are no longer using the XGBoost model for initial ranking.
        # The primary ranking mechanism is now symptom similarity, if symptoms were provided.
        # If symptoms were not provided, the results are ranked by default (e.g., index order, though sorting by rating or experience could be added).

        # Combine with symptom similarity if available for ranking
        if 'symptom_similarity' in filtered_df.columns:
            # Use symptom similarity as the primary ranking score
            filtered_df['combined_score'] = filtered_df['symptom_similarity']
        else:
            # If no symptoms, no specific ranking score from the model or similarity
            # You could add ranking by 'Rating' or 'Experience_Years' here if desired
            filtered_df['combined_score'] = filtered_df['Rating'] # Example: default sort by Rating


        # Sort by combined score (descending)
        # If 'combined_score' is not present (e.g., symptom vectorizer failed), sort by 'Rating' as a fallback
        if 'combined_score' in filtered_df.columns:
            result_df = filtered_df.sort_values('combined_score', ascending=False)
        elif 'Rating' in filtered_df.columns:
             print("Warning: No combined score calculated, sorting by Rating.")
             result_df = filtered_df.sort_values('Rating', ascending=False)
        else:
             print("Warning: No combined score or Rating, returning unfiltered (but filtered) order.")
             result_df = filtered_df # Return as is

        # Select output columns
        output_columns = [
            'Doctor_ID', 'Doctor_Name', 'Specialty', 'Location',
            'Experience_Years', 'Rating', 'Availability',
            'Symptoms_Handled', 'fees', 'gender', 'email', 'phoneNumber',
            # 'combined_score' # Remove combined_score from display unless needed
        ]

        # Ensure all output columns exist before selecting
        output_cols_exist = [col for col in output_columns if col in result_df.columns]

        # Add combined_score back only if it exists and we want to show it
        if 'combined_score' in result_df.columns:
             output_cols_exist.append('combined_score')
             # Ensure combined_score is the last column for display clarity if added
             output_cols_exist = sorted(output_cols_exist, key=lambda x: x != 'combined_score')


        return result_df[output_cols_exist], inferred_specialty

# Example usage
if __name__ == "__main__":
    # Initialize with your dataset path
    # Ensure the file path matches the actual file name and location
    # Double-check the file extension and format. If it's a CSV saved with .xlsx extension, use pd.read_csv
    # Example: recommender = DoctorRecommender("/content/doctorsDataset.csv")
    # Assuming the file is indeed an Excel file based on pd.read_excel in load_data
    data_file_path = "/content/DoctorsDatasets.xlsx" # Use the correct file name and extension used in load_data
    try:
        recommender = DoctorRecommender(data_file_path)

        # Example recommendation query
        recommendations, inferred_spec = recommender.recommend_doctors(
            symptoms="back pain and spine issues",
            location="Mumbai",
            min_fee=60000,
            max_fee=70800,
            gender="female"
        )

        print("\nRecommended Doctors:")
        if len(recommendations) > 0:
            # Check if 'combined_score' exists before deciding which columns to display
            # display_cols = [col for col in recommendations.columns if col != 'combined_score'] # Removed this as combined_score might be useful
            print(recommendations.to_string(index=False))
        else:
            print("No doctors found matching all your criteria.")

        # Check if inferred_spec is not None before trying to clean or print it
        # The logic here to only print if 'ortho' is NOT in the inferred specialty might be specific.
        # Consider printing the inferred specialty always if it exists.
        if inferred_spec is not None:
             cleaned_inferred_spec = recommender.clean_text(inferred_spec)
             # if 'ortho' not in cleaned_inferred_spec: # Remove or adjust this condition as needed
             print(f"\nBased on your symptoms, you might want to consider {inferred_spec} specialists.")
             # else:
             #     print("\nRecommended specialists match your symptoms.")

    except FileNotFoundError:
        print(f"Please ensure the dataset file '{data_file_path}' exists.")
    except AssertionError as e:
        print(f"Data loading failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
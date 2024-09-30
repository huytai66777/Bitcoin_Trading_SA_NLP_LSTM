import sys
import re
sys.path.append('./script')
import contractions

# Load all scraped pickle files and combine it to one dataframe
def combine_pickle_files(folder_name): 
    # Location of git folder
    git_folder_location = os.path.abspath(os.path.dirname('bitcoin_trader'))

    # list of pickled files
    pickle_list = os.listdir(git_folder_location+'/'+folder_name+'/')
    if '.DS_Store' in pickle_list:
        pickle_list.remove('.DS_Store')

    # Create a DataFrame to dump all individual DataFrames from scraped data
    with open(folder_name+'/'+pickle_list[0], 'rb') as picklefile: 
        df = pickle.load(picklefile)    
    df_merged = pd.DataFrame(columns=df.keys())

    for file in pickle_list:
        with open(folder_name+'/'+file, 'rb') as picklefile: 
            df = pickle.load(picklefile)
        df_merged = pd.concat([df_merged,df],ignore_index=True,axis=0)
    return df_merged


def expand_contractions(text):
    # Use the contractions library to fix contractions
    expanded_text = contractions.fix(text)
    return expanded_text


def preprocess_text(text):
    if text is not None:
        # Lowercasing
        text = text.lower()

        # Removing punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Lowercasing
        text = text.lower()

        # Removing hashtags (e.g., "#word")
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)

        # Removing URLs starting with "http"
        text = re.sub(r"http\S+", "", text)

        # Removing URLs starting with "www"
        text = re.sub(r"www.\S+", "", text)

        # Removing special characters and punctuation
        text = re.sub(r'[()!?]', ' ', text)

        # Removing non-alphanumeric characters
        text = re.sub(r'[^a-z0-9]', ' ', text)

        # Tokenization
        tokens = word_tokenize(text)

        # Removing stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)
    else:
        return ''  # Return empty string if text is None
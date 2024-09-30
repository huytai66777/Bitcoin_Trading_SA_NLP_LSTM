   
def custom_train_test_split(df, train_cols, date_gap, min_date, max_date, max_max_date):
    from sklearn.preprocessing import MinMaxScaler

    mask_train = (df['date'] > min_date) & (df['date'] < (max_date - timedelta(days=date_gap)))
    mask_test = (df['date'] > max_date) & (df['date'] < max_max_date)

    df_train = df[mask_train]
    df_test = df[mask_test]

    print(f"Train and Test size: {len(df_train)}, {len(df_test)}")

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(df_train[train_cols])
    x_test = scaler.transform(df_test[train_cols])
    
    return x_train, x_test, scaler

# TIME_STEPS is how many units back in time you want your network to see
def build_timeseries(mat, TIME_STEPS, y_col_index):
    dim_0 = mat.shape[0] - TIME_STEPS
    x = np.array([mat[i:TIME_STEPS + i] for i in range(dim_0)])
    y = mat[TIME_STEPS:, y_col_index]
    
    print(f"Length of time-series i/o: {x.shape}, {y.shape}")
    return x, y

# Small batch_size increase train time and too big batch size reduce model's ability to generalize, but quicker
def trim_dataset(mat, batch_size):
    """
    Trims dataset to a size divisible by batch_size.
    """
    return mat[:mat.shape[0] - (mat.shape[0] % batch_size)]

# Use of return_sequences
def create_model(lr):
    model = Sequential()
    model.add(LSTM(128, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), stateful=True, kernel_initializer='random_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = optimizers.RMSprop(learning_rate=lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model
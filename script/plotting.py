def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def set_plot_params():
    SMALL_SIZE = 15
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 40

    plt.rc('font', size=MEDIUM_SIZE)          
    plt.rc('axes', titlesize=MEDIUM_SIZE)     
    plt.rc('axes', labelsize=MEDIUM_SIZE)    
    plt.rc('xtick', labelsize=SMALL_SIZE)    
    plt.rc('ytick', labelsize=SMALL_SIZE)    
    plt.rc('legend', fontsize=SMALL_SIZE)    
    plt.rc('figure', titlesize=BIGGER_SIZE)  


def compute_sentiment_profit(sentiment_df, df_daily, positive_sentiment_threshold):
    sentiment_df['vader_buy_sell'] = sentiment_df['compound'].apply(lambda x: 1 if x > positive_sentiment_threshold else 0)
    sentiment_df['vader_profit'] = (sentiment_df['vader_buy_sell'] * 100 * sentiment_df['target_daily'])

    sentiment_df['txtblob_buy_sell'] = sentiment_df['txtblob'].apply(lambda x: 1 if x > positive_sentiment_threshold else 0)
    sentiment_df['txtblob_profit'] = (sentiment_df['txtblob_buy_sell'] * 100 * sentiment_df['target_daily'])

    sentiment_df['compound_buy_sell'] = sentiment_df['final_sentiment'].apply(lambda x: 1 if x > positive_sentiment_threshold else 0)
    sentiment_df['compound_profit'] = (sentiment_df['compound_buy_sell'] * 100 * sentiment_df['target_daily'])

    sentiment_df['compound_change_buy_sell'] = sentiment_df['daily_sentiment_change'].apply(lambda x: 1 if x > positive_sentiment_threshold else 0)
    sentiment_df['compound_change_profit'] = (sentiment_df['compound_change_buy_sell'] * 100 * sentiment_df['target_daily'])

    sentiment_df.dropna(inplace=True)

    return sentiment_df


def compute_cumulative_profit(sentiment_df, profit_columns):
    cumulative_profits = {}
    for column in profit_columns:
        cumulative_profits[column] = sentiment_df[column].cumsum()
    return cumulative_profits


def plot_sentiment_profit(sentiment_df, cumulative_profits, publisher, positive_sentiment_threshold):
    plt.figure(figsize=(20, 8))
    for label, profit in cumulative_profits.items():
        sns.lineplot(x=sentiment_df['date'], y=profit)

    plt.xlabel('Date')
    plt.ylabel('Profit ($)')
    plt.title(f'Cumulative Profit ({publisher}) - Pos Threshold = {positive_sentiment_threshold}')
    plt.legend(cumulative_profits.keys())
    plt.grid()
    plt.show()


def graph_sentiment_profit(publisher, positive_sentiment_threshold, min_date):
    set_plot_params()

    if publisher != 'all':
        asdf_df = df_news_sentiment[df_news_sentiment['date'] > min_date]
        sentiment_by_publisher = asdf_df.groupby(by=['date', 'publisher'], as_index=False).mean()

        sentiment_publisher_1 = sentiment_by_publisher[sentiment_by_publisher['publisher'] == publisher]
        sentiment_publisher_1 = sentiment_publisher_1.merge(df_daily[['date', 'daily_sentiment_change', 'target_daily']],
                                                             on='date')

        sentiment_publisher_1 = compute_sentiment_profit(sentiment_publisher_1, df_daily, positive_sentiment_threshold)
        
        cumulative_profits = compute_cumulative_profit(sentiment_publisher_1, 
                                                        ['vader_profit', 'txtblob_profit', 'compound_profit', 'compound_change_profit'])
        
        plot_sentiment_profit(sentiment_publisher_1, cumulative_profits, publisher, positive_sentiment_threshold)

    else:
        sentiment_by_publisher = df_news_sentiment.groupby(by=['date'], as_index=False).mean()
        sentiment_publisher_1 = sentiment_by_publisher[sentiment_by_publisher['date'] > min_date]
        sentiment_publisher_1 = sentiment_publisher_1.merge(df_daily[['date', 'daily_sentiment_change', 
                                                                     'wkly_sentiment_change', '2wk_sentiment_change', '4wk_sentiment_change', 
                                                                     'target_daily']], on='date')

        sentiment_publisher_1 = compute_sentiment_profit(sentiment_publisher_1, df_daily, positive_sentiment_threshold)

        cumulative_profits = compute_cumulative_profit(sentiment_publisher_1,
                                                        ['compound_profit', 'compound_change_profit', 
                                                         'compound_change_profit_wkly', 'compound_change_profit_2wk', 
                                                         'compound_change_profit_monthly'])

        plot_sentiment_profit(sentiment_publisher_1, cumulative_profits, 'All Publishers', positive_sentiment_threshold)


        
rom tqdm import trange

def best_threshold(min_threshold, max_threshold, step, min_date, max_date):
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 15, 20, 40

    # Graph formatting
    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    thresholds, raw_senti, senti_change_daily, senti_change_weekly, senti_change_2wk, senti_change_monthly = [], [], [], [], [], []
    threshold_list = np.arange(min_threshold, max_threshold, step)
    
    sentiment_by_publisher = df_news_sentiment.groupby('date', as_index=False).mean()
    mask = (sentiment_by_publisher['date'] > min_date) & (sentiment_by_publisher['date'] < max_date)
    
    sentiment_publisher_1 = sentiment_by_publisher[mask].merge(
        df_daily[['date', 'daily_sentiment_change', 'wkly_sentiment_change', '2wk_sentiment_change', '4wk_sentiment_change', 'target_daily']], 
        on='date'
    )
    
    for threshold in trange(len(threshold_list)):
        positive_sentiment_threshold = threshold_list[threshold]
        
        # Calculate buy/sell signals and profits using vectorized operations
        sentiment_publisher_1['compound_buy_sell'] = (sentiment_publisher_1['final_sentiment'] > positive_sentiment_threshold).astype(int)
        sentiment_publisher_1['compound_profit'] = sentiment_publisher_1['compound_buy_sell'] * 100 * sentiment_publisher_1['target_daily']
        
        # Calculate change-based profits
        sentiment_publisher_1['compound_change_buy_sell'] = (sentiment_publisher_1['daily_sentiment_change'] > positive_sentiment_threshold).astype(int)
        sentiment_publisher_1['compound_change_profit'] = sentiment_publisher_1['compound_change_buy_sell'] * 100 * sentiment_publisher_1['target_daily']

        sentiment_publisher_1['compound_change_buy_sell_wkly'] = (sentiment_publisher_1['wkly_sentiment_change'] > positive_sentiment_threshold).astype(int)
        sentiment_publisher_1['compound_change_profit_wkly'] = sentiment_publisher_1['compound_change_buy_sell_wkly'] * 100 * sentiment_publisher_1['target_daily']

        sentiment_publisher_1['compound_change_buy_sell_2wk'] = (sentiment_publisher_1['2wk_sentiment_change'] > positive_sentiment_threshold).astype(int)
        sentiment_publisher_1['compound_change_profit_2wk'] = sentiment_publisher_1['compound_change_buy_sell_2wk'] * 100 * sentiment_publisher_1['target_daily']

        sentiment_publisher_1['compound_change_buy_sell_4wk'] = (sentiment_publisher_1['4wk_sentiment_change'] > positive_sentiment_threshold).astype(int)
        sentiment_publisher_1['compound_change_profit_monthly'] = sentiment_publisher_1['compound_change_buy_sell_4wk'] * 100 * sentiment_publisher_1['target_daily']

        # Cumulative profit calculations
        cumulative_profit = sentiment_publisher_1[['compound_profit', 'compound_change_profit', 'compound_change_profit_wkly', 'compound_change_profit_2wk', 'compound_change_profit_monthly']].cumsum()

        thresholds.append(positive_sentiment_threshold)
        raw_senti.append(cumulative_profit['compound_profit'].iloc[-1])
        senti_change_daily.append(cumulative_profit['compound_change_profit'].iloc[-1])
        senti_change_weekly.append(cumulative_profit['compound_change_profit_wkly'].iloc[-1])
        senti_change_2wk.append(cumulative_profit['compound_change_profit_2wk'].iloc[-1])
        senti_change_monthly.append(cumulative_profit['compound_change_profit_monthly'].iloc[-1])

    return thresholds, raw_senti, senti_change_daily, senti_change_weekly, senti_change_2wk, senti_change_monthly


def best_threshold2(min_threshold, max_threshold, step, min_date, max_date):
    thresholds, senti_change_weekly, senti_change_2wk, senti_change_monthly = [], [], [], []
    threshold_list = np.arange(min_threshold, max_threshold, step)

    sentiment_by_publisher = df_news_sentiment.groupby('date', as_index=False).mean()
    sentiment_publisher_1 = sentiment_by_publisher.merge(
        df_daily[['date', 'wkly_sentiment_change', '2wk_sentiment_change', '4wk_sentiment_change', 'target_daily']], 
        on='date'
    )

    for threshold in range(len(threshold_list)):
        positive_sentiment_threshold = threshold_list[threshold]

        sentiment_publisher_1['compound_change_buy_sell_wkly'] = (sentiment_publisher_1['wkly_sentiment_change'] > positive_sentiment_threshold).astype(int)
        sentiment_publisher_1['compound_change_profit_wkly'] = sentiment_publisher_1['compound_change_buy_sell_wkly'] * 100 * sentiment_publisher_1['target_daily']

        sentiment_publisher_1['compound_change_buy_sell_2wk'] = (sentiment_publisher_1['2wk_sentiment_change'] > positive_sentiment_threshold).astype(int)
        sentiment_publisher_1['compound_change_profit_2wk'] = sentiment_publisher_1['compound_change_buy_sell_2wk'] * 100 * sentiment_publisher_1['target_daily']

        sentiment_publisher_1['compound_change_buy_sell_4wk'] = (sentiment_publisher_1['4wk_sentiment_change'] > positive_sentiment_threshold).astype(int)
        sentiment_publisher_1['compound_change_profit_monthly'] = sentiment_publisher_1['compound_change_buy_sell_4wk'] * 100 * sentiment_publisher_1['target_daily']

        cumulative_profit = sentiment_publisher_1[['compound_change_profit_wkly', 'compound_change_profit_2wk', 'compound_change_profit_monthly']].cumsum()

        thresholds.append(positive_sentiment_threshold)
        senti_change_weekly.append(cumulative_profit['compound_change_profit_wkly'].iloc[-1])
        senti_change_2wk.append(cumulative_profit['compound_change_profit_2wk'].iloc[-1])
        senti_change_monthly.append(cumulative_profit['compound_change_profit_monthly'].iloc[-1])

    return thresholds, senti_change_weekly, senti_change_2wk, senti_change_monthly


def threshold_vs_profit(thresholds, model_list):
    plt.figure(figsize=(20, 8))
    for model in model_list:
        sns.lineplot(x=thresholds, y=model)

    plt.xlabel("Threshold")
    plt.ylabel("Profit")
    plt.title("Threshold vs Profit")
    plt.legend(labels=["Raw Sentiment Profit", "Sentiment Change Daily Profit", "Sentiment Change Weekly Profit", "Sentiment Change Bi-weekly Profit", "Sentiment Change Monthly Profit"])
    plt.grid()
    plt.show()
    
    
 # Get profit based on model's predictions. Punish it if it predicted to buy but actually it was a loss. Else, no action = profit = 0.
def model_profit_graph(model_output, min_date, max_date):
    # Get Cumulative Profit for Generic Models       
    thresholds, senti_change_weekly, senti_change_2wk, senti_change_monthly = best_threshold2(0, 0.3, 0.002, min_date, max_date)
    best_threshold_weekly = thresholds[np.argmax(senti_change_weekly)]
    best_threshold_biweekly = thresholds[np.argmax(senti_change_2wk)]
    best_threshold_monthly = thresholds[np.argmax(senti_change_monthly)]
    
    # Filtering Data and Merging
    sentiment_by_publisher = df_news_sentiment[df_news_sentiment['date'] > max_date].groupby('date', as_index=False).mean()
    sentiment_publisher_1 = sentiment_by_publisher.merge(df_daily[['date', 'wkly_sentiment_change', '2wk_sentiment_change', '4wk_sentiment_change', 'target_daily']], on='date')

    # Define buy/sell and profit conditions
    def calc_buy_sell_profit(col, threshold):
        buy_sell = (col > threshold).astype(int)
        profit = buy_sell * 100 * sentiment_publisher_1['target_daily']
        return profit.cumsum()

    weekly_profit = calc_buy_sell_profit(sentiment_publisher_1['wkly_sentiment_change'], best_threshold_weekly)
    biweekly_profit = calc_buy_sell_profit(sentiment_publisher_1['2wk_sentiment_change'], best_threshold_biweekly)
    monthly_profit = calc_buy_sell_profit(sentiment_publisher_1['4wk_sentiment_change'], best_threshold_monthly)

    # Get Cumulative Profit for LightGBM
    df_test['pred'] = model_output   
    pred_profit = (df_test['pred'] > 0).astype(int) * df_test['target_daily'] * 100
    pred_profit = pred_profit.cumsum()

    # Calculate target profit
    df_test['target_daily_profit'] = df_test['target_daily'].apply(lambda x: 100 * x if x > 0 else 0)
    target_profit = df_test['target_daily_profit'].cumsum()

    print(f"Profit : ${pred_profit.iloc[-1]:.2f}")
    
    # Plot cumulative profit
    plt.figure(figsize=(20, 8))
    sns.lineplot(x=df_test['date'], y=target_profit, label='Actual')
    sns.lineplot(x=df_test['date'], y=pred_profit, label='Trading Bot')
    sns.lineplot(x=sentiment_publisher_1['date'], y=weekly_profit, label='Weekly Sentiment MA')
    sns.lineplot(x=sentiment_publisher_1['date'], y=biweekly_profit, label='Bi-weekly Sentiment MA')
    sns.lineplot(x=sentiment_publisher_1['date'], y=monthly_profit, label='Monthly Sentiment MA')
    plt.axhline(y=0, color='r', linestyle='-.')
    plt.xlabel('Date')
    plt.ylabel("Profit ($)")
    plt.title('Cumulative Profit')
    plt.legend()
    plt.grid()
    plt.show()
 
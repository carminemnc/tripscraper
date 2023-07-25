import altair as alt
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px

def rating_charts(data):
    
    ''' pie ratings chart'''
    src = data.groupby(['review_year', 'review_rating'],
                       as_index=False).agg(count=('review_date', 'count'))
    
    chart1 = alt.Chart(src).mark_bar().encode(
        x=alt.X('review_year',title='Year:'),
        y=alt.Y('count', title='Reviews Percentage:').stack('normalize'),
        color=alt.Color('review_rating',
                        legend=alt.Legend(
                            title='Rating'),title='Review Rating:')
    ).properties(width=650,height=300).interactive()

    ''' # of reviews by date '''
    src2 = data.groupby(['review_date', 'review_rating'],
                        as_index=False).agg(count=('review_date', 'count'))

    chart2 = alt.Chart(src2).mark_line(point=True).encode(
        alt.X('review_date:T', axis=alt.Axis(
            format="%b %y"), title='Date'),
        y=alt.Y('count', title='# of reviews'),
        color=alt.Color('review_rating',
                        legend=alt.Legend(
                            title='Review rating:'))
    ).properties(
        width=650,
        height=300
    ).interactive().configure_point(size=50)

    
    ''' reviews percentage per month '''

    src3 = data.groupby(['review_month', 'review_rating'],
                        as_index=False).agg(count=('review_date', 'count'))

    chart3 = alt.Chart(src3).mark_bar().encode(
        x=alt.X('count', title='Reviews percentage:').stack('normalize'),
        y=alt.Y('review_month', title='Month:'),
        color=alt.Color('review_rating',
                        legend=alt.Legend(
                            title='Review rating:'))
    ).properties(
        width=650,
        height=300
    ).interactive()

    return chart1, chart2, chart3

def words_network_graph(data, columnname):

    # Create dictionary with their counts
    d = data.set_index(columnname).T.to_dict('records')

    # Create network plot
    G = nx.Graph()

    # Create connections between nodes
    for k, v in d[0].items():
        G.add_edge(k[0], k[1], weight=(v*30))

    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(G, k=2)

    weights = nx.get_edge_attributes(G, 'weight').values()
    weights = list(weights)
    weights = [w*0.0060 for w in weights]

    # Plot networks
    nx.draw_networkx(G, pos,
                     font_size=12,
                     width=list(weights),
                     edge_color='white',
                     node_color='#008b8b',
                     with_labels=False,
                     ax=ax,
                     node_size=50)

    # Create offset labels
    for key, value in pos.items():
        x, y = value[0]+.00135, value[1]+.045
        ax.text(x, y,
                s=key,
                bbox=dict(facecolor='white', alpha=0.60),
                horizontalalignment='center', fontsize=9)
    fig.set_facecolor('#181616')
    plt.axis('off')
    return fig

def word_overtime_chart(data, word_list): 

    df = data.groupby('review_date', as_index=False)['tokens'].sum()
    
    for word in word_list:
        df[word] = df['tokens'].apply(lambda x: x.count(word)/len(x))
    
    word_list.append('review_date')
    df = df[word_list]
    df = df.melt(id_vars=['review_date'], var_name='words', value_name='value')

    c = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('review_date:T', axis=alt.Axis(
            format="%d-%b-%Y",
            labelOverlap=False,
            labelAngle=-45), title='Date'),
        y=alt.Y('value', title='frequency'),
        color=alt.Color('words',
                        legend=alt.Legend(
                            title='Words:'))).properties(
                                width=650,
                                height=300).interactive().configure_point(size=50)

    return c

def emotion_radar_chart(data):
    fig = px.line_polar(data, r='mean_score', theta='emotion', line_close=True)
    fig.update_traces(fill='toself')
    return fig

def barchart(data, x_axis, y_axis,title_name):

    c = alt.Chart(data).mark_bar().encode(
        x=alt.X(x_axis,axis=None),
        y=alt.Y(f'{y_axis}:N',title='').sort('-x'),
    ).properties(
        title=title_name,
        width=100,
        height=200
    ).interactive()

    return c

def words_sentiment_chart(data, word_list):

    data = data[word_list]
    df = data.melt().groupby(['variable', 'value'], as_index=False)[
        'value'].agg(['max', 'count']).reset_index().drop(columns='max')
    df['feeling'] = df['count'] / len(data)
    src = df
    c = alt.Chart(src).mark_arc(innerRadius=60).encode(
        theta='feeling',
        column=alt.Column('variable', title=''),
        color=alt.Color('value',
                        legend=alt.Legend(
                            title='Sentiment'),title='Sentiment'),
    ).properties(width=300, height=200).configure_header(labelColor='white',labelFontSize=14).interactive()

    return c

def words_sentiment_over_time(data, word):

    src = data.groupby(['review_date', 'review_short_date', word], as_index=False)[
        'review_body'].count().sort_values('review_date', ascending=True).drop(columns='review_date')

    c = alt.Chart(src).mark_bar().encode(
        x=alt.X('review_short_date',
                sort=src['review_short_date'].to_list(), title='Date'),
        y=alt.Y('review_body', title='# of reviews').stack('normalize'),
        color=alt.Color(word,
                        legend=alt.Legend(
                            title='Sentiment')),
        tooltip=[alt.Tooltip(word, title='Sentiment:'),
                 alt.Tooltip('review_body', title='# of reviews')
                 ]
    ).properties(title=f'{word} sentiment trend',width=500,height=200).interactive()
    
    return c
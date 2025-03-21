import altair as alt
import matplotlib

# matplotlib.use('nbagg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def multi_line_chart(df, width, height):
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['published_at'], empty='none')
    brush = alt.selection(type='interval', encodings=['x'])
    # The basic line
    line = alt.Chart(df).mark_line(interpolate='basis').encode(
        x='published_at:T',
        y='count:Q',
        color='feed:N'
    )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='published_at:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'count:Q', alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='published_at:T',
    ).transform_filter(
        nearest
    )
    # Put the five layers into a chart and bind the data
    chart = alt.layer(
        line, selectors, points, rules, text
    ).add_selection(brush)
    # make legend max width bigger
    chart = chart.configure_legend(
        labelLimit=400
    )
    chart = chart.properties(width=width, height=height)

    return chart


def plot_windows_of_interest(news_df, interesting_windows, weights):
    cmap = matplotlib.cm.cool

    x = [mdates.datestr2num(d) for d in list(news_df.index.array)]
    y = news_df['count']

    fig, ax = plt.subplots()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
    ax.plot(x, y)

    # note wierd string formatting to cut off `+00:00` from datestrings
    for (start, end), weight in zip(interesting_windows, weights):
        ax.axvspan(mdates.datestr2num(start.isoformat().split('+')[0] + 'Z'),
                   mdates.datestr2num(end.isoformat().split('+')[0] + 'Z'),
                   color=cmap(weight), alpha=0.3)

    plt.gcf().autofmt_xdate()
    plt.show()

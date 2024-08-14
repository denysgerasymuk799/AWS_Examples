from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql import types as t
from pyspark.sql.window import Window as w


spark = SparkSession.builder\
    .appName("Big_Data")\
    .getOrCreate()


def df_print(df, n_rows=5):
    return df.limit(n_rows).toPandas().head(n_rows)


def convert_df_to_column_array(old_df, from_col, to_col):
    new_df = (
        old_df
            .agg(F.collect_list(col(from_col)).alias(to_col))
            .select([to_col])
    )
    return new_df


# ## Query 1
# 
# **Description:** Find Top 10 videos that were amongst the trending videos for the highest
# number of days (it doesn't need to be a consecutive period of time).
# You should also include information about different metrics for each day
# the video was trending.


def get_top_trending_videos(videos_df, for_query_6=False):
    # Create initial dataframe for the query
    df = videos_df.select(
        col('video_id'),
        col("title"),
        col("description\r").alias("description"),
        F.struct(
            F.to_date(F.from_unixtime(F.unix_timestamp(col("trending_date"), "yy.dd.MM"))).alias("date"),
            col("likes").cast(t.LongType()).alias("likes"),
            col("dislikes").cast(t.LongType()).alias("dislikes"),
            col("views").cast(t.LongType()).alias("views")
        ).alias('trending_day')
    )

    # Get top 10 most trending video ids
    most_trending_df = df.groupBy(df.video_id)\
        .agg(F.count('video_id').alias('num_trending_days'))\
        .sort(col('num_trending_days').desc()).limit(10)

    # Get title, description and trending_day for each video from top 10
    detailed_most_trending_df = most_trending_df.alias('df1').join(
        df.alias('df2'), col('df1.video_id') == col('df2.video_id'), 'inner')\
            .select(
                col('df1.video_id').alias('video_id'),
                col('num_trending_days'),
                col('title'),
                col('description'),
                col('trending_day')
            )
    # Return enough information for query 6, avoiding unnecessary computations
    if for_query_6:
        return detailed_most_trending_df

    # Find the latest day video statistics
    window = w.partitionBy("video_id").orderBy(col("trending_day.date").desc())
    latest_day_info_df = detailed_most_trending_df.withColumn("rank", F.row_number().over(window))\
        .where(col("rank") == 1)\
        .select(
                col("video_id"),
                col("title"),
                col("description"),
                col("trending_day.likes").alias("latest_likes"),
                col("trending_day.dislikes").alias("latest_dislikes"),
                col("trending_day.views").alias("latest_views")
            )

    # Get all trending days for top 10 videos
    trending_days_df = latest_day_info_df.alias('df1')\
                                .join(df.alias('df2'), col('df1.video_id') == col('df2.video_id'), how='inner')\
                                .select(
                                    col('df1.video_id').alias('video_id'),
                                    col('trending_day')
                                )
    trending_days_df = trending_days_df.groupBy("video_id")\
        .agg(F.collect_list(col("trending_day")).alias("trending_days"))

    # Join and get final results
    final_df = latest_day_info_df.alias('df1')\
                            .join(trending_days_df.alias('df2'), col('df1.video_id') == col('df2.video_id'))\
                            .select(
                                col('df1.video_id').alias('video_id'),
                                col('title'),
                                col('description'),
                                col('latest_views'),
                                col('latest_likes'),
                                col('latest_dislikes'),
                                col('trending_days')
                            )

    top_videos_df = final_df.select(
                                F.struct(
                                    col("video_id"),
                                    col('title'),
                                    col('description'),
                                    col('latest_views'),
                                    col('latest_likes'),
                                    col('latest_dislikes'),
                                    col('trending_days')
                                ).alias("video")
                            )

    q1_videos_df = convert_df_to_column_array(top_videos_df, from_col="video", to_col="videos")
    return q1_videos_df


# ## Query 2
# 
# **Description:** Find what was the most popular category for each week (7 days slices).
# Popularity is decided based on the total number of views for videos of
# this category. Note, to calculate it you can’t just sum up the number of views.
# If a particular video appeared only once during the given period, it shouldn’t be
# counted. Only if it appeared more than once you should count the number of new
# views. For example, if video A appeared on day 1 with 100 views, then on day 4
# with 250 views and again on day 6 with 400 views, you should count it as 400 - 100 = 300.
# For our purpose, it will mean that this particular video was watched 300 times
# in the given time period.


week_dates_schema = t.StructType([
    t.StructField("start_date", t.DateType()),
    t.StructField("end_date", t.DateType())
])


@F.udf(week_dates_schema)
def get_week_dates_udf(trending_date):
    date_dt = datetime.strptime(str(trending_date), "%Y-%m-%d")
    start_date = date_dt - timedelta(days=date_dt.weekday())
    end_date = start_date + timedelta(days=6)
    return start_date, end_date


def get_most_popular_categories(videos_df, video_categories_df):
    # Split video ids on 7-days chunks
    df = videos_df.select(
            col("video_id"),
            F.to_date(F.from_unixtime(F.unix_timestamp(col("trending_date"), "yy.dd.MM"))).alias("trending_date"),
            col("views").cast(t.LongType()).alias("views"),
            col("category_id")
        )

    chunks_df = df.withColumn("week_dates", get_week_dates_udf(col("trending_date")))\
                    .select(
                        col("video_id"),
                        col("views"),
                        col("category_id"),
                        col("trending_date"),
                        col("week_dates.start_date").alias("start_date"),
                        col("week_dates.end_date").alias("end_date"))

    # Count number of video appearance during this 7 days.
    # And filter video ids, which appeared more than once in its period chunk
    weekly_video_views_df = (
        chunks_df
            .groupBy("video_id", "category_id", "start_date", "end_date")
            .agg(
                F.count("video_id").alias("video_count"),
                F.collect_list(col("views")).alias("views_lst")
            )
            .where(col("video_count") >= 2)
            .select(
                col("video_id"),
                col("start_date"),
                col("end_date"),
                col("category_id"),
                col("views_lst")
            )
    )

    # Get views_delta for videos, which relates to the same period chunk and category, sum it
    # and find the most popular category for each week
    window = w.partitionBy("start_date", "end_date").orderBy(col("total_views").desc())
    weekly_most_popular_categories_df = (
        weekly_video_views_df
            .withColumn("views_delta", F.array_max(col("views_lst")) - F.array_min(col("views_lst")))
            .groupBy("start_date", "end_date", "category_id")
            .agg(
                F.count("video_id").alias("number_of_videos"),
                F.sum(col("views_delta")).alias("total_views"),
                F.collect_list(col("video_id")).alias("video_ids_lst")
            )
            .withColumn("rank", F.row_number().over(window))
            .where(col("rank") == 1)
    )

    # Join to get category titles
    final_df = (
        weekly_most_popular_categories_df.alias("df1")
            .join(video_categories_df.alias("df2"), col("df1.category_id") == col("df2.id"), how="left")
            .sort(col("df1.start_date").desc())
            .select(
                col("df1.start_date"),
                col("df1.end_date"),
                col("df1.category_id"),
                col("df2.title").alias("category_name"),
                col("df1.number_of_videos"),
                col("df1.total_views"),
                col("df1.video_ids_lst").alias("video_ids")
            )
    )

    final_df = final_df.select(
                                F.struct(
                                    col("start_date"),
                                    col("end_date"),
                                    col("category_id"),
                                    col("category_name"),
                                    col("number_of_videos"),
                                    col("total_views"),
                                    col("video_ids")
                                ).alias("week")
                            )

    q2_weeks_df = convert_df_to_column_array(final_df, from_col="week", to_col="weeks")
    return q2_weeks_df


# ## Query 3
# 
# **Description:** What were the 10 most used tags amongst trending videos for each 30days time period?
# Note, if during the specified period the same video appears multiple times,
# you should count tags related to that video only once.

def get_most_used_tags(videos_df):
    # Create 30-days windows and filter periods by distinct video_ids
    df = (
        videos_df
            .withColumn('tags', F.regexp_replace(col("tags"), '\"', ''))
            .withColumn('trending_date',
                        F.to_date(F.from_unixtime(F.unix_timestamp(col("trending_date"), "yy.dd.MM"))) )
            .withColumn("30_days_period", F.window("trending_date", "30 days"))
            .select(
                col('video_id'),
                col('tags'),
                col('30_days_period.start').alias('start_date'),
                col('30_days_period.end').alias('end_date')
            ).distinct()
    )

    # Check if window function worked correctly
    df.select(['start_date', 'end_date']).distinct().sort(col('start_date')).show()

    # Group by 30-days periods and count number of tags and video_ids for each period
    period_df = (
          df
            .withColumn('tags_array', F.split(col('tags'), "\|"))
            .withColumn('tag', F.explode(col('tags_array')))
            .groupBy('tag', 'start_date', 'end_date')
            .agg(
                F.count(col('video_id')).alias('number_of_videos'),
                F.collect_list(col('video_id')).alias('video_ids')
            )
            .select(['tag', 'number_of_videos', 'video_ids', 'start_date', 'end_date'])
    )

    # Sort number_of_videos in descending order. Take top 10 most used tags
    window = w.partitionBy("start_date", "end_date").orderBy(col("number_of_videos").desc())
    top_tags_df = (
        period_df
                .withColumn("rank", F.row_number().over(window))
                .where(col("rank") <= 10)
                .select(
                    col("start_date"),
                    col("end_date"),
                    F.struct(
                        col("tag"),
                        col("number_of_videos"),
                        col("video_ids")
                    ).alias("tag_stat")
                )
        )

    # Transform to month's schema
    month_df = (
        top_tags_df
            .groupBy("start_date", "end_date")
            .agg(F.collect_list(col("tag_stat")).alias("tags"))
            .select(F.struct(
                col("start_date"),
                col("end_date"),
                col("tags")
            ).alias("month"))
    )

    # Transform to months' schema
    months_df = convert_df_to_column_array(month_df, from_col="month", to_col="months")
    return months_df


# ## Query 4
# 
# **Description:** Show the top 20 channels by the number of views for the whole period.
# Note, if there are multiple appearances of the same video for some channel,
# you should take into account only the last appearance (with the highest
# number of views).

def get_top_channels_by_number_of_views(videos_df):
    # Create initial dataframe for the query
    df = videos_df\
            .withColumn("trending_date",
                        F.to_date(F.from_unixtime(F.unix_timestamp(col("trending_date"), "yy.dd.MM"))))\
            .select(
                col("channel_title"),
                col("video_id"),
                col("trending_date"),
                col("views").cast(t.LongType()).alias("views")
            )

    # Group by channel_title and video_id. Take video with max number of views for the same channel
    window = w.partitionBy("channel_title", "video_id").orderBy(col("views").desc())
    filtered_videos_df = (
          df
            .withColumn("rank", F.row_number().over(window))
            .where(col("rank") == 1)
    )

    # Group by channel_title to get a number of views for all popular videos.
    # Also get top 20 channels by the number of views for the whole period
    most_popular_channels = (
        filtered_videos_df
            .withColumn("video_stat",
                        F.struct(
                           col("video_id"),
                           col("views")
                       ))
            .groupBy("channel_title")
            .agg(
                F.sum(col("views")).alias("total_views"),
                F.min("trending_date").alias("start_date"),
                F.max("trending_date").alias("end_date"),
                F.collect_list(col("video_stat")).alias("videos_views")
            )
            .orderBy(col("total_views").desc())
            .select(
                F.struct(
                    col("channel_title").alias("channel_name"),
                    col("start_date"),
                    col("end_date"),
                    col("total_views"),
                    col("videos_views")
                ).alias("channel")
            )
            .limit(20)
    )

    q4_channels_df = convert_df_to_column_array(most_popular_channels, from_col="channel", to_col="channels")
    return q4_channels_df


# ## Query 5
# 
# **Description:** Show the top 10 channels with videos trending for the highest number of days
# (it doesn't need to be a consecutive period of time) for the whole period.
# In order to calculate it, you may use the results from the question No1.
# The total_trending_days count will be a sum of the numbers of trending days
# for videos from this channel.

def get_top_channels_by_trending_days(videos_df):
    detailed_most_trending_df = get_top_trending_videos(videos_df, for_query_6=True)

    # Use detailed_most_trending_df, which was created for query 1
    video_id_to_channel_df = videos_df.select(['video_id', 'channel_title']).distinct()

    # Get channel names for each video_id
    full_detailed_most_trending_df = (
        detailed_most_trending_df.alias('df1')
            .join(video_id_to_channel_df.alias('df2'),
                    col("df1.video_id") == col("df2.video_id"), how='inner')
            .select(
                col("channel_title").alias("channel_name"),
                F.struct(
                    col("df1.video_id").alias("video_id"),
                    col("df1.title").alias("video_title"),
                    col("df1.num_trending_days").alias("trending_days")
                ).alias("video_day")
            )
    )

    # Find top 10 channels with videos trending for the highest number of days
    top_channels_df = (
        full_detailed_most_trending_df
            .groupBy("channel_name")
            .agg(
                F.collect_list(col("video_day")).alias("videos_days"),
                F.sum(col("video_day.trending_days")).alias("total_trending_days")
            )
            .orderBy(col("total_trending_days").desc())
            .select(
                F.struct(
                    col("channel_name"),
                    col("total_trending_days"),
                    col("videos_days")
                ).alias("channel")
            ).limit(10)
    )

    q5_channels_df = convert_df_to_column_array(top_channels_df, from_col="channel", to_col="channels")
    return q5_channels_df


# ## Query 6
# 
# **Description:** Show the top 10 videos by the ratio of likes/dislikes for each category
# for the whole period. You should consider only videos with more than 100K views.
# If the same video occurs multiple times you should take the record when
# the ratio was the highest.

def get_top_category_videos_by_ratio_likes_dislikes(videos_df, video_categories_df):
    # Create initial dataframe for the query. Filter videos, which have more than 100K views
    df = (
        videos_df
            .where(col("views") > 100_000)
            .select(
                col('video_id'),
                col("category_id"),
                col("title").alias("video_title"),
                col("views").cast(t.LongType()).alias("views"),
                col("likes").cast(t.LongType()).alias("likes"),
                col("dislikes").cast(t.LongType()).alias("dislikes")
            )
    )

    # Count the ratio of likes/dislikes for each video.
    # If the same video occurs multiple times you should take the record when the ratio was the highest
    video_ratio_window = w.partitionBy("video_id").orderBy(col("ratio_likes_dislikes").desc())
    video_ratios_df = (
           df
            .withColumn("ratio_likes_dislikes", col("likes") / col("dislikes"))
            .withColumn("video_ratio_rank", F.row_number().over(video_ratio_window))
            .where(col("video_ratio_rank") == 1)
    )

    # Group by category and get top 10 videos by the ratio of likes/dislikes for each category
    category_ratio_window = w.partitionBy("category_id").orderBy(col("ratio_likes_dislikes").desc())
    category_ratios_df = (
        video_ratios_df.alias("df1")
            .withColumn("category_ratio_rank", F.row_number().over(category_ratio_window))
            .where(col("category_ratio_rank") <= 10)
            .join(video_categories_df.alias("df2"),
                 col("df1.category_id") == col("df2.id"))
            .select(
                col("category_id"),
                col("df2.title").alias("category_name"),
                F.struct(
                    col("video_id"),
                    col("video_title"),
                    col("ratio_likes_dislikes"),
                    col("views")
                ).alias("video")
            )
    )

    top_category_videos_df = (
        category_ratios_df
            .groupBy("category_id", "category_name")
            .agg(
                F.collect_list(col("video")).alias("videos")
            )
            .select(
                F.struct(
                    col("category_id"),
                    col("category_name"),
                    col("videos")
                ).alias("category")
            )
    )

    q6_category_videos_df = convert_df_to_column_array(top_category_videos_df, from_col="category", to_col="categories")
    return q6_category_videos_df


if __name__ == '__main__':
    print("=" * 14, "Start pyspark program", "=" * 14)

    # Explore data
    video_categories_df = spark.read.format("json") \
        .option("multiline", "true") \
        .load("s3://big-data-labs-2022/data/GB_category_id.json")
    print('video_categories_df schema: ')
    video_categories_df.printSchema()

    video_categories_df = (
        video_categories_df
            .withColumn("categories", F.explode(F.arrays_zip("items.id", "items.snippet.title")))
            .select(
            col("categories.0").alias("id"),
            col("categories.1").alias("title"),
        )
    )
    print('Show video_categories_df:')
    video_categories_df.show()

    videos_df = spark.read.format("csv") \
        .option("multiline", True) \
        .option("sep", ",") \
        .option("header", True) \
        .load("s3://big-data-labs-2022/data/GBvideos.csv")
    print('Show videos_df:')
    videos_df.show()

    # ===================== QUERY 1 =====================
    top_trending_videos_df = get_top_trending_videos(videos_df)
    print("=" * 14, "QUERY 1", "=" * 14)
    print('top_trending_videos_df schema: ')
    top_trending_videos_df.printSchema()
    print('top_trending_videos_df show: ')
    top_trending_videos_df.show()
    top_trending_videos_df.write.json('s3://big-data-labs-2022/results/top_trending_videos.json')

    # ===================== QUERY 2 =====================
    most_popular_categories_df = get_most_popular_categories(videos_df, video_categories_df)
    print("=" * 14, "QUERY 2", "=" * 14)
    print('most_popular_categories_df schema: ')
    most_popular_categories_df.printSchema()
    print('most_popular_categories_df show: ')
    most_popular_categories_df.show()
    most_popular_categories_df.write.json('s3://big-data-labs-2022/results/most_popular_categories.json')

    # ===================== QUERY 3 =====================
    most_used_tags_df = get_most_used_tags(videos_df)
    print("=" * 14, "QUERY 3", "=" * 14)
    print('most_used_tags_df schema: ')
    most_used_tags_df.printSchema()
    print('most_used_tags_df show: ')
    most_used_tags_df.show()
    most_used_tags_df.write.json('s3://big-data-labs-2022/results/most_used_tags.json')

    # ===================== QUERY 4 =====================
    top_channels_by_number_of_views_df = get_top_channels_by_number_of_views(videos_df)
    print("=" * 14, "QUERY 4", "=" * 14)
    print('top_channels_by_number_of_views_df schema: ')
    top_channels_by_number_of_views_df.printSchema()
    print('top_channels_by_number_of_views_df show: ')
    top_channels_by_number_of_views_df.show()
    top_channels_by_number_of_views_df.write.json('s3://big-data-labs-2022/results/top_channels_by_number_of_views.json')

    # ===================== QUERY 5 =====================
    top_channels_by_trending_days_df = get_top_channels_by_trending_days(videos_df)
    print("=" * 14, "QUERY 5", "=" * 14)
    print('top_channels_by_trending_days_df schema: ')
    top_channels_by_trending_days_df.printSchema()
    print('top_channels_by_trending_days_df show: ')
    top_channels_by_trending_days_df.show()
    top_channels_by_trending_days_df.write.json('s3://big-data-labs-2022/results/top_channels_by_trending_days.json')

    # ===================== QUERY 6 =====================
    top_category_videos_by_ratio_likes_dislikes_df = get_top_category_videos_by_ratio_likes_dislikes(videos_df,
                                                                                                     video_categories_df)
    print("=" * 14, "QUERY 6", "=" * 14)
    print('top_category_videos_by_ratio_likes_dislikes_df schema: ')
    top_category_videos_by_ratio_likes_dislikes_df.printSchema()
    print('top_category_videos_by_ratio_likes_dislikes_df show: ')
    top_category_videos_by_ratio_likes_dislikes_df.show()
    top_category_videos_by_ratio_likes_dislikes_df.write.json('s3://big-data-labs-2022/results/top_category_videos_by_ratio_likes_dislikes.json')

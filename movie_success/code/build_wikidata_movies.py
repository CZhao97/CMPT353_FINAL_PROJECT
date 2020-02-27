import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder \
    .appName('wikidata movies extraction') \
    .getOrCreate()

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.1' # make sure we have Spark 2.1+

import datetime
ISOFMT = '+%Y-%m-%dT%H:%M:%SZ'


def extract_isodate(d):
    """
    Handle the slightly crazy date format from wikidata.
    """
    try:
        return datetime.datetime.strptime(d, ISOFMT).date()
    except ValueError:
        try:
            # days == '00' aren't acceptable. Adapt.
            d0 = d[:9] + '01' + d[11:]
            return datetime.datetime.strptime(d0, ISOFMT).date()
        except ValueError:
            # ... same for month == '00'
            d1 = d0[:6] + '01' + d0[8:]
            return datetime.datetime.strptime(d1, ISOFMT).date()


@functions.udf(returnType=types.DateType())
def first_publication_date(pds):
    """
    Get the earliest publication date, deciphering the wikidata date format along the way.
    """
    if pds is None:
        return None
    dates = map(extract_isodate, pds)
    return min(dates)


def take_first(wd, col):
    """
    For list columns are generally just one item: extract it.
    """
    return wd[col].getItem(0).alias(col)


def make_label_map():
    """
    Create a full list of wikidata_id values to human-readable labels.

    Off by default: creates about 1GB json.gz output.

    Could probably trim the output significantly by taking only values that appear in the wikidata_movies output.
    """
    wd = spark.read.parquet(sys.argv[1])
    label_map = wd.filter(wd['label'].isNotNull()).select(wd['id'].alias('wikidata_id'), wd['label'])
    label_map.repartition(10).write.json('./label_map', mode='overwrite', compression='gzip')


def make_wikidata_movies():
    wd = spark.read.parquet(sys.argv[1])

    # just movies
    wd = wd.where(wd['imdb_id'].isNotNull()).where(functions.substring(wd['imdb_id'].getItem(0), 0, 2) == 'tt')

    # ... with reasonable depth of data
    wd = wd.where(wd['genre'].isNotNull())
    wd = wd.where(wd['enwiki_title'].isNotNull())
    wd = wd.where(wd['rotten_tomatoes_id'].isNotNull())

    # calculate made_profit boolean
    wd = wd.withColumn('nbox', wd.box_office.getItem(0).getItem(0).cast(types.FloatType()))
    wd = wd.withColumn('ncost', wd.cost.getItem(0).getItem(0).cast(types.FloatType()))
    wd = wd.withColumn('made_profit', (wd.nbox - wd.ncost) > 0)

    # actual fields we want to output: other fields could be included and/or modified here.
    output_data = wd.select(
        wd['id'].alias('wikidata_id'),
        wd['label'],

        take_first(wd, 'imdb_id'),
        take_first(wd, 'rotten_tomatoes_id'),
        take_first(wd, 'metacritic_id'),
        wd['enwiki_title'],

        wd['genre'],
        wd['main_subject'],
        wd['filming_location'],
        wd['director'],
        #wd['screenwriter'],
        wd['cast_member'],
        #wd['narrative_location'],
        #wd['director_of_photography'],
        #wd['film_editor'],
        #wd['filming_location'],
        take_first(wd, 'series'),
        #wd['voice_actor'],
        #wd['executive_producer'],
        #wd['composer'],
        take_first(wd, 'production_company'),
        #take_first(wd, 'distributor'),
        first_publication_date(wd['publication_date']).alias('publication_date'),
        take_first(wd, 'based_on'),
        take_first(wd, 'country_of_origin'),
        take_first(wd, 'original_language'),
        wd['made_profit'],
        wd['nbox'],
        wd['ncost']
    )
    # output is about 4MB compressed: safe to .coalesce().
    output_data.coalesce(1).write.json('./wikidata-movies', mode='overwrite', compression='gzip')


def make_genre_map():
    """
    Make mapping of genre wikidata_id value to numan-readable label.
    """
    wd = spark.read.parquet(sys.argv[1])
    label_map = wd.filter(wd.label.isNotNull()).select(wd['id'].alias('wikidata_id'), wd['label'])
    genres = wd.select(functions.explode(wd['genre']).alias('wikidata_id')).distinct()
    genres = functions.broadcast(genres) # only a few thousand values that we want to keep
    genres = genres.join(label_map, on='wikidata_id')
    genres = genres.withColumnRenamed('label', 'genre_label')
    # output is about <1MB compressed: safe to .coalesce().
    genres.coalesce(1).write.json('./genres', mode='overwrite', compression='gzip')


if __name__ == "__main__":
    make_wikidata_movies()
    make_genre_map()
    make_label_map()

import sys
from pyspark.sql import SparkSession, functions, types
import json

spark = SparkSession.builder \
    .appName('wikidata data extractor') \
    .getOrCreate()

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.1' # make sure we have Spark 2.1+

sc = spark.sparkContext

interesting_claims = {
    'P31': 'instance_of',
    'P279': 'subclass_of',
    'P1476': 'title',
    'P921': 'main_subject',
    'P136': 'genre',
    'P495': 'country_of_origin',
    'P577': 'publication_date',
    'P57': 'director',
    'P58': 'screenwriter',
    'P161': 'cast_member',
    'P2142': 'box_office',
    'P2130': 'cost',
    'P345': 'imdb_id',
    'P1258': 'rotten_tomatoes_id',
    'P1712': 'metacritic_id',
    'P3302': 'omdb_id',
    'P144': 'based_on',
    'P364': 'original_language',
    'P840': 'narrative_location',
    'P344': 'director_of_photography',
    'P1040': 'film_editor',
    'P915': 'filming_location',
    'P179': 'series',
    'P725': 'voice_actor',
    'P1431': 'executive_producer',
    'P86': 'composer',
    'P272': 'production_company',
    'P750': 'distributor',
}


def no_trailing_comma(l):
    """
    By-lines wikidata dumps have trailing commas (on all but the last line). Remove them.
    """
    if l[-1] == ',':
        return l[:-1]
    else:
        return l


def to_useful_data(e):
    data = {'id': e['id']}

    try:
        data['label'] = e['labels']['en']['value']
    except KeyError:
        pass

    try:
        data['enwiki_title'] = e['sitelinks']['enwiki']['title']
    except KeyError:
        pass

    for p in e['claims'].keys() & interesting_claims:
        vals = []
        for c in e['claims'][p]:
            if 'datavalue' not in c['mainsnak']:
                continue

            value = c['mainsnak']['datavalue']['value']
            if isinstance(value, str):
                vals.append(value)
            elif 'id' in value:
                vals.append(value['id'])
            elif 'text' in value:
                vals.append(value['text'])
            elif 'time' in value:
                vals.append(value['time'])
            elif 'amount' in value:
                vals.append((value['amount'], value['unit'].split('/')[-1]))
            else:
                raise ValueError('unknown value: %r' % (value,))

        if vals:
            data[interesting_claims[p]] = vals

    return data


def main():
    lines = sc.textFile(sys.argv[1])
    lines = lines.filter(lambda l: len(l) > 1) # throw away initial '[' and ']' lines
    lines = lines.map(no_trailing_comma)
    entities = lines.map(json.loads)

    useful_data = entities.map(to_useful_data)
    useful_data = useful_data.map(json.dumps)
    # output as by-line JSON data that Spark SQL can deal with.
    useful_data.saveAsTextFile(sys.argv[2], compressionCodecClass='org.apache.hadoop.io.compress.GzipCodec')

    # read it back and make the parquet version of the data while we're here...
    wd = spark.read.json(sys.argv[2])
    wd.write.parquet(sys.argv[2] + '-parquet', mode='overwrite', compression='gzip')


main()
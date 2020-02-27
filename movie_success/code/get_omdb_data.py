import requests
import json, glob, dbm, sys
import pandas as pd

API_KEY = sys.argv[1]
cachefile = 'omdb-cache.dbm'
NOT_FOUND = 'notfound'
request_limit = False


def get_omdb_data(imdb_id):
    """
    Fetch this IMDb ID from the OMDb API.
    """
    global request_limit
    if request_limit:
        # after we hit the limit, stop trying.
        return None
    if not imdb_id.startswith('tt'):
        raise ValueError('movies only')

    url = 'http://www.omdbapi.com/?i=%s&apikey=%s&plot=full' % (imdb_id, API_KEY)
    print('fetching', url)
    r = requests.get(url)

    data = json.loads(r.text)
    if data['Response'] == 'False':
        if data['Error'] == 'Error getting data.':
            return NOT_FOUND
        elif data['Error'] == 'Request limit reached!':
            request_limit = True
            return None
        else:
            raise ValueError(data['Error'])

    return data


def get_omdb_data_cache(imdb_id):
    """
    Get OMDb data, but cache in cachefile, so we don't hammer the same URL multiple times.
    """
    db = dbm.open(cachefile, 'cs')
    try:
        data = json.loads(db[imdb_id])

    except KeyError:
        data = get_omdb_data(imdb_id)
        if data is not None:
            db[imdb_id] = json.dumps(data)

    if data == NOT_FOUND:
        return None

    return data


def main():
    infile = glob.glob('./wikidata-movies/part*')[0]
    #infile = './wikidata-movies.json.gz'
    movie_data = pd.read_json(infile, orient='records', lines=True)

    # hacky heuristic to fetch most interesting before hitting the rate limit: longest data rows first
    movie_data['len'] = movie_data.apply(lambda s: len(json.dumps(s.to_dict())), axis=1)
    movie_data = movie_data.sort_values('len', ascending=False).reset_index()
    #movie_data = movie_data.truncate(after=10)

    movie_data['omdb'] = movie_data['imdb_id'].apply(get_omdb_data_cache)
    movie_data = movie_data[movie_data['omdb'].notnull()]

    # extract the data we care about...
    movie_data['omdb_genres'] = movie_data['omdb'].apply(lambda d: d['Genre'].split(', '))
    movie_data['omdb_plot'] = movie_data['omdb'].apply(lambda d: d['Plot'])
    movie_data['omdb_awards'] = movie_data['omdb'].apply(lambda d: d['Awards'])

    movie_data = movie_data[['imdb_id', 'omdb_genres', 'omdb_plot', 'omdb_awards']]
    movie_data.to_json('./omdb-data.json.gz', orient='records', lines=True, compression='gzip')


if __name__ == '__main__':
    main()